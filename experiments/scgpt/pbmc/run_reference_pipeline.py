#!/usr/bin/env python3
from __future__ import annotations

import argparse, gc, importlib.util, json, random, sys, time
from pathlib import Path
from typing import Any, Iterator

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Sampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("./data/raw/pbmc_seurat_v4/pbmc_seurat_v4_rna_only.h5ad"))
    p.add_argument("--split", type=Path, default=Path("./data/processed/pbmc_seurat_v4/p5_holdout/split_indices.npz"))
    p.add_argument("--cache-dir", type=Path, default=Path("./data/processed/pbmc_seurat_v4/p5_holdout/scgpt_cache"))
    p.add_argument("--output-dir", type=Path, default=Path("./outputs/scgpt/pbmc_p5/main"))
    p.add_argument("--repo", type=Path, default=Path("./scGPT"))
    p.add_argument("--model-dir", type=Path, default=Path("./checkpoints/scgpt"))
    p.add_argument("--full-script", type=Path, default=Path("./experiments/scgpt/_shared/full_input.py"))
    p.add_argument("--fixed-script", type=Path, default=Path("./experiments/scgpt/_shared/fixed_selection.py"))
    p.add_argument("--corrected-script", type=Path, default=Path("./experiments/scgpt/_shared/runtime_utils.py"))
    p.add_argument("--label-col", default="celltype.l1")
    p.add_argument("--secondary-label-col", default="celltype.l2")
    p.add_argument("--time-col", default="time")
    p.add_argument("--tau", type=float, default=0.90)
    p.add_argument("--k-min", type=int, default=128)
    p.add_argument("--k-max", type=int, default=600)
    p.add_argument("--native-max-length", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--timing-repeats", type=int, default=5)
    p.add_argument("--quality-seeds", type=int, nargs="+", default=[0,1,2,3,4])
    p.add_argument("--reference-cells", type=int, default=20000)
    p.add_argument("--matrix-chunk-rows", type=int, default=2048)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--neighbor-k", type=int, default=10)
    p.add_argument("--similarity-chunk", type=int, default=512)
    p.add_argument("--overwrite-methods", action="store_true")
    p.add_argument("--rebuild-matrix-cache", action="store_true")
    p.add_argument("--rebuild-selection-cache", action="store_true")
    return p.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)
    def __iter__(self) -> Iterator[int]:
        return iter(self.indices.tolist())
    def __len__(self):
        return int(self.indices.size)


class CollatorWithOriginalIDs:
    def __init__(self, collator):
        self.collator = collator
    def __call__(self, examples):
        ids = torch.as_tensor(
            [int(x["id"].item()) if torch.is_tensor(x["id"]) else int(x["id"]) for x in examples],
            dtype=torch.long,
        )
        batch = self.collator(examples)
        batch["original_id"] = ids
        return batch


class RemappedSubsetDataset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        lengths = np.asarray(base.sequence_lengths, dtype=np.int64)
        self.sequence_lengths = lengths[self.indices]
        self.gene_lengths = self.sequence_lengths - 1
    def __len__(self):
        return int(self.indices.size)
    def __getitem__(self, i):
        item = dict(self.base[int(self.indices[i])])
        item["id"] = torch.tensor(i, dtype=torch.long)
        return item


class PreselectedTokenDataset(Dataset):
    def __init__(self, cache_dir, gene_ids, cls_id, cls_value):
        self.indptr = np.load(cache_dir / "indptr.npy", mmap_mode="r")
        self.gene_pos = np.load(cache_dir / "gene_pos.npy", mmap_mode="r")
        self.values = np.load(cache_dir / "values.npy", mmap_mode="r")
        self.gene_ids = np.asarray(gene_ids, dtype=np.int64)
        self.cls_id = int(cls_id)
        self.cls_value = float(cls_value)
        self.gene_lengths = np.diff(self.indptr).astype(np.int64, copy=False)
        self.sequence_lengths = self.gene_lengths + 1
    def __len__(self):
        return int(self.indptr.size - 1)
    def __getitem__(self, i):
        s, e = int(self.indptr[i]), int(self.indptr[i+1])
        pos = np.asarray(self.gene_pos[s:e], dtype=np.int64)
        expr = np.asarray(self.values[s:e], dtype=np.float32)
        genes = np.empty(e-s+1, dtype=np.int64)
        vals = np.empty(e-s+1, dtype=np.float32)
        genes[0], vals[0] = self.cls_id, self.cls_value
        genes[1:], vals[1:] = self.gene_ids[pos], expr
        return {
            "id": torch.tensor(i, dtype=torch.long),
            "genes": torch.from_numpy(genes),
            "expressions": torch.from_numpy(vals),
        }


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    s = torch.initial_seed() % (2**32)
    random.seed(s); np.random.seed(s)


def get_lengths(dataset):
    return np.asarray(dataset.sequence_lengths, dtype=np.int64)


def make_loader(dataset, collator, batch_size, num_workers, seed, batching, effective_max_length=None):
    t0 = time.perf_counter()
    lengths = get_lengths(dataset)
    if batching == "length_sorted":
        if effective_max_length is not None:
            lengths = np.minimum(lengths, effective_max_length)
        order = np.argsort(lengths, kind="stable")
    elif batching == "sequential":
        order = np.arange(len(dataset), dtype=np.int64)
    else:
        raise ValueError(batching)
    bucket_time = time.perf_counter() - t0
    g = torch.Generator(); g.manual_seed(seed)
    kw = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=FixedOrderSampler(order),
        collate_fn=CollatorWithOriginalIDs(collator),
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kw["prefetch_factor"] = 2
    return DataLoader(**kw), bucket_time


def materialize_rows(backed_matrix, rows, cols, chunk_rows, name):
    blocks = []
    total = len(rows)
    for s in range(0, total, chunk_rows):
        e = min(s + chunk_rows, total)
        block = backed_matrix[rows[s:e], :]
        if not sparse.issparse(block):
            block = sparse.csr_matrix(np.asarray(block))
        block = block.tocsr()[:, cols].tocsr()
        block.sum_duplicates(); block.eliminate_zeros(); block.sort_indices()
        blocks.append(block)
        print(f"[matrix] {name}: {e:,}/{total:,} ({100*e/total:.1f}%)", flush=True)
    out = sparse.vstack(blocks, format="csr")
    out.sum_duplicates(); out.eliminate_zeros(); out.sort_indices()
    return out


def load_or_build_matrix_cache(args, source, train_idx, test_idx, matched_idx):
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tr = args.cache_dir / "train_matched_csr.npz"
    te = args.cache_dir / "test_matched_csr.npz"
    if tr.exists() and te.exists() and not args.rebuild_matrix_cache:
        print("[matrix] Loading CSR cache", flush=True)
        return sparse.load_npz(tr).tocsr(), sparse.load_npz(te).tocsr()
    train = materialize_rows(source.X, train_idx, matched_idx, args.matrix_chunk_rows, "train")
    test = materialize_rows(source.X, test_idx, matched_idx, args.matrix_chunk_rows, "test")
    print("[matrix] Saving CSR cache", flush=True)
    sparse.save_npz(tr, train, compressed=False)
    sparse.save_npz(te, test, compressed=False)
    return train, test


def compute_idf(train):
    train.sum_duplicates(); train.eliminate_zeros()
    df = np.bincount(train.indices, minlength=train.shape[1]).astype(np.float64)
    return (np.log((train.shape[0]+1)/(df+1))+1).astype(np.float32)


def top_order(scores, top):
    n = scores.size
    if top >= n:
        return np.argsort(-scores, kind="stable")
    chosen = np.argpartition(scores, n-top)[n-top:]
    return chosen[np.argsort(-scores[chosen], kind="stable")]


def build_selection_cache(matrix, idf, cache_dir, method, budget, tau, k_min, k_max, rebuild):
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    required = [cache_dir/"indptr.npy", cache_dir/"gene_pos.npy", cache_dir/"values.npy", meta_path]
    if all(x.exists() for x in required) and not rebuild:
        return json.loads(meta_path.read_text())

    t0 = time.perf_counter()
    pos_parts, val_parts = [], []
    lengths = np.empty(matrix.shape[0], dtype=np.int64)
    retained_sum = 0.0
    upper = lower = 0

    for row in range(matrix.shape[0]):
        s, e = matrix.indptr[row], matrix.indptr[row+1]
        pos = matrix.indices[s:e]
        vals = matrix.data[s:e].astype(np.float32, copy=False)
        n = pos.size

        if n == 0:
            chosen = np.empty(0, dtype=np.int64)
            retained = 1.0
        else:
            scores = vals * idf[pos]
            total = float(scores.sum())
            if method == "fixed_tfidf":
                k = min(n, int(budget))
                chosen = top_order(scores, k)
            elif method == "adaptive":
                order = top_order(scores, min(n, k_max))
                if total <= 0:
                    raw_k = min(n, k_max)
                else:
                    raw_k = int(np.searchsorted(
                        np.cumsum(scores[order], dtype=np.float64),
                        tau * total, side="left"
                    ) + 1)
                if n >= k_min and raw_k < k_min:
                    lower += 1
                if raw_k > k_max:
                    upper += 1
                k = min(n, k_max, max(k_min, raw_k))
                chosen = order[:k]
            else:
                raise ValueError(method)
            retained = float(scores[chosen].sum() / max(total, 1e-12))

        lengths[row] = chosen.size
        pos_parts.append(pos[chosen].astype(np.int32, copy=False))
        val_parts.append(vals[chosen].astype(np.float32, copy=False))
        retained_sum += retained

        if (row+1) % 5000 == 0 or row+1 == matrix.shape[0]:
            print(f"[selection] {cache_dir.name}: {row+1:,}/{matrix.shape[0]:,} "
                  f"({100*(row+1)/matrix.shape[0]:.1f}%)", flush=True)

    indptr = np.empty(matrix.shape[0]+1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])
    gene_pos = np.concatenate(pos_parts) if pos_parts else np.empty(0, dtype=np.int32)
    values = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.float32)
    np.save(cache_dir/"indptr.npy", indptr)
    np.save(cache_dir/"gene_pos.npy", gene_pos)
    np.save(cache_dir/"values.npy", values)

    meta = dict(
        method=method,
        budget=budget,
        tau=tau,
        k_min=k_min,
        k_max=k_max,
        rows=int(matrix.shape[0]),
        mean_selected_genes=float(lengths.mean()),
        median_selected_genes=float(np.median(lengths)),
        selection_time_s=float(time.perf_counter()-t0),
        retained_mass_mean=float(retained_sum/matrix.shape[0]),
        upper_clipped_fraction=float(upper/matrix.shape[0]),
        lower_clipped_fraction=float(lower/matrix.shape[0]),
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def matched_fixed_budget(nnz, target):
    lo, hi = 1, int(nnz.max())
    best_k, best_mean, best_diff = 1, float(np.minimum(nnz,1).mean()), float("inf")
    while lo <= hi:
        mid = (lo+hi)//2
        mean = float(np.minimum(nnz, mid).mean())
        diff = abs(mean-target)
        if diff < best_diff:
            best_k, best_mean, best_diff = mid, mean, diff
        if mean < target:
            lo = mid+1
        else:
            hi = mid-1
    for k in range(max(1,best_k-2), min(int(nnz.max()),best_k+2)+1):
        mean = float(np.minimum(nnz,k).mean())
        if abs(mean-target) < best_diff:
            best_k, best_mean, best_diff = k, mean, abs(mean-target)
    return best_k, best_mean


def stratified_reference_indices(labels, max_cells):
    if len(labels) <= max_cells:
        return np.arange(len(labels), dtype=np.int64)
    s = StratifiedShuffleSplit(n_splits=1, train_size=max_cells, random_state=0)
    idx, _ = next(s.split(np.zeros(len(labels)), labels))
    return np.sort(idx.astype(np.int64))


def exact_cosine_knn_predict(ref_emb, ref_labels, query_emb, k, device, chunk):
    classes, enc = np.unique(ref_labels.astype(str), return_inverse=True)
    ref = F.normalize(torch.from_numpy(ref_emb.astype(np.float32, copy=False)).to(device), dim=1)
    ref_y = torch.from_numpy(enc.astype(np.int64)).to(device)
    out = []
    for s in range(0, len(query_emb), chunk):
        e = min(s+chunk, len(query_emb))
        q = F.normalize(torch.from_numpy(query_emb[s:e].astype(np.float32, copy=False)).to(device), dim=1)
        sim = q @ ref.T
        neigh = sim.topk(k=k, dim=1).indices
        labels = ref_y[neigh]
        votes = F.one_hot(labels, num_classes=len(classes)).sum(1)
        out.append(votes.argmax(1).cpu().numpy())
        del q, sim, neigh, labels, votes
    return classes[np.concatenate(out)]


def exact_self_neighbors(emb, k, device, chunk):
    x = F.normalize(torch.from_numpy(emb.astype(np.float32, copy=False)).to(device), dim=1)
    out = np.empty((len(emb),k), dtype=np.int32)
    for s in range(0, len(emb), chunk):
        e = min(s+chunk, len(emb))
        sim = x[s:e] @ x.T
        local = torch.arange(e-s, device=device)
        global_idx = torch.arange(s,e, device=device)
        sim[local, global_idx] = -torch.inf
        out[s:e] = sim.topk(k=k, dim=1).indices.cpu().numpy().astype(np.int32)
        del sim
    return out


def fidelity_metrics(full, comp, full_neighbors, neighbor_k, device, chunk):
    fn = full / np.clip(np.linalg.norm(full,axis=1,keepdims=True),1e-12,None)
    cn = comp / np.clip(np.linalg.norm(comp,axis=1,keepdims=True),1e-12,None)
    cosine = np.sum(fn*cn,axis=1)
    comp_neighbors = exact_self_neighbors(comp, neighbor_k, device, chunk)
    overlap = (comp_neighbors[:,:,None] == full_neighbors[:,None,:]).any(2).sum(1)/neighbor_k
    return dict(
        embedding_cosine_mean=float(cosine.mean()),
        embedding_cosine_median=float(np.median(cosine)),
        embedding_cosine_min=float(cosine.min()),
        neighbor_recall_10=float(overlap.mean()),
    )


def classification_metrics(y, p):
    return dict(
        accuracy=float(accuracy_score(y,p)),
        macro_f1=float(f1_score(y,p,average="macro",zero_division=0)),
        weighted_f1=float(f1_score(y,p,average="weighted",zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y,p)),
    )


def sample_std(values):
    return float(np.std(values,ddof=1)) if len(values)>1 else 0.0


def load_model(args, fixed_module):
    sys.path.insert(0, str(args.repo))
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    vocab = GeneVocab.from_file(args.model_dir/"vocab.json")
    for token in ("<pad>","<cls>","<eoc>"):
        if token not in vocab:
            vocab.append_token(token)
    cfg = json.loads((args.model_dir/"args.json").read_text())
    pad_token = cfg["pad_token"]
    pad_id = vocab[pad_token]
    pad_value = float(cfg["pad_value"])
    vocab.set_default_index(pad_id)

    model = TransformerModel(
        ntoken=len(vocab), d_model=cfg["embsize"], nhead=cfg["nheads"],
        d_hid=cfg["d_hid"], nlayers=cfg["nlayers"],
        nlayers_cls=cfg["n_layers_cls"], n_cls=1, vocab=vocab,
        dropout=cfg["dropout"], pad_token=pad_token, pad_value=pad_value,
        do_mvc=True, do_dab=False, use_batch_labels=False,
        domain_spec_batchnorm=False, explicit_zero_prob=False,
        use_fast_transformer=False, fast_transformer_backend="flash",
        pre_norm=False,
    )
    ckpt = fixed_module.load_checkpoint(args.model_dir/"best_model.pt")
    load_pretrained(model, ckpt, strict=False, verbose=False)
    device = torch.device("cuda:0")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, vocab, cfg, pad_id, pad_value, device


def save_method(method_dir, summary, seed_rows, timing_rows):
    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir/"summary.json").write_text(json.dumps(summary,indent=2))
    pd.DataFrame(seed_rows).to_csv(method_dir/"seed_metrics.csv",index=False)
    pd.DataFrame(timing_rows).to_csv(method_dir/"timing_repeats.csv",index=False)


def main():
    args = parse_args()
    for path in [
        args.source,args.split,args.repo,
        args.model_dir/"args.json",args.model_dir/"vocab.json",args.model_dir/"best_model.pt",
        args.full_script,args.fixed_script,args.corrected_script,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    print("="*110, flush=True)
    print("PBMC P5-HELD-OUT scGPT SIX-METHOD BENCHMARK", flush=True)
    print("="*110, flush=True)
    print("GPU:",torch.cuda.get_device_name(0), flush=True)

    full_module = load_module("pbmc_full_module",args.full_script)
    fixed_module = load_module("pbmc_fixed_module",args.fixed_script)
    corrected = load_module("pbmc_corrected_module",args.corrected_script)

    sys.path.insert(0, str(args.repo))
    from scgpt.tokenizer import GeneVocab
    vocab_for_match = GeneVocab.from_file(args.model_dir/"vocab.json")
    for token in ("<pad>","<cls>","<eoc>"):
        if token not in vocab_for_match:
            vocab_for_match.append_token(token)

    split = np.load(args.split)
    train_idx = split["train_indices"].astype(np.int64)
    test_idx = split["test_indices"].astype(np.int64)

    source = ad.read_h5ad(args.source,backed="r")
    for col in [args.label_col,args.secondary_label_col,args.time_col]:
        if col not in source.obs:
            raise KeyError(f"Missing obs column: {col}")
    train_labels = source.obs.iloc[train_idx][args.label_col].astype(str).to_numpy()
    test_labels = source.obs.iloc[test_idx][args.label_col].astype(str).to_numpy()

    genes = np.asarray(source.var_names.astype(str))
    matched_mask = np.asarray([g in vocab_for_match for g in genes])
    matched_idx = np.flatnonzero(matched_mask)
    matched_genes = genes[matched_mask]
    np.save(args.output_dir/"matched_genes.npy",matched_genes)
    print(f"Matched genes: {matched_mask.sum()}/{len(matched_mask)} ({matched_mask.mean():.4f})",flush=True)

    train_matrix,test_matrix = load_or_build_matrix_cache(
        args,source,train_idx,test_idx,matched_idx
    )
    source.obs.iloc[test_idx][
        [args.label_col,args.secondary_label_col,args.time_col]
    ].to_csv(args.output_dir/"test_obs.csv")
    source.file.close()

    gc.collect()
    torch.cuda.empty_cache()
    model,vocab,cfg,pad_id,pad_value,device = load_model(args,fixed_module)
    gene_ids = np.asarray(vocab(list(matched_genes)),dtype=np.int64)

    full_train = full_module.FullCellDataset(
        matrix=train_matrix,gene_ids=gene_ids,cls_id=vocab["<cls>"],cls_value=pad_value
    )
    full_test = full_module.FullCellDataset(
        matrix=test_matrix,gene_ids=gene_ids,cls_id=vocab["<cls>"],cls_value=pad_value
    )
    ref_idx = stratified_reference_indices(train_labels,args.reference_cells)
    ref_labels = train_labels[ref_idx]
    np.save(args.output_dir/"reference_train_indices.npy",ref_idx)
    full_train_ref = RemappedSubsetDataset(full_train,ref_idx)

    from scgpt.data_collator import DataCollator
    max_full_length = int(max(
        np.max(full_train.sequence_lengths),np.max(full_test.sequence_lengths)
    ))
    collator_full = DataCollator(
        do_padding=True,pad_token_id=pad_id,pad_value=pad_value,
        do_mlm=False,do_binning=True,max_length=max_full_length,
        sampling=False,keep_first_n_tokens=1,
    )
    collator_native = DataCollator(
        do_padding=True,pad_token_id=pad_id,pad_value=pad_value,
        do_mlm=False,do_binning=True,max_length=args.native_max_length,
        sampling=True,keep_first_n_tokens=1,
    )

    idf_path = args.cache_dir/"idf.npy"
    if idf_path.exists() and not args.rebuild_selection_cache:
        idf = np.load(idf_path)
    else:
        print("[IDF] Computing train-only IDF",flush=True)
        idf = compute_idf(train_matrix)
        np.save(idf_path,idf)

    ada_tr_dir = args.cache_dir/"adaptive_train"
    ada_te_dir = args.cache_dir/"adaptive_test"
    ada_tr_meta = build_selection_cache(
        train_matrix,idf,ada_tr_dir,"adaptive",None,
        args.tau,args.k_min,args.k_max,args.rebuild_selection_cache
    )
    ada_te_meta = build_selection_cache(
        test_matrix,idf,ada_te_dir,"adaptive",None,
        args.tau,args.k_min,args.k_max,args.rebuild_selection_cache
    )

    fixed_k,fixed_expected = matched_fixed_budget(
        np.diff(train_matrix.indptr),ada_tr_meta["mean_selected_genes"]
    )
    print(f"[budget] adaptive train mean={ada_tr_meta['mean_selected_genes']:.4f}; "
          f"matched fixed K={fixed_k}; expected mean={fixed_expected:.4f}",flush=True)

    fix_tr_dir = args.cache_dir/f"fixed_tfidf_k{fixed_k}_train"
    fix_te_dir = args.cache_dir/f"fixed_tfidf_k{fixed_k}_test"
    fix_tr_meta = build_selection_cache(
        train_matrix,idf,fix_tr_dir,"fixed_tfidf",fixed_k,
        args.tau,args.k_min,args.k_max,args.rebuild_selection_cache
    )
    fix_te_meta = build_selection_cache(
        test_matrix,idf,fix_te_dir,"fixed_tfidf",fixed_k,
        args.tau,args.k_min,args.k_max,args.rebuild_selection_cache
    )

    ada_train = PreselectedTokenDataset(ada_tr_dir,gene_ids,vocab["<cls>"],pad_value)
    ada_test = PreselectedTokenDataset(ada_te_dir,gene_ids,vocab["<cls>"],pad_value)
    fix_train = PreselectedTokenDataset(fix_tr_dir,gene_ids,vocab["<cls>"],pad_value)
    fix_test = PreselectedTokenDataset(fix_te_dir,gene_ids,vocab["<cls>"],pad_value)
    ada_train_ref = RemappedSubsetDataset(ada_train,ref_idx)
    fix_train_ref = RemappedSubsetDataset(fix_train,ref_idx)

    collator_random = DataCollator(
        do_padding=True,pad_token_id=pad_id,pad_value=pad_value,
        do_mlm=False,do_binning=True,max_length=fixed_k+1,
        sampling=True,keep_first_n_tokens=1,
    )
    collator_fixed = DataCollator(
        do_padding=True,pad_token_id=pad_id,pad_value=pad_value,
        do_mlm=False,do_binning=True,max_length=fixed_k+1,
        sampling=False,keep_first_n_tokens=1,
    )
    collator_ada = DataCollator(
        do_padding=True,pad_token_id=pad_id,pad_value=pad_value,
        do_mlm=False,do_binning=True,max_length=args.k_max+1,
        sampling=False,keep_first_n_tokens=1,
    )

    specs = [
        dict(name="full_bucketed",train_q=full_train_ref,test_q=full_test,
             train_t=full_train,test_t=full_test,collator=collator_full,
             batching="length_sorted",effective=None,seeds=[0],
             selection_time=0.0,sampling="none",fixed_k=None),
        dict(name="scgpt_native_sequential",train_q=full_train_ref,test_q=full_test,
             train_t=full_train,test_t=full_test,collator=collator_native,
             batching="sequential",effective=args.native_max_length,seeds=args.quality_seeds,
             selection_time=0.0,sampling="random_in_dataloader",fixed_k=args.native_max_length-1),
        dict(name="scgpt_native_bucketed",train_q=full_train_ref,test_q=full_test,
             train_t=full_train,test_t=full_test,collator=collator_native,
             batching="length_sorted",effective=args.native_max_length,seeds=args.quality_seeds,
             selection_time=0.0,sampling="random_in_dataloader",fixed_k=args.native_max_length-1),
        dict(name="matched_random_fixed_k",train_q=full_train_ref,test_q=full_test,
             train_t=full_train,test_t=full_test,collator=collator_random,
             batching="length_sorted",effective=fixed_k+1,seeds=args.quality_seeds,
             selection_time=0.0,sampling="random_in_dataloader",fixed_k=fixed_k),
        dict(name="matched_fixed_tfidf",train_q=fix_train_ref,test_q=fix_test,
             train_t=fix_train,test_t=fix_test,collator=collator_fixed,
             batching="length_sorted",effective=fixed_k+1,seeds=[0],
             selection_time=fix_tr_meta["selection_time_s"]+fix_te_meta["selection_time_s"],
             sampling="fixed_tfidf",fixed_k=fixed_k),
        dict(name="adaptive",train_q=ada_train_ref,test_q=ada_test,
             train_t=ada_train,test_t=ada_test,collator=collator_ada,
             batching="length_sorted",effective=args.k_max+1,seeds=[0],
             selection_time=ada_tr_meta["selection_time_s"]+ada_te_meta["selection_time_s"],
             sampling="adaptive_tfidf_mass",fixed_k=None),
    ]

    summaries = {}
    full_test_emb = None
    full_neighbors = None
    metric_names = [
        "accuracy","macro_f1","weighted_f1","balanced_accuracy",
        "embedding_cosine_mean","embedding_cosine_median",
        "embedding_cosine_min","neighbor_recall_10",
    ]

    for spec in specs:
        name = spec["name"]
        method_dir = args.output_dir/name
        summary_path = method_dir/"summary.json"

        if summary_path.exists() and not args.overwrite_methods:
            print(f"[resume] {name}",flush=True)
            summaries[name] = json.loads(summary_path.read_text())
            if name == "full_bucketed":
                d = np.load(method_dir/"quality_seed0.npz",allow_pickle=True)
                full_test_emb = d["test_embeddings"].astype(np.float32)
                full_neighbors = np.load(method_dir/"full_neighbors_10.npy")
            continue

        print("\n"+"="*110,flush=True)
        print("METHOD:",name,flush=True)
        print("="*110,flush=True)
        seed_rows = []
        store = {k:[] for k in metric_names}

        for seed in spec["seeds"]:
            seed_all(seed)
            tr_loader,_ = make_loader(
                spec["train_q"],spec["collator"],args.batch_size,args.num_workers,
                seed,spec["batching"],spec["effective"]
            )
            te_loader,_ = make_loader(
                spec["test_q"],spec["collator"],args.batch_size,args.num_workers,
                seed+100000,spec["batching"],spec["effective"]
            )
            tr_emb,_ = corrected.extract_once(
                model,tr_loader,spec["train_q"],pad_id,device,cfg["embsize"]
            )
            te_emb,_ = corrected.extract_once(
                model,te_loader,spec["test_q"],pad_id,device,cfg["embsize"]
            )
            pred = exact_cosine_knn_predict(
                tr_emb,ref_labels,te_emb,args.knn_k,device,args.similarity_chunk
            )
            metrics = classification_metrics(test_labels,pred)

            if name == "full_bucketed":
                full_test_emb = te_emb.astype(np.float32)
                full_neighbors = exact_self_neighbors(
                    full_test_emb,args.neighbor_k,device,args.similarity_chunk
                )
                fidelity = dict(
                    embedding_cosine_mean=1.0,embedding_cosine_median=1.0,
                    embedding_cosine_min=1.0,neighbor_recall_10=1.0
                )
            else:
                if full_test_emb is None or full_neighbors is None:
                    raise RuntimeError("Full must run first")
                fidelity = fidelity_metrics(
                    full_test_emb,te_emb,full_neighbors,args.neighbor_k,
                    device,args.similarity_chunk
                )

            row = dict(method=name,seed=int(seed),**metrics,**fidelity)
            seed_rows.append(row)
            for k in store:
                store[k].append(float(row[k]))

            method_dir.mkdir(parents=True,exist_ok=True)
            if seed == spec["seeds"][0]:
                np.savez_compressed(
                    method_dir/f"quality_seed{seed}.npz",
                    train_reference_indices=ref_idx,
                    train_reference_labels=ref_labels,
                    test_labels=test_labels,
                    test_predictions=pred,
                    test_embeddings=te_emb.astype(np.float32),
                )
                if name == "full_bucketed":
                    np.save(method_dir/"full_neighbors_10.npy",full_neighbors)

            print(f"[quality] seed={seed} acc={metrics['accuracy']:.6f} "
                  f"macro={metrics['macro_f1']:.6f} "
                  f"balanced={metrics['balanced_accuracy']:.6f} "
                  f"cos={fidelity['embedding_cosine_mean']:.6f} "
                  f"neighbor={fidelity['neighbor_recall_10']:.6f}",flush=True)
            del tr_emb,te_emb,tr_loader,te_loader
            gc.collect(); torch.cuda.empty_cache()

        seed_all(int(spec["seeds"][0]))
        tr_time_loader,tr_bucket = make_loader(
            spec["train_t"],spec["collator"],args.batch_size,args.num_workers,
            999001,spec["batching"],spec["effective"]
        )
        te_time_loader,te_bucket = make_loader(
            spec["test_t"],spec["collator"],args.batch_size,args.num_workers,
            999002,spec["batching"],spec["effective"]
        )
        bucket_time = tr_bucket+te_bucket
        corrected.warmup(model,tr_time_loader,pad_id,device,args.warmup_steps)
        corrected.warmup(model,te_time_loader,pad_id,device,args.warmup_steps)

        timing_rows=[]; total_times=[]; embedding_times=[]; throughputs=[]; memories=[]; pads=[]
        total_cells = len(spec["train_t"])+len(spec["test_t"])

        for repeat in range(args.timing_repeats):
            tr_emb,tr_t = corrected.extract_once(
                model,tr_time_loader,spec["train_t"],pad_id,device,cfg["embsize"]
            )
            te_emb,te_t = corrected.extract_once(
                model,te_time_loader,spec["test_t"],pad_id,device,cfg["embsize"]
            )
            emb_time = tr_t["embedding_time_s"]+te_t["embedding_time_s"]
            e2e = spec["selection_time"]+bucket_time+emb_time
            throughput = total_cells/e2e
            emb_throughput = total_cells/emb_time
            actual = tr_t["actual_sequence_tokens"]+te_t["actual_sequence_tokens"]
            padded = tr_t["padded_tokens_processed"]+te_t["padded_tokens_processed"]
            pad_ratio = padded/actual
            memory = max(tr_t["peak_gpu_memory_gb"],te_t["peak_gpu_memory_gb"])
            total_times.append(float(e2e)); embedding_times.append(float(emb_time))
            throughputs.append(float(throughput)); memories.append(float(memory)); pads.append(float(pad_ratio))
            timing_rows.append(dict(
                method=name,repeat=repeat,selection_time_s=spec["selection_time"],
                bucketing_time_s=bucket_time,embedding_time_s=emb_time,
                end_to_end_time_s=e2e,cells_per_s=throughput,
                embedding_only_cells_per_s=emb_throughput,
                peak_gpu_memory_gb=memory,padding_overhead_ratio=pad_ratio,
            ))
            print(f"[timing] repeat={repeat+1} e2e={throughput:.2f} cells/s "
                  f"embedding-only={emb_throughput:.2f} memory={memory:.3f}GB "
                  f"padding={pad_ratio:.4f}",flush=True)
            del tr_emb,te_emb
            gc.collect(); torch.cuda.empty_cache()

        train_lengths = get_lengths(spec["train_t"])-1
        test_lengths = get_lengths(spec["test_t"])-1
        if spec["sampling"] == "random_in_dataloader":
            train_lengths = np.minimum(train_lengths,int(spec["fixed_k"]))
            test_lengths = np.minimum(test_lengths,int(spec["fixed_k"]))

        summary = dict(
            method=name,tau=args.tau if name=="adaptive" else None,
            fixed_k=spec["fixed_k"],batching=spec["batching"],sampling=spec["sampling"],
            quality_num_seeds=len(spec["seeds"]),timing_repeats=args.timing_repeats,
            selection_in_dataloader=(spec["sampling"]=="random_in_dataloader"),
            selection_time_s=float(spec["selection_time"]),bucketing_time_s=float(bucket_time),
            train_mean_selected_genes=float(train_lengths.mean()),
            test_mean_selected_genes=float(test_lengths.mean()),
            train_median_selected_genes=float(np.median(train_lengths)),
            test_median_selected_genes=float(np.median(test_lengths)),
            end_to_end_time_mean_s=float(np.mean(total_times)),
            end_to_end_time_std_s=sample_std(total_times),
            embedding_time_mean_s=float(np.mean(embedding_times)),
            cells_per_s_mean=float(np.mean(throughputs)),
            cells_per_s_std=sample_std(throughputs),
            embedding_only_cells_per_s_mean=float(total_cells/np.mean(embedding_times)),
            peak_gpu_memory_gb=float(np.max(memories)),
            padding_overhead_ratio_mean=float(np.mean(pads)),
            matched_fixed_k=fixed_k,
            target_adaptive_train_mean=ada_tr_meta["mean_selected_genes"],
            matched_fixed_expected_mean=fixed_expected,
        )
        for k,v in store.items():
            summary[k]=float(np.mean(v)); summary[k+"_std"]=sample_std(v)

        if name=="adaptive":
            summary.update(
                train_retained_mass_mean=ada_tr_meta["retained_mass_mean"],
                test_retained_mass_mean=ada_te_meta["retained_mass_mean"],
                train_upper_clipped_fraction=ada_tr_meta["upper_clipped_fraction"],
                test_upper_clipped_fraction=ada_te_meta["upper_clipped_fraction"],
                train_lower_clipped_fraction=ada_tr_meta["lower_clipped_fraction"],
                test_lower_clipped_fraction=ada_te_meta["lower_clipped_fraction"],
            )

        if name=="full_bucketed":
            summary["speedup_vs_full_bucketed"]=1.0
            summary["memory_reduction_vs_full_bucketed"]=0.0
        else:
            full_s = summaries["full_bucketed"]
            summary["speedup_vs_full_bucketed"]=summary["cells_per_s_mean"]/full_s["cells_per_s_mean"]
            summary["memory_reduction_vs_full_bucketed"]=1-summary["peak_gpu_memory_gb"]/full_s["peak_gpu_memory_gb"]

        save_method(method_dir,summary,seed_rows,timing_rows)
        summaries[name]=summary
        pd.DataFrame(list(summaries.values())).to_csv(args.output_dir/"summary_partial.csv",index=False)
        del tr_time_loader,te_time_loader
        gc.collect(); torch.cuda.empty_cache()

    order = [
        "full_bucketed","scgpt_native_sequential","scgpt_native_bucketed",
        "matched_random_fixed_k","matched_fixed_tfidf","adaptive",
    ]
    df = pd.DataFrame([summaries[x] for x in order])
    df.to_csv(args.output_dir/"summary.csv",index=False)
    (args.output_dir/"config.json").write_text(json.dumps({
        **{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "matched_genes":int(matched_mask.sum()),"total_genes":int(len(matched_mask)),
        "fixed_k":fixed_k,"reference_cells":int(len(ref_idx)),
    },indent=2))

    cols = [
        "method","test_mean_selected_genes","cells_per_s_mean",
        "embedding_only_cells_per_s_mean","speedup_vs_full_bucketed",
        "peak_gpu_memory_gb","padding_overhead_ratio_mean","accuracy",
        "macro_f1","balanced_accuracy","embedding_cosine_mean","neighbor_recall_10",
    ]
    print("\n"+"="*140,flush=True)
    print("FINAL SIX-METHOD SUMMARY",flush=True)
    print("="*140,flush=True)
    print(df[cols].to_string(index=False),flush=True)
    print("FINAL STATUS: PASS",flush=True)
    print("SUMMARY:",args.output_dir/"summary.csv",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
