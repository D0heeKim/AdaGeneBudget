#!/usr/bin/env python3
from __future__ import annotations

import argparse, gc, importlib.util, json, os, shutil, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

P = Path(".")
MAIN_SCRIPT = P/"experiments/geneformer/pbmc/run_main.py"
BASE = P/"data/processed/pbmc_seurat_v4/p5_holdout/geneformer_cache/base"
METHODS = BASE.parent/"methods"
NATIVE = METHODS/"native_max2048"
NEW_CACHE = METHODS/"native_rank_fixed_k599"
CKPT = Path("./checkpoints/Geneformer-V1-10M")
OUT = P/"outputs/geneformer/pbmc_p5/main"
NEW_OUT = OUT/"native_rank_fixed_k"
METHOD = "native_rank_fixed_k"
LABEL = "Native-Rank Top-K"
K = 599
TRAIN_N, TEST_N = 132137, 19957

def req(p: Path):
    if not p.exists():
        raise FileNotFoundError(p)

def load_mod():
    req(MAIN_SCRIPT)
    spec = importlib.util.spec_from_file_location("pbmc_main", MAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import PBMC main script")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m

def write_json(p: Path, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

def load_native(split: str):
    d = NATIVE/split
    indptr = np.load(d/"indptr.npy", mmap_mode="r")
    gene_pos = np.load(d/"gene_pos.npy", mmap_mode="r")
    n = TRAIN_N if split == "train" else TEST_N
    if indptr.shape != (n+1,) or int(indptr[-1]) != len(gene_pos):
        raise RuntimeError(f"Invalid native cache: {split}")
    return indptr, gene_pos, np.diff(indptr).astype(np.int64, copy=False)

def make_cache(overwrite: bool):
    if NEW_CACHE.exists():
        if not overwrite:
            raise FileExistsError(f"{NEW_CACHE} exists; use --overwrite-new-method")
        shutil.rmtree(NEW_CACHE)
    tmp = NEW_CACHE.with_name(NEW_CACHE.name+".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    root = {"status":"PASS","method":METHOD,"method_label":LABEL,"fixed_k":K,
            "base_native_cache":str(NATIVE.resolve()),"selection_rule":"official native rank top-K"}
    for split in ("train","test"):
        indptr, gene_pos, lengths = load_native(split)
        selected = np.minimum(lengths, K).astype(np.uint16)
        out_indptr = np.empty(len(selected)+1, dtype=np.int64)
        out_indptr[0] = 0
        np.cumsum(selected.astype(np.int64), out=out_indptr[1:])
        sd = tmp/split
        sd.mkdir()
        np.save(sd/"indptr.npy", out_indptr)
        np.save(sd/"selected_k.npy", selected)
        out = np.lib.format.open_memmap(sd/"gene_pos.npy", mode="w+", dtype=np.uint16,
                                        shape=(int(out_indptr[-1]),))
        for i, k in enumerate(selected):
            a = int(indptr[i]); b = int(out_indptr[i]); e = int(out_indptr[i+1])
            out[b:e] = gene_pos[a:a+int(k)]
        out.flush(); del out
        meta = {"status":"PASS","method":METHOD,"split":split,"fixed_k":K,
                "cells":len(selected),"mean_selected_genes":float(selected.mean()),
                "median_selected_genes":float(np.median(selected)),
                "minimum_selected_genes":int(selected.min()),
                "maximum_selected_genes":int(selected.max())}
        write_json(sd/"meta.json", meta)
        root[split] = meta
    write_json(tmp/"meta.json", root)
    os.replace(tmp, NEW_CACHE)
    return root

def selection_time(base_test, gene_median, expected):
    t0 = time.perf_counter()
    total = 0
    for i in range(TEST_N):
        a = int(base_test["indptr"][i]); b = int(base_test["indptr"][i+1])
        pos = np.asarray(base_test["gene_pos"][a:b], dtype=np.int64)
        cnt = np.asarray(base_test["counts"][a:b], dtype=np.float64)
        score = cnt / gene_median[pos]
        order = np.lexsort((pos, -score))
        total += len(order[:min(K, len(pos))])
    elapsed = time.perf_counter()-t0
    if total != expected:
        raise RuntimeError(f"Selection total mismatch: {total} != {expected}")
    return float(elapsed)

def backup(path: Path):
    d = OUT/"backups"; d.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = d/f"{path.stem}_before_{METHOD}_{ts}{path.suffix}"
    shutil.copy2(path, dst)
    return dst

def atomic_csv(df, path):
    tmp = path.with_name(path.name+".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite-new-method", action="store_true")
    args = ap.parse_args()

    for p in [BASE, METHODS, NATIVE, CKPT, OUT, MAIN_SCRIPT,
              OUT/"summary.csv", OUT/"timing_runs.csv",
              OUT/"quality_by_variant.csv", OUT/"per_class_f1.csv",
              OUT/"native_reference_neighbors10.npy"]:
        req(p)

    csvs = {name: pd.read_csv(OUT/name) for name in
            ["summary.csv","timing_runs.csv","quality_by_variant.csv","per_class_f1.csv"]}
    for name, df in csvs.items():
        if METHOD in set(df["method"].astype(str)):
            raise RuntimeError(f"{name} already contains {METHOD}; nothing modified")

    if NEW_OUT.exists():
        if not args.overwrite_new_method:
            raise FileExistsError(f"{NEW_OUT} exists; use --overwrite-new-method")
        shutil.rmtree(NEW_OUT)

    cache_meta = make_cache(args.overwrite_new_method)
    m = load_mod()
    inputs = m.load_inputs(BASE, METHODS)
    cache = {s: m.load_method_split(NEW_CACHE, s) for s in ("train","test")}

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    model, pad = m.load_model(CKPT, device)

    timing = m.measure_embedding_timing(
        model=model, cache=cache["test"], token_ids=inputs["token_ids"],
        pad_token_id=pad, device=device, batch_size=32, warmup_batches=5,
        timing_repeats=3, batching="bucketed")

    base_test = {
        "indptr": np.load(BASE/"test/indptr.npy", mmap_mode="r"),
        "gene_pos": np.load(BASE/"test/gene_pos.npy", mmap_mode="r"),
        "counts": np.load(BASE/"test/counts.npy", mmap_mode="r"),
    }
    expected = int(np.sum(cache["test"]["lengths"], dtype=np.int64))
    sel_time = selection_time(base_test, inputs["gene_median"], expected)

    timing_rows = []
    for r, emb_t in enumerate(timing["embedding_times_s"]):
        e2e = sel_time + timing["bucketing_time_s"] + emb_t
        timing_rows.append({
            "method":METHOD,"seed":np.nan,"repeat":r,"batching":"bucketed",
            "selection_time_s":sel_time,"selection_time_std_s":0.0,
            "bucketing_time_s":timing["bucketing_time_s"],
            "embedding_time_s":emb_t,"end_to_end_time_s":e2e,
            "cells_per_s":TEST_N/e2e,
            "embedding_only_cells_per_s":TEST_N/emb_t,
            "peak_gpu_memory_gb":timing["peak_gpu_memory_gb"],
            "padding_overhead_ratio":timing["padding_overhead_ratio"]})

    NEW_OUT.mkdir(parents=True)
    tr_path, te_path = NEW_OUT/"train_embeddings.npy", NEW_OUT/"test_embeddings.npy"
    m.extract_embeddings_to_file(model, cache["train"], inputs["token_ids"], pad, device, 32, tr_path, False)
    m.extract_embeddings_to_file(model, cache["test"], inputs["token_ids"], pad, device, 32, te_path, False)
    del model; gc.collect(); torch.cuda.empty_cache()

    tr_text = np.asarray(inputs["base_splits"]["train"]["labels_l1"]).astype(str)
    te_text = np.asarray(inputs["base_splits"]["test"]["labels_l1"]).astype(str)
    le = LabelEncoder(); ytr = le.fit_transform(tr_text); yte = le.transform(te_text)
    pred = m.predict_exact_knn_cosine(tr_path, te_path, ytr, device, 512, 5, len(le.classes_))
    metrics = {
        "accuracy":float(accuracy_score(yte,pred)),
        "macro_f1":float(f1_score(yte,pred,average="macro",zero_division=0)),
        "weighted_f1":float(f1_score(yte,pred,average="weighted",zero_division=0)),
        "balanced_accuracy":float(balanced_accuracy_score(yte,pred))}
    native_test = OUT/"geneformer_native_bucketed/test_embeddings.npy"
    cos = m.rowwise_embedding_cosine(native_test, te_path)
    ref_nei = np.load(OUT/"native_reference_neighbors10.npy")
    nei = m.exact_neighbors_cosine(te_path, device, 512, 10)
    r10 = float(m.neighbor_recall(ref_nei, nei))
    pred_text = le.inverse_transform(pred)
    m.save_quality_npz(NEW_OUT/"quality_seed0.npz", inputs, pred_text, te_path)

    quality = pd.DataFrame([{
        "variant_id":"native_rank_fixed_k_seed0","method":METHOD,"method_label":LABEL,"seed":np.nan,
        **metrics,"embedding_cosine_mean":float(cos.mean()),
        "embedding_cosine_median":float(np.median(cos)),
        "embedding_cosine_min":float(cos.min()),"neighbor_recall_10":r10}])

    per_class = []
    for cid, cname in enumerate(le.classes_):
        yt = (yte==cid).astype(int); yp = (pred==cid).astype(int)
        per_class.append({"method":METHOD,"method_label":LABEL,"seed":np.nan,
                          "cell_type":str(cname),"support":int(yt.sum()),
                          "f1":float(f1_score(yt,yp,zero_division=0))})
    per_class = pd.DataFrame(per_class)

    timing_df = pd.DataFrame(timing_rows)
    old_summary = csvs["summary.csv"]
    native = old_summary.loc[old_summary["method"]=="geneformer_native_bucketed"].iloc[0]
    row = {c:np.nan for c in old_summary.columns}
    row.update({
        "method":METHOD,"method_label":LABEL,"cache":str(NEW_CACHE.resolve()),
        "tau":np.nan,"fixed_k":K,"batching":"bucketed","sampling":"native_rank",
        "quality_num_seeds":1,"timing_repeats":3,"selection_timing_repeats":1,
        "selection_in_dataloader":False,"selection_time_s":sel_time,
        "selection_time_std_s":0.0,"bucketing_time_s":timing["bucketing_time_s"],
        "train_mean_selected_genes":float(cache["train"]["lengths"].mean()),
        "test_mean_selected_genes":float(cache["test"]["lengths"].mean()),
        "train_median_selected_genes":float(np.median(cache["train"]["lengths"])),
        "test_median_selected_genes":float(np.median(cache["test"]["lengths"])),
        "end_to_end_time_mean_s":float(timing_df["end_to_end_time_s"].mean()),
        "end_to_end_time_std_s":float(timing_df["end_to_end_time_s"].std(ddof=0)),
        "embedding_time_mean_s":float(timing_df["embedding_time_s"].mean()),
        "embedding_time_std_s":float(timing_df["embedding_time_s"].std(ddof=0)),
        "cells_per_s_mean":float(timing_df["cells_per_s"].mean()),
        "cells_per_s_std":float(timing_df["cells_per_s"].std(ddof=0)),
        "embedding_only_cells_per_s_mean":float(timing_df["embedding_only_cells_per_s"].mean()),
        "embedding_only_cells_per_s_std":float(timing_df["embedding_only_cells_per_s"].std(ddof=0)),
        "peak_gpu_memory_gb":float(timing["peak_gpu_memory_gb"]),
        "padding_overhead_ratio_mean":float(timing["padding_overhead_ratio"]),
        "matched_fixed_k":K,
        "target_adaptive_train_mean":float(old_summary["target_adaptive_train_mean"].dropna().iloc[0]),
        "matched_fixed_expected_mean":float(old_summary["matched_fixed_expected_mean"].dropna().iloc[0]),
        "reference_method":"geneformer_native_bucketed",
        "accuracy":metrics["accuracy"],"accuracy_std":0.0,
        "macro_f1":metrics["macro_f1"],"macro_f1_std":0.0,
        "weighted_f1":metrics["weighted_f1"],"weighted_f1_std":0.0,
        "balanced_accuracy":metrics["balanced_accuracy"],"balanced_accuracy_std":0.0,
        "embedding_cosine_mean":float(cos.mean()),"embedding_cosine_mean_std":0.0,
        "embedding_cosine_median":float(np.median(cos)),"embedding_cosine_median_std":0.0,
        "embedding_cosine_min":float(cos.min()),"embedding_cosine_min_std":0.0,
        "neighbor_recall_10":r10,"neighbor_recall_10_std":0.0})
    row["speedup_vs_native_bucketed"] = float(native["end_to_end_time_mean_s"]/row["end_to_end_time_mean_s"])
    row["memory_reduction_vs_native_bucketed"] = float(1-row["peak_gpu_memory_gb"]/native["peak_gpu_memory_gb"])
    new_summary = pd.DataFrame([row], columns=old_summary.columns)

    backups = {name:str(backup(OUT/name)) for name in csvs}
    atomic_csv(pd.concat([old_summary,new_summary],ignore_index=True), OUT/"summary.csv")
    atomic_csv(pd.concat([csvs["timing_runs.csv"],timing_df],ignore_index=True), OUT/"timing_runs.csv")
    atomic_csv(pd.concat([csvs["quality_by_variant.csv"],quality],ignore_index=True), OUT/"quality_by_variant.csv")
    atomic_csv(pd.concat([csvs["per_class_f1.csv"],per_class],ignore_index=True), OUT/"per_class_f1.csv")
    write_json(NEW_OUT/"result_manifest.json", {"status":"PASS","summary_row":row,"backups":backups})

    print("="*100)
    print("PBMC NATIVE-RANK TOP-599 RESULT")
    for k in ["test_mean_selected_genes","selection_time_s","embedding_time_mean_s",
              "cells_per_s_mean","peak_gpu_memory_gb","accuracy","macro_f1",
              "balanced_accuracy","embedding_cosine_mean","neighbor_recall_10"]:
        print(f"{k}: {row[k]}")
    print("Output:", NEW_OUT)
    print("FINAL STATUS: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
