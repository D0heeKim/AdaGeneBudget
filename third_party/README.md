# Third-party model repositories

The official scGPT and Geneformer source trees are **not** vendored in this repository.
Clone the exact upstream revisions used for the paper and either place them at the repository root or create local symlinks:

```bash
ln -s /path/to/scGPT ./scGPT
ln -s /path/to/Geneformer ./Geneformer
```

The root `.gitignore` excludes these paths, so the symlinks/clones are not committed.

Before public release, record the exact upstream URLs and commit hashes used in the experiments here.
