"""Module 1: gene features (X). One row per knocked-down gene.

- ``fetch_sequences``: gene symbol -> canonical human protein sequence (UniProt, reviewed).
- ``esm2_embed``: protein sequence -> one vector (mean over residues of ESM-2's last layer).
- ``random_features``: same-shaped random vectors, the "is ESM-2 better than noise?" control.

Heavy dependencies (requests, torch, transformers) are imported inside functions.
"""
from __future__ import annotations

import numpy as np


def fetch_sequences(symbols, timeout: int = 30) -> dict[str, str]:
    """Canonical human protein sequence per gene symbol from UniProt (reviewed entries).

    Genes with no reviewed entry (e.g. non-coding) are left out; log how many.
    """
    import requests

    seqs = {}
    for s in symbols:
        r = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={"query": f"gene_exact:{s} AND organism_id:9606 AND reviewed:true",
                    "format": "fasta", "size": 1},
            timeout=timeout,
        )
        lines = r.text.strip().splitlines()
        if lines:
            seqs[s] = "".join(lines[1:])
    print(f"sequences found for {len(seqs)} of {len(symbols)} genes")
    return seqs


def esm2_embed(seqs: dict[str, str], model_name: str = "facebook/esm2_t6_8M_UR50D",
               max_len: int = 1022) -> dict[str, np.ndarray]:
    """One vector per protein: mean of ESM-2's per-residue vectors (320 numbers for the 8M model).

    Proteins longer than ``max_len`` are truncated; record how many in your log.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    out = {}
    with torch.no_grad():
        for g, seq in seqs.items():
            batch = tok(seq[:max_len], return_tensors="pt")
            h = model(**batch).last_hidden_state[0, 1:-1]   # drop start/end tokens
            out[g] = h.mean(0).numpy()
    return out


def random_features(genes, dim: int, seed: int = 0) -> dict[str, np.ndarray]:
    """A fixed random vector per gene, same width as the real features.

    If ESM-2 cannot beat this, the model is learning "which gene is which", not biology.
    """
    rng = np.random.default_rng(seed)
    return {g: rng.normal(size=dim) for g in sorted(set(genes))}


def stack(emb: dict[str, np.ndarray], genes) -> tuple[np.ndarray, np.ndarray]:
    """Rows of X in the order of ``genes``; returns (X, mask of genes that had a vector)."""
    has = np.array([g in emb for g in genes])
    X = np.vstack([emb[g] for g, h in zip(genes, has) if h])
    return X, has
