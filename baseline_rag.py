# baseline_rag.py
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
import ollama

# -------- CONFIG --------
INDEX_DIR   = Path("baseline_index")
EMBED_MODEL = "nomic-embed-text"
CTX_CHAR_BUDGET = 3500   # keep prompts sane

_index = None
_meta  = None

def _ensure_loaded():
    global _index, _meta
    if _index is None or _meta is None:
        _index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        _meta  = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))

def _embed(text: str) -> np.ndarray:
    text = text[:8000]
    e = ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
    v = np.array(e, dtype="float32")
    v /= max(np.linalg.norm(v), 1e-12)
    return v

def search_baseline(query_text: str, top_k: int = 5) -> List[Tuple[float, dict]]:
    _ensure_loaded()
    q = _embed(query_text).reshape(1, -1)
    k = min(top_k, len(_meta))
    D, I = _index.search(q, k)
    out = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1: 
            continue
        out.append((float(score), _meta[idx]))
    return out

def get_baseline_context(query_text: str, top_k: int = 5) -> str:
    """Return a compact text block with top-K baseline snippets, budgeted by CTX_CHAR_BUDGET."""
    hits = search_baseline(query_text, top_k=top_k)
    buf, remain = [], CTX_CHAR_BUDGET
    for score, m in hits:
        preview = m.get("preview","")
        take = preview[:max(0, remain)]
        if not take:
            break
        buf.append(f"[BASELINE {m['path']} | score={score:.3f}]\n{take}")
        remain -= len(take)
        if remain <= 0:
            break
    return "\n\n".join(buf)
