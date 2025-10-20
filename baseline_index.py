# build_baseline_index.py
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
import ollama
import re

# -------- CONFIG --------
BASELINE_DIR   = Path("baseline_kb")                         # curated base facts (txt/json)
ZERO_MEM_DIR   = Path("memories_by_chapter") / "0-memories"  # hand-written 0-memories as .txt in subfolders
SNAPSHOT_DIR   = Path("snapshots") / "ch-000"                # output snapshot dir
EMBED_MODEL    = "nomic-embed-text"                          # ollama pull nomic-embed-text
MAX_CHARS      = 4000                                        # cap per doc for embedding

# -------- Regex helpers --------
YAML_FM_RE      = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n", re.DOTALL)              # YAML FM at top
KV_RE           = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*:\s*(.+?)\s*$")      # simple key: value
JSON_FENCE_RE   = re.compile(r"^\s*```(?:json)?\s*\n(\{.*?\})\s*\n```", re.DOTALL | re.IGNORECASE)
JSON_BLOCK_RE   = re.compile(r"^\s*(\{.*?\})\s*(?:\n{2,}|$)", re.DOTALL)
SEP_RE          = re.compile(r"\n\s*={4,}\s*\n")  # ONLY '====' as multi-snippet separator

def _str2bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "on"}

def _parse_front_matter(block: str) -> Dict[str, Any]:
    """
    Accepts mini-YAML style 'key: value' lines. No PyYAML dependency.
    """
    meta: Dict[str, Any] = {}
    for line in block.splitlines():
        m = KV_RE.match(line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        if key in {"omniscient", "pinned"}:
            meta[key] = _str2bool(val)
        else:
            meta[key] = val
    return meta

def _split_snippets(raw: str) -> List[str]:
    """
    Split a file into multiple snippets using ONLY lines of '===='.
    Avoids collision with YAML '---' fences.
    """
    txt = raw.replace("\r\n", "\n")
    parts = SEP_RE.split(txt)
    return [p.strip() for p in parts if p.strip()]

def _parse_json_front_matter(snippet: str) -> Tuple[Dict[str, Any] | None, str]:
    """
    Supports either fenced ```json { ... } ``` or a bare { ... } object at the top
    followed by a blank line or EOF.
    """
    m = JSON_FENCE_RE.match(snippet)
    if m:
        try:
            meta = json.loads(m.group(1))
            body = snippet[m.end():].strip()
            return meta, body
        except Exception:
            pass
    m = JSON_BLOCK_RE.match(snippet)
    if m:
        try:
            meta = json.loads(m.group(1))
            body = snippet[m.end():].strip()
            return meta, body
        except Exception:
            pass
    return None, snippet

def _extract_meta_and_body(snippet: str, src_path: Path, idx: int) -> Tuple[Dict[str, Any], str]:
    """
    Return (meta, body_text). Priority:
      1) fenced JSON front-matter ```json {...}```
      2) bare JSON object at top { ... }
      3) YAML front-matter --- ... ---
      4) simple key:value header lines until the first blank line
    """
    meta: Dict[str, Any] = {}
    body = snippet

    # 1–2) JSON front-matter
    jmeta, jbody = _parse_json_front_matter(snippet)
    if jmeta is not None:
        meta = jmeta
        body = jbody
    else:
        # 3) YAML front-matter
        m = YAML_FM_RE.match(snippet)
        if m:
            fm = m.group(1)
            meta = _parse_front_matter(fm)
            body = snippet[m.end():].strip()
        else:
            # 4) simple KV header until blank line
            header_lines, rest_lines, seen_blank = [], [], False
            for line in snippet.splitlines():
                if not seen_blank and line.strip() and KV_RE.match(line):
                    header_lines.append(line)
                else:
                    seen_blank = True
                    rest_lines.append(line)
            if header_lines:
                meta |= _parse_front_matter("\n".join(header_lines))
                body = "\n".join(rest_lines).strip()

    # Defaults (folder will override tag)
    meta.setdefault("type", "world_rule")
    meta.setdefault("omniscient", True)
    meta.setdefault("pinned", True)
    meta.setdefault("id", f"zero:{src_path.as_posix()}#{idx+1}")

    return meta, body

# -------- I/O helpers --------
def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _embed(text: str) -> np.ndarray:
    text = (text or "")[:MAX_CHARS]
    e = ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
    v = np.asarray(e, dtype="float32")
    n = np.linalg.norm(v)
    if n > 0:
        v /= n  # cosine via normalized inner-product
    return v

def _gather_baseline_docs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not BASELINE_DIR.exists():
        return out
    for p in BASELINE_DIR.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".txt":
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
            if not txt:
                continue
            out.append({
                "id": f"base:{p.as_posix()}",
                "text": txt[:MAX_CHARS],
                "chapter": 0,
                "tag": "baseline",
                "type": "world_rule",
                "omniscient": True,
                "pinned": True,
                "source_path": p.as_posix(),
            })
        elif p.suffix.lower() == ".json":
            obj = _read_json(p)
            txt = (obj.get("text") or obj.get("body") or obj.get("content") or obj.get("summary"))
            if not isinstance(txt, str) or not txt.strip():
                txt = json.dumps(obj, ensure_ascii=False)
            out.append({
                "id": obj.get("id") or f"base:{p.as_posix()}",
                "text": txt.strip()[:MAX_CHARS],
                "chapter": 0,
                "tag": obj.get("tag", "baseline"),
                "type": obj.get("type", "world_rule"),
                "omniscient": bool(obj.get("omniscient", True)),
                "pinned": bool(obj.get("pinned", True)),
                "source_path": p.as_posix(),
            })
    return out

def _tag_from_subdir(path_under_zero: Path) -> str:
    """
    Derive tag from immediate subdirectory under 0-memories.
    0-memories/characters/foo.txt -> 'characters'
    """
    rel = path_under_zero.relative_to(ZERO_MEM_DIR)
    if len(rel.parts) < 2:
        return "baseline"
    tag = rel.parts[0]
    return tag.strip().lower().replace(" ", "_")

def _gather_zero_mem_texts() -> List[Dict[str, Any]]:
    """
    Read .txt files under 0-memories/**.txt.
    Each file may contain multiple snippets separated by '===='.
    Tag is taken from the immediate subdirectory (characters/locations/relations/world_rules/...).
    Any 'tag' from front-matter is ignored; folder wins.
    """
    out: List[Dict[str, Any]] = []
    if not ZERO_MEM_DIR.exists():
        return out

    for p in ZERO_MEM_DIR.rglob("*.txt"):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        snippets = _split_snippets(raw)
        folder_tag = _tag_from_subdir(p)

        for i, sn in enumerate(snippets):
            meta, body = _extract_meta_and_body(sn, p, i)
            if not body:
                continue

            # Enforce tag from folder
            meta["tag"] = folder_tag

            out.append({
                "id": meta["id"],
                "text": body[:MAX_CHARS],
                "chapter": 0,
                "tag": meta["tag"],
                "type": meta["type"],
                "omniscient": bool(meta["omniscient"]),
                "pinned": bool(meta["pinned"]),
                "canonical_name": meta.get("canonical_name"),
                "relationship_key": meta.get("relationship_key"),
                "entity_type": meta.get("entity_type"),
                "source_path": f"{p.as_posix()}#{i+1}",
            })
    return out

def main():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    corpus: List[Dict[str, Any]] = []
    corpus += _gather_baseline_docs()
    corpus += _gather_zero_mem_texts()

    if not corpus:
        print("[baseline] Nothing to index. Put .txt in memories_by_chapter/0-memories/<tag>/ or files in baseline_kb/.")
        return

    vecs = []
    for item in corpus:
        try:
            vecs.append(_embed(item["text"]))
        except Exception as e:
            print(f"[baseline] embed failed for {item.get('id')} ({e}); skipping")
    if not vecs:
        print("[baseline] No embeddings produced.")
        return

    X = np.vstack(vecs).astype("float32")
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)

    # Write snapshot files compatible with the runner and labeller/extractor
    faiss.write_index(index, str(SNAPSHOT_DIR / "faiss.index"))
    (SNAPSHOT_DIR / "meta.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "chapter": 0,
        "entries": len(corpus),
        "embedding_model": EMBED_MODEL,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "baseline_dir": BASELINE_DIR.as_posix(),
            "zero_memories_dir": ZERO_MEM_DIR.as_posix()
        }
    }
    (SNAPSHOT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[baseline] 0-snapshot built → {SNAPSHOT_DIR} ({len(corpus)} docs)")

if __name__ == "__main__":
    main()
