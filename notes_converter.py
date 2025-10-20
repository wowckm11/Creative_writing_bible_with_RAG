# auto_label_notes_ch0.py
# Convert freeform notes into baseline 0-memories without RAG.
# Drop .txt files into notes_inbox/. Each file may have multiple snippets separated by --- or ====.

import re, json, uuid, shutil
from pathlib import Path
from typing import List, Dict, Any
import ollama

# ---------------- CONFIG ----------------
INBOX_DIR      = Path("notes_inbox")
ZERO_MEM_ROOT  = Path("memories_by_chapter") / "0-memories"
LLM_MODEL      = "qwen3:8b"
TEMPERATURE    = 0.0
MAX_NOTE_CHARS = 1200
DRY_RUN        = False

ALLOWED_TAGS   = {"characters", "locations", "relations", "world_rules"}
DEFAULT_TYPE   = "world_rule"
DEFAULT_OMNI   = True
DEFAULT_PINNED = True

# ---------------- UTIL ----------------
SEP_RE = re.compile(r"\n\s*(?:---+|===+)\s*\n")
NAME_RE = re.compile(r"[A-Z][A-Za-z\-']+")

def split_snippets(text: str) -> List[str]:
    txt = text.replace("\r\n", "\n").strip()
    parts = SEP_RE.split(txt)
    return [p.strip() for p in parts if p.strip()]

def infer_hints_from_filename(p: Path) -> Dict[str, str]:
    base = p.stem
    names = NAME_RE.findall(base)
    if len(names) >= 2:
        return {"relationship_key": f"{names[0]}↔{names[1]}"}
    elif len(names) == 1:
        return {"canonical_name": names[0]}
    return {}

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9\-]+", "-", s.lower()).strip("-")

def _remove_think_blocks(s: str) -> str:
    # nukes <think>...</think> (multi-line, case-insensitive)
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s

# ---------------- LLM CALL (NO RAG) ----------------
SCHEMA = """{
  "tag": "characters|locations|relations|world_rules",
  "type": "world_rule|trait|relationship_baseline|fact|event",
  "entity_type": "character|location|relationship|concept",
  "canonical_name": string|null,
  "relationship_key": string|null,
  "text": "1–2 sentences, <=45 words, atomic, from the note only",
  "omniscient": true|false,
  "pinned": true|false
}"""

SYSTEM = (
    "You convert a short note into ONE atomic, verifiable fact for a baseline knowledge base.\n"
    "No external knowledge. No inventions. JSON only."
)

def ask_model_to_label(note: str, filename_hints: Dict[str,str]) -> Dict[str, Any]:
    hint_lines = [f"{k}: {v}" for k, v in filename_hints.items()]
    hints = "\n".join(hint_lines)
    user = (
    f"Note:\n{note[:MAX_NOTE_CHARS]}"
    + ("\n\nFilename hints:\n" + hints if hints else "")
    + "\n\nOutput JSON schema:\n" + SCHEMA +
    "\n\nRules:\n"
    "- The 'canonical_name' field must be the **main entity** this note is about — "
    "for example, 'Aura' for a note describing her traits, or 'Grainwick' for a note about a village.\n"
    "- Do NOT use the filename as canonical_name unless it directly matches the entity described in the text.\n"
    "- If the note describes a relationship between two entities, fill 'relationship_key' as 'Entity1↔Entity2' and leave canonical_name null.\n"
    "- If it's about a general rule or concept, leave both canonical_name and relationship_key null.\n"
    "- Prefer concise, verifiable phrasing; no speculation.\n"
    "- If unsure, default tag=world_rules, type=world_rule, entity_type=concept.\n"
    "- Return JSON only, no prose, no code fences, no extra keys."
    )
    res = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        options={"temperature": TEMPERATURE},
    )
    content = res["message"]["content"]
    content = _strip_code_fences(_remove_think_blocks(content))
    return json.loads(content)

# ---------------- NORMALIZE/VALIDATE ----------------
def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    tag = (rec.get("tag") or "").strip().lower().replace(" ", "_")
    if tag not in ALLOWED_TAGS:
        tag = "world_rules"
    t = (rec.get("type") or DEFAULT_TYPE).strip()
    et = (rec.get("entity_type") or "concept").strip()
    txt = (rec.get("text") or "").strip()
    txt = re.sub(r"\s+", " ", txt)[:300]

    out = {
        "tag": tag,
        "type": t,
        "entity_type": et,
        "canonical_name": rec.get("canonical_name"),
        "relationship_key": rec.get("relationship_key"),
        "text": txt,
        "omniscient": bool(rec.get("omniscient", DEFAULT_OMNI)),
        "pinned": bool(rec.get("pinned", DEFAULT_PINNED)),
    }
    return out

def validate_record(out: Dict[str, Any]) -> List[str]:
    errs = []
    if out["tag"] not in ALLOWED_TAGS:
        errs.append(f"bad tag: {out['tag']}")
    if not out["text"]:
        errs.append("empty text")
    if len(out["text"].split()) > 45:
        errs.append("too long (>45 words)")
    return errs

# ---------------- WRITE ----------------
def write_zero_memory(rec: Dict[str, Any], src_path: Path, snippet_idx: int) -> Path:
    tag_dir = ZERO_MEM_ROOT / rec["tag"]
    tag_dir.mkdir(parents=True, exist_ok=True)

    base = src_path.stem
    name_hint = rec.get("canonical_name") or rec.get("relationship_key") or base
    fname = f"{slugify(name_hint)}-{slugify(rec['type'])}-{uuid.uuid4().hex[:6]}.txt"
    out_path = tag_dir / fname

    front = []
    # tag comes from folder; don’t write tag in front-matter
    front.append(f"type: {rec['type']}")
    front.append(f"entity_type: {rec['entity_type']}")
    if rec.get("canonical_name"):
        front.append(f"canonical_name: {rec['canonical_name']}")
    if rec.get("relationship_key"):
        front.append(f"relationship_key: {rec['relationship_key']}")
    front.append(f"omniscient: {str(rec['omniscient']).lower()}")
    front.append(f"pinned: {str(rec['pinned']).lower()}")

    body = rec["text"]
    content = "---\n" + "\n".join(front) + "\n---\n" + body + "\n"
    if not DRY_RUN:
        out_path.write_text(content, encoding="utf-8")
    return out_path

# ---------------- MAIN ----------------
def process_inbox():
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INBOX_DIR.glob("*.txt"))
    if not files:
        print("[auto] No notes found in notes_inbox/")
        return

    for p in files:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        snippets = split_snippets(raw)
        hints = infer_hints_from_filename(p)

        if not snippets:
            print(f"[auto] empty note: {p.name}")
            continue

        for i, sn in enumerate(snippets):
            note = sn.strip()
            try:
                rec = ask_model_to_label(note, hints)
            except Exception as e:
                print(f"[auto] LLM parse error in {p.name}#{i+1}: {e}")
                continue

            # keep to schema fields we care about
            rec = {k: rec.get(k) for k in (
                "tag","type","entity_type","canonical_name","relationship_key","text","omniscient","pinned"
            )}
            norm = normalize_record(rec)
            errs = validate_record(norm)
            if errs:
                print(f"[auto] rejected {p.name}#{i+1}: {errs}")
                continue

            out_path = write_zero_memory(norm, p, i)
            print(f"[auto] wrote {out_path}")

        # move processed file aside
        done_dir = INBOX_DIR / "_processed"
        done_dir.mkdir(exist_ok=True)
        if not DRY_RUN:
            shutil.move(str(p), done_dir / p.name)

if __name__ == "__main__":
    try:
        process_inbox()
    except Exception as e:
        print(f"[FATAL] {e}")
