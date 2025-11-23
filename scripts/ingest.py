from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "approved_sections.jsonl"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "sred_reports"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_text(record: Dict[str, Any]) -> str:
    title = record.get("project_title") or ""
    section = record.get("section") or ""
    text = record.get("text") or ""
    parts = []
    if title:
        parts.append(title)
    if section:
        parts.append(f"[{section}]")
    if text:
        parts.append(text)
    return "\n\n".join(parts)


def main() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records(PROCESSED_PATH)
    if not records:
        raise RuntimeError(f"No records found in {PROCESSED_PATH}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(allow_reset=True),
    )

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        doc_id = f"{record.get('report_id','000')}-{record.get('section','unknown')}-{idx}"
        ids.append(doc_id)
        texts.append(build_text(record))
        metadatas.append(
            {
                "report_id": record.get("report_id"),
                "project_title": record.get("project_title"),
                "status": record.get("status"),
                "industry": record.get("industry"),
                "tech_code": record.get("tech_code"),
                "section": record.get("section"),
                "source_path": record.get("source_path"),
            }
        )

    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )


if __name__ == "__main__":
    main()

