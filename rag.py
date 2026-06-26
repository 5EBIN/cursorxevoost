"""RAG layer — semantic search over the genuinely textual data (OSM amenities).

Hybrid rule (per INTEGRATION.md): numbers -> pandas (scoring.py); meaning/text -> RAG.
This only embeds free-text columns so fuzzy queries like "family-friendly cultural
spots near a park" work, while all numeric ranking stays in fast deterministic pandas.

Embeddings:
  * If OPENAI_API_KEY is set -> OpenAI 'text-embedding-3-small'.
  * Otherwise -> Chroma's built-in local ONNX model (all-MiniLM-L6-v2): no key,
    fully offline ("demo insurance").

The index is persisted under backend/vectorstore/ so restarts are instant.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

import data_loader

load_dotenv()

STORE = Path(__file__).resolve().parent / "vectorstore"
COLLECTION = "amenities"

_collection = None  # cached chroma collection


def _embedding_function():
    """OpenAI embeddings when a key is available, else a local keyless model."""
    from chromadb.utils import embedding_functions

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    # Local, offline default (downloads a small ONNX model once).
    return embedding_functions.DefaultEmbeddingFunction()


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    import chromadb

    STORE.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(STORE))
    _collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def build_index(force: bool = False) -> Dict[str, Any]:
    """Embed the OSM amenities into the vector store. Idempotent unless force=True."""
    col = _get_collection()
    if col.count() > 0 and not force:
        return {"status": "exists", "count": col.count()}

    if force and col.count() > 0:
        # Recreate the collection cleanly.
        import chromadb

        client = chromadb.PersistentClient(path=str(STORE))
        client.delete_collection(COLLECTION)
        global _collection
        _collection = None
        col = _get_collection()

    am = data_loader.get("amenities")

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for _, r in am.iterrows():
        name = str(r.get("name") or "").strip() or "(unnamed)"
        category = str(r.get("category") or "")
        subtype = str(r.get("subtype") or "")
        district = str(r.get("district") or "")
        ids.append(str(r["amenity_id"]))
        documents.append(f"{name} — {subtype or category} ({category}) in {district}")
        metadatas.append(
            {
                "district": district,
                "category": category,
                "subtype": subtype,
                "name": name,
                "lat": float(r["latitude"]),
                "lng": float(r["longitude"]),
            }
        )

    # Add in batches to stay well under Chroma's max batch size.
    batch = 1000
    for i in range(0, len(ids), batch):
        col.add(
            ids=ids[i : i + batch],
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )

    return {"status": "built", "count": col.count()}


def ensure_index() -> None:
    """Build the index lazily on first use if it's empty."""
    col = _get_collection()
    if col.count() == 0:
        build_index()


def search(query: str, k: int = 8, district: Optional[str] = None) -> Dict[str, Any]:
    """Semantically rank amenity text hits; optionally filter by district."""
    ensure_index()
    col = _get_collection()

    where = {"district": district} if district else None
    res = col.query(query_texts=[query], n_results=k, where=where)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(docs)

    hits: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        hit = {"text": doc}
        hit.update(meta)
        if dist is not None:
            hit["score"] = round(1 - float(dist), 4)  # cosine distance -> similarity
        hits.append(hit)

    return {"query": query, "district": district, "count": len(hits), "results": hits}
