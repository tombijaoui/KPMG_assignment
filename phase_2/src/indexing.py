from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.auth import get_text_embedding_ada_config
from config.logger import get_logger
from phase_2.src.chunking import KnowledgeChunk, build_embedded_chunks
from phase_2.src.constants import (
    DEFAULT_KB_DIR,
    FAISS_INDEX_FILENAME,
    KB_CHUNKS_FILENAME,
    KB_MANIFEST_FILENAME,
)

logger = get_logger(__name__)


def _embeddings_matrix(chunks: list[KnowledgeChunk]) -> np.ndarray:
    missing = [chunk.chunk_id for chunk in chunks if chunk.embedding is None]

    if missing:
        raise ValueError(
            f"{len(missing)} chunk(s) missing embeddings (e.g. {missing[0]}). Run embed_chunks first."
        )

    return np.array([chunk.embedding for chunk in chunks], dtype=np.float32)


def build_faiss_index(chunks: list[KnowledgeChunk]) -> faiss.Index:
    """Build a cosine-similarity index (L2-normalized vectors + inner product)."""
    vectors = _embeddings_matrix(chunks)
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    logger.info("Built FAISS IndexFlatIP — %s vectors, dimension=%s", index.ntotal, dimension)
    return index


def save_faiss_knowledge_base(chunks: list[KnowledgeChunk], index: faiss.Index, *, out_dir: Path | None = None, embedding_model: str | None = None) -> Path:
    """Persist FAISS index, chunk metadata, and manifest under ``out_dir``."""
    target_dir = out_dir or DEFAULT_KB_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_text_embedding_ada_config()
    model_name = embedding_model or cfg.model_name
    dimension = int(index.d)

    index_path = target_dir / FAISS_INDEX_FILENAME
    chunks_path = target_dir / KB_CHUNKS_FILENAME
    manifest_path = target_dir / KB_MANIFEST_FILENAME

    faiss.write_index(index, str(index_path))

    with chunks_path.open("w", encoding="utf-8") as handle:
        json.dump([chunk.to_store_dict() for chunk in chunks], handle, ensure_ascii=False, indent=2)

    manifest = {
        "embedding_model": model_name,
        "dimension": dimension,
        "chunk_count": len(chunks),
        "index_type": "IndexFlatIP",
        "vectors_normalized": True,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    logger.info(
        "Saved knowledge base to %s (%s chunks, %s)",
        target_dir,
        len(chunks),
        index_path.name,
    )
    return target_dir


def build_and_save_faiss_index(chunks: list[KnowledgeChunk], *, out_dir: Path | None = None) -> Path:
    """Build FAISS from chunk embeddings and write index + metadata to disk."""
    index = build_faiss_index(chunks)
    return save_faiss_knowledge_base(chunks, index, out_dir=out_dir)


def load_faiss_knowledge_base(kb_dir: Path | None = None) -> tuple[faiss.Index, list[dict[str, Any]], dict[str, Any]]:
    """Load FAISS index, stored chunks, and manifest (for API / retrieval)."""
    target_dir = kb_dir or DEFAULT_KB_DIR
    index_path = target_dir / FAISS_INDEX_FILENAME
    chunks_path = target_dir / KB_CHUNKS_FILENAME
    manifest_path = target_dir / KB_MANIFEST_FILENAME

    for path in (index_path, chunks_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Knowledge base file missing: {path}")

    index = faiss.read_index(str(index_path))

    with chunks_path.open(encoding="utf-8") as handle:
        stored_chunks = json.load(handle)

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    logger.info(
        "Loaded knowledge base from %s — %s vectors, %s chunks",
        target_dir,
        index.ntotal,
        len(stored_chunks),
    )

    return index, stored_chunks, manifest


def run_indexing_pipeline(html_dir: Path | None = None, kb_dir: Path | None = None) -> Path:
    """Parse HTML, embed chunks, build FAISS index, and save to ``knowledge_base``."""
    chunks = build_embedded_chunks(html_dir)
    return build_and_save_faiss_index(chunks, out_dir=kb_dir)