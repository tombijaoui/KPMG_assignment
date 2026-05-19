from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import AzureOpenAI

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.auth import create_text_embedding_ada_client, get_text_embedding_ada_config
from config.logger import get_logger
from phase_2.src.constants import ALLOWED_HMO, ALLOWED_INSURANCE_TIERS
from phase_2.src.models import UserProfile

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by semantic search."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float


def _metadata_values(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]

    return [str(value)]


def chunk_matches_profile(metadata: dict[str, Any], user_profile: UserProfile) -> bool:
    """True if chunk metadata applies to the member's HMO and insurance tier."""
    hmo = user_profile.hmo
    tier = user_profile.insurance_tier

    if not hmo or not tier:
        return False

    if hmo not in ALLOWED_HMO or tier not in ALLOWED_INSURANCE_TIERS:
        return False

    chunk_hmos = _metadata_values(metadata.get("hmo", []))
    chunk_tiers = _metadata_values(metadata.get("tier", []))

    return hmo in chunk_hmos and tier in chunk_tiers


def _filter_chunk_indices(stored_chunks: list[dict[str, Any]], user_profile: UserProfile) -> list[int]:
    """Filter chunk indices by user profile."""
    return [
        index
        for index, chunk in enumerate(stored_chunks)
        if chunk_matches_profile(chunk.get("metadata", {}), user_profile)
    ]


def embed_query(query: str, *, client: AzureOpenAI | None = None, model_name: str | None = None) -> np.ndarray:
    """Embed and L2-normalize a search query (same space as the FAISS index)."""
    stripped = query.strip()

    if not stripped:
        raise ValueError("Query must not be empty")

    cfg = get_text_embedding_ada_config()
    embedding_client = client or create_text_embedding_ada_client(cfg)
    deployment = model_name or cfg.model_name

    response = embedding_client.embeddings.create(model=deployment, input=[stripped])
    vector = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vector)

    return vector


def search_hmo_knowledge(query: str, k: int, user_profile: UserProfile, faiss_index: faiss.Index, stored_chunks: list[dict[str, Any]], *, 
                         embedding_client: AzureOpenAI | None = None, model_name: str | None = None) -> list[RetrievedChunk]:
    """Retrieve top-k chunks: filter by user HMO/tier, then cosine similarity on FAISS vectors."""
    if k < 1:
        raise ValueError("k must be at least 1")

    if user_profile.hmo is None or user_profile.insurance_tier is None:
        raise ValueError("user_profile must include hmo and insurance_tier for retrieval")

    candidate_ids = _filter_chunk_indices(stored_chunks, user_profile)

    if not candidate_ids:
        logger.warning(
            "No chunks match profile hmo=%s tier=%s",
            user_profile.hmo,
            user_profile.insurance_tier,
        )
        return []

    query_vector = embed_query(
        query,
        client=embedding_client,
        model_name=model_name,
    )

    vectors = np.vstack([faiss_index.reconstruct(int(chunk_id)) for chunk_id in candidate_ids])
    faiss.normalize_L2(vectors)
    scores = (vectors @ query_vector.T).flatten()

    top_positions = np.argsort(-scores)[:k]
    results: list[RetrievedChunk] = []

    for position in top_positions:
        chunk_index = candidate_ids[int(position)]
        chunk = stored_chunks[chunk_index]

        results.append(
            RetrievedChunk(
                chunk_id=str(chunk["chunk_id"]),
                text=str(chunk["text"]),
                metadata=dict(chunk.get("metadata", {})),
                score=float(scores[position]),
            )
        )

    logger.info(
        "Retrieved %s chunk(s) for hmo=%s tier=%s query=%r",
        len(results),
        user_profile.hmo,
        user_profile.insurance_tier,
        query[:80],
    )
    
    return results


def format_retrieved_chunks_for_llm(chunks: list[RetrievedChunk]) -> str:
    """Format search hits as context text for the Q&A LLM."""
    if not chunks:
        return "No matching knowledge base entries were found."

    parts: list[str] = []

    for rank, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        header = f"[{rank}] score={chunk.score:.3f}"

        if meta.get("domain"):
            header += f" | domain={meta['domain']}"

        if meta.get("service_name"):
            header += f" | service={meta['service_name']}"

        parts.append(f"{header}\n{chunk.text}")

    return "\n\n---\n\n".join(parts)


def run_search_hmo_knowledge_tool(tool_arguments: str, k: int, user_profile: UserProfile, faiss_index: faiss.Index, stored_chunks: list[dict[str, Any]], *,
                                  embedding_client: AzureOpenAI | None = None, model_name: str | None = None) -> str:
    """
    Execute the ``search_hmo_knowledge`` tool (LLM supplies JSON with ``query`` only).

    Returns formatted text for a ``role: tool`` message.
    """
    payload = json.loads(tool_arguments)
    query = str(payload.get("query", "")).strip()

    if not query:
        raise ValueError("search_hmo_knowledge requires a non-empty query")

    hits = search_hmo_knowledge(
        query,
        k,
        user_profile,
        faiss_index,
        stored_chunks,
        embedding_client=embedding_client,
        model_name=model_name,
    )
    
    return format_retrieved_chunks_for_llm(hits)
