from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bs4 import BeautifulSoup, NavigableString, Tag

from openai import AzureOpenAI

from config.auth import create_text_embedding_ada_client, get_text_embedding_ada_config
from config.logger import get_logger
from phase_2.src.constants import (
    ALL_HMOS,
    ALL_TIERS,
    ALLOWED_HMO,
    ALLOWED_INSURANCE_TIERS,
    CHUNK_TYPE_CONTACT_DETAILS,
    CHUNK_TYPE_CONTACT_PHONE,
    CHUNK_TYPE_COVERAGE,
    CHUNK_TYPE_INTRO,
    CHUNK_TYPE_SERVICE_OVERVIEW,
    DEFAULT_HTML_DIR,
    EMBEDDING_BATCH_SIZE,
    HMOS,
)

logger = get_logger(__name__)


@dataclass
class KnowledgeChunk:
    """One retrievable document (text + metadata + optional embedding vector)."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }


def _detect_hmo_prefix(text: str) -> str | None:
    stripped = text.strip()
    for hmo in HMOS:
        if stripped.startswith(hmo):
            return hmo
    return None


def split_cell_by_tier(cell: Tag) -> dict[str, str]:
    """Split one table <td> into tier -> benefit text (זהב / כסף / ארד)."""
    tiers: dict[str, str] = {}
    for strong in cell.find_all("strong"):
        label = strong.get_text(strip=True).rstrip(":")
        if label not in ALLOWED_INSURANCE_TIERS:
            continue
        parts: list[str] = []
        for sibling in strong.next_siblings:
            if isinstance(sibling, Tag) and sibling.name == "strong":
                break
            if isinstance(sibling, NavigableString):
                piece = str(sibling).strip()
                if piece:
                    parts.append(piece)
            elif isinstance(sibling, Tag):
                piece = sibling.get_text(" ", strip=True)
                if piece:
                    parts.append(piece)
        benefit = " ".join(parts).strip()
        if benefit:
            tiers[label] = benefit
    return tiers


def _format_coverage_text(domain: str, service_name: str, hmo: str, tier: str, benefit: str) -> str:
    return (
        f"Domain: {domain}\n"
        f"Service: {service_name}\n"
        f"HMO: {hmo}\n"
        f"Tier: {tier}\n\n"
        f"{benefit}"
    )


def _format_contact_text(domain: str, hmo: str, body: str, *, details: bool) -> str:
    kind = "Contact details" if details else "Customer service phone"
    return f"Domain: {domain}\n{kind}\nHMO: {hmo}\n\n{body}"


def _next_chunk_id(source_file: str, index: int) -> str:
    return f"{source_file}:{index}"


def parse_html_file(path: Path) -> list[KnowledgeChunk]:
    """Parse one HMO knowledge HTML fragment into chunks with metadata."""
    raw_html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")
    source_file = path.name

    h2 = soup.find("h2")
    if h2 is None:
        raise ValueError(f"No <h2> found in {path}")

    domain = h2.get_text(strip=True)
    table = soup.find("table")
    if table is None:
        raise ValueError(f"No <table> found in {path}")

    chunks: list[KnowledgeChunk] = []
    chunk_index = 0

    intro_blocks: list[str] = []
    overview_block: str | None = None

    for sibling in h2.find_next_siblings():
        if sibling is table:
            break
        if not isinstance(sibling, Tag):
            continue
        if sibling.name == "p":
            intro_blocks.append(sibling.get_text("\n", strip=True))
        elif sibling.name == "ul" and overview_block is None:
            overview_block = sibling.get_text("\n", strip=True)

    if intro_blocks:
        intro_text = f"Domain: {domain}\n\n" + "\n\n".join(intro_blocks)
        chunks.append(
            KnowledgeChunk(
                chunk_id=_next_chunk_id(source_file, chunk_index),
                text=intro_text,
                metadata={
                    "chunk_type": CHUNK_TYPE_INTRO,
                    "source_file": source_file,
                    "domain": domain,
                    "hmo": ALL_HMOS,
                    "tier": ALL_TIERS,
                },
            )
        )
        chunk_index += 1

    if overview_block:
        overview_text = f"Domain: {domain}\n\nServices covered:\n{overview_block}"
        chunks.append(
            KnowledgeChunk(
                chunk_id=_next_chunk_id(source_file, chunk_index),
                text=overview_text,
                metadata={
                    "chunk_type": CHUNK_TYPE_SERVICE_OVERVIEW,
                    "source_file": source_file,
                    "domain": domain,
                    "hmo": ALL_HMOS,
                    "tier": ALL_TIERS,
                },
            )
        )
        chunk_index += 1

    header_cells = table.find("tr").find_all(["th", "td"])
    headers = [cell.get_text(strip=True) for cell in header_cells]
    hmo_by_column: dict[int, str] = {}
    for col_idx, header in enumerate(headers):
        if header in ALLOWED_HMO:
            hmo_by_column[col_idx] = header

    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        service_name = cells[0].get_text(strip=True)
        if not service_name:
            continue

        for col_idx, hmo in hmo_by_column.items():
            if col_idx >= len(cells):
                continue
            cell = cells[col_idx]
            for tier, benefit in split_cell_by_tier(cell).items():
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=_next_chunk_id(source_file, chunk_index),
                        text=_format_coverage_text(domain, service_name, hmo, tier, benefit),
                        metadata={
                            "chunk_type": CHUNK_TYPE_COVERAGE,
                            "source_file": source_file,
                            "domain": domain,
                            "service_name": service_name,
                            "hmo": hmo,
                            "tier": tier,
                        },
                    )
                )
                chunk_index += 1

    for h3 in soup.find_all("h3"):
        title = h3.get_text(strip=True)
        ul = h3.find_next_sibling("ul")
        if ul is None:
            continue

        if "טלפון" in title or "מספרי" in title:
            chunk_type = CHUNK_TYPE_CONTACT_PHONE
            details = False
        elif "פרטים" in title:
            chunk_type = CHUNK_TYPE_CONTACT_DETAILS
            details = True
        else:
            continue

        for li in ul.find_all("li", recursive=False):
            li_text = li.get_text("\n", strip=True)
            hmo = _detect_hmo_prefix(li_text)
            if hmo is None:
                logger.warning("Could not detect HMO in %s: %s", path.name, li_text[:80])
                continue
            chunks.append(
                KnowledgeChunk(
                    chunk_id=_next_chunk_id(source_file, chunk_index),
                    text=_format_contact_text(domain, hmo, li_text, details=details),
                    metadata={
                        "chunk_type": chunk_type,
                        "source_file": source_file,
                        "domain": domain,
                        "hmo": hmo,
                        "tier": ALL_TIERS,
                    },
                )
            )
            chunk_index += 1

    return chunks


def scrape_html_directory(html_dir: Path | None = None) -> list[KnowledgeChunk]:
    """Read and parse all ``*.html`` knowledge files under ``html_dir``."""
    directory = html_dir or DEFAULT_HTML_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"HTML directory not found: {directory}")

    html_paths = sorted(directory.glob("*.html"))
    if not html_paths:
        raise FileNotFoundError(f"No HTML files in {directory}")

    all_chunks: list[KnowledgeChunk] = []
    for path in html_paths:
        file_chunks = parse_html_file(path)
        logger.info("Parsed %s — %s chunks", path.name, len(file_chunks))
        all_chunks.extend(file_chunks)

    logger.info("Scraped %s files — %s chunks total", len(html_paths), len(all_chunks))
    return all_chunks


def embed_chunks(
    chunks: list[KnowledgeChunk],
    *,
    client: AzureOpenAI | None = None,
    model_name: str | None = None,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[KnowledgeChunk]:
    """Compute ADA-002 embeddings for every chunk (mutates and returns the same list)."""
    if not chunks:
        logger.info("No chunks to embed")
        return chunks

    cfg = get_text_embedding_ada_config()
    embedding_client = client or create_text_embedding_ada_client(cfg)
    deployment = model_name or cfg.model_name
    texts = [chunk.text for chunk in chunks]
    all_vectors: list[list[float]] = []

    logger.info(
        "Embedding %s chunks with %s (batch_size=%s)",
        len(chunks),
        deployment,
        batch_size,
    )

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = embedding_client.embeddings.create(model=deployment, input=batch)
        batch_vectors = [item.embedding for item in sorted(response.data, key=lambda row: row.index)]
        all_vectors.extend(batch_vectors)
        logger.info(
            "Embedded batch %s–%s / %s",
            start + 1,
            start + len(batch),
            len(texts),
        )

    if len(all_vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(chunks)}, got {len(all_vectors)}"
        )

    for chunk, vector in zip(chunks, all_vectors, strict=True):
        chunk.embedding = vector

    dimension = len(all_vectors[0])
    logger.info("Finished embedding %s chunks (dimension=%s)", len(chunks), dimension)
    return chunks


if __name__ == "__main__":
    chunks = scrape_html_directory()
    chunks = embed_chunks(chunks)