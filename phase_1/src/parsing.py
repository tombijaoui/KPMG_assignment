from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from azure.ai.documentintelligence.models import AnalyzeResult

from config.auth import create_document_intelligence_client
from config.logger import get_logger
from phase_1.src.constants import CONTENT_TYPE_BY_SUFFIX, DEFAULT_PDF, LAYOUT_MODEL_ID
from phase_1.src.llm_extraction import extract_merged_fields, refine_extracted_fields
from phase_1.src.validation import (
    normalize_extracted_fields,
    run_validation_pipeline,
    validation_result_to_dict,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class OcrResult:
    """Layer 1 — Document Intelligence layout output."""

    text: str
    page_count: int
    analyze_result: AnalyzeResult


@dataclass(frozen=True)
class ParsingPipelineResult:
    """Full pipeline output (extended as layers 2 and 3 are implemented)."""

    ocr: OcrResult
    extracted_fields: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None


def _content_type_for_suffix(suffix: str) -> str:
    content_type = CONTENT_TYPE_BY_SUFFIX.get(suffix.lower())
    if not content_type:
        raise ValueError(f"Unsupported file type: {suffix}")
    return content_type


def get_text_from_analyze_result(result: AnalyzeResult) -> str:
    """Return document text in reading order."""
    return result.content or ""


def run_ocr(
    document: bytes | Path | str,
    *,
    filename: str | None = None,
    model_id: str = LAYOUT_MODEL_ID,
) -> OcrResult:
    """
    Layer 1 — send a document to Document Intelligence and return extracted text.

    Args:
        document: File path or raw bytes (bytes require ``filename`` for content type).
        filename: Original file name when ``document`` is bytes (e.g. Streamlit upload).
        model_id: Azure model id (prebuilt-layout recommended for forms).
    """
    if isinstance(document, (str, Path)):
        path = Path(document)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        content_type = _content_type_for_suffix(path.suffix)
        with path.open("rb") as file:
            document_bytes = file.read()
    else:
        if not filename:
            raise ValueError("filename is required when document is provided as bytes")
        content_type = _content_type_for_suffix(Path(filename).suffix)
        document_bytes = document

    logger.info("Layer 1 — starting OCR (model=%s)", model_id)
    client = create_document_intelligence_client()
    poller = client.begin_analyze_document(
        model_id=model_id,
        body=io.BytesIO(document_bytes),
        content_type=content_type,
    )
    analyze_result = poller.result()
    text = get_text_from_analyze_result(analyze_result)
    page_count = len(analyze_result.pages or [])

    logger.info(
        "Layer 1 — OCR completed: %s page(s), %s character(s)",
        page_count,
        len(text),
    )
    return OcrResult(
        text=text,
        page_count=page_count,
        analyze_result=analyze_result,
    )


def parsing_pipeline(
    document: bytes | Path | str,
    *,
    filename: str | None = None,
) -> ParsingPipelineResult:
    """
    Run the full parsing pipeline on a PDF or image.

    Layer 1: OCR (Document Intelligence)
    Layer 2: JSON field extraction (GPT-4o mini — two passes merged, then refinement pass)
    Layer 3: Field validation (dates, ID, phones, postal code)
    """
    logger.info("Starting parsing pipeline")

    ocr_result = run_ocr(document, filename=filename)
    merged = extract_merged_fields(ocr_result.text)
    refined = refine_extracted_fields(ocr_result.text, merged)
    extracted_fields = normalize_extracted_fields(refined)

    validation_report = run_validation_pipeline(extracted_fields)
    validation = validation_result_to_dict(validation_report)

    logger.info("Parsing pipeline completed")
    return ParsingPipelineResult(
        ocr=ocr_result,
        extracted_fields=extracted_fields,
        validation=validation,
    )


if __name__ == "__main__":
    pipeline_result = parsing_pipeline(DEFAULT_PDF)
    logger.info("OCR text:\n%s", pipeline_result.ocr.text)
    logger.info(
        "Extracted JSON:\n%s",
        json.dumps(pipeline_result.extracted_fields, ensure_ascii=False, indent=2),
    )
    logger.info(
        "Validation:\n%s",
        json.dumps(pipeline_result.validation, ensure_ascii=False, indent=2),
    )
