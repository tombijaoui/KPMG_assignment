from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from openai import AzureOpenAI

from config.auth import create_llm_gpt_4o_mini_client, get_llm_gpt_4o_mini_config
from config.logger import get_logger
from phase_1.src.constants import (
    ACCIDENT_DETAILS_SCHEMA,
    PERSONAL_INFO_SCHEMA,
    REFINEMENT_FOCUS_SCHEMA,
    REFINEMENT_TEMPERATURE,
    TEMPERATURE,
)
from phase_1.src.prompts import (
    ACCIDENT_EXTRACTION_TASK,
    EXTRACTION_SYSTEM_PROMPT,
    PERSONAL_EXTRACTION_TASK,
    REFINEMENT_SYSTEM_PROMPT,
    build_extraction_user_message,
    build_refinement_user_message,
)

logger = get_logger(__name__)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)

    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)

        else:
            result[key] = value

    return result


def empty_form_template() -> dict[str, Any]:
    """Full readme JSON schema with empty values."""
    return _deep_merge(PERSONAL_INFO_SCHEMA, ACCIDENT_DETAILS_SCHEMA)


def _draft_slice_for_refinement(draft: dict[str, Any], shape: dict[str, Any]) -> dict[str, Any]:
    """Copy from draft only the keys present in ``shape`` (same nesting as REFINEMENT_FOCUS_SCHEMA)."""
    out: dict[str, Any] = {}

    for key, template in shape.items():
        if isinstance(template, dict):
            src = draft.get(key)
            src_dict = src if isinstance(src, dict) else {}
            out[key] = _draft_slice_for_refinement(src_dict, template)

        else:
            val = draft.get(key, "")
            out[key] = "" if val is None else str(val)

    return out


def _coerce_refinement_response_to_focus_shape(refined: dict[str, Any], shape: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys allowed by ``shape``; ignore extra keys from the model."""
    out: dict[str, Any] = {}

    for key, template in shape.items():
        if key not in refined:
            continue

        val = refined[key]

        if isinstance(template, dict):
            if isinstance(val, dict):
                nested = _coerce_refinement_response_to_focus_shape(val, template)

                if nested:
                    out[key] = nested

        else:
            out[key] = "" if val is None else str(val)

    return out


def _call_llm_for_schema(client: AzureOpenAI, model_name: str, schema: dict[str, Any], task_description: str, ocr_text: str) -> dict[str, Any]:
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    user_message = build_extraction_user_message(task_description, schema_json, ocr_text)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=TEMPERATURE,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("LLM returned an empty response")

    return json.loads(content)


def _call_llm_refinement(client: AzureOpenAI, model_name: str, focus_schema: dict[str, Any], draft_focus_slice: dict[str, Any], ocr_text: str) -> dict[str, Any]:
    """Third LLM pass — model returns only REFINEMENT_FOCUS_SCHEMA-shaped JSON."""
    schema_json = json.dumps(focus_schema, ensure_ascii=False, indent=2)
    draft_json = json.dumps(draft_focus_slice, ensure_ascii=False, indent=2)
    user_message = build_refinement_user_message(schema_json, draft_json, ocr_text)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=REFINEMENT_TEMPERATURE,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("LLM returned an empty response")

    return json.loads(content)


def extract_personal_info(ocr_text: str) -> dict[str, Any]:
    """LLM pass 1 — personal and form administrative fields."""
    logger.info("Layer 2a — extracting personal information")

    config = get_llm_gpt_4o_mini_config()
    client = create_llm_gpt_4o_mini_client(config)

    result = _call_llm_for_schema(
        client,
        config.model_name,
        PERSONAL_INFO_SCHEMA,
        PERSONAL_EXTRACTION_TASK,
        ocr_text,
    )

    logger.info("Layer 2a — personal information extraction completed")
    return result


def extract_accident_details(ocr_text: str) -> dict[str, Any]:
    """LLM pass 2 — accident and medical institution fields."""
    logger.info("Layer 2b — extracting accident details")

    config = get_llm_gpt_4o_mini_config()
    client = create_llm_gpt_4o_mini_client(config)

    result = _call_llm_for_schema(
        client,
        config.model_name,
        ACCIDENT_DETAILS_SCHEMA,
        ACCIDENT_EXTRACTION_TASK,
        ocr_text,
    )

    logger.info("Layer 2b — accident details extraction completed")
    return result


def refine_extracted_fields(ocr_text: str, draft: dict[str, Any]) -> dict[str, Any]:
    """
    LLM pass 3 — correct fragile fields (health fund, gender, nature of accident).

    The model receives only REFINEMENT_FOCUS_SCHEMA-shaped values; its JSON is merged into ``draft``.
    """
    logger.info("Layer 2c — refining extraction with full OCR context")
    config = get_llm_gpt_4o_mini_config()
    client = create_llm_gpt_4o_mini_client(config)
    focus_slice = _draft_slice_for_refinement(draft, REFINEMENT_FOCUS_SCHEMA)

    refined_focus = _call_llm_refinement(
        client,
        config.model_name,
        REFINEMENT_FOCUS_SCHEMA,
        focus_slice,
        ocr_text,
    )

    refined_focus = _coerce_refinement_response_to_focus_shape(
        refined_focus, REFINEMENT_FOCUS_SCHEMA
    )

    merged = _deep_merge(deepcopy(draft), refined_focus)
    logger.info("Layer 2c — refinement completed")
    return merged


def extract_merged_fields(ocr_text: str) -> dict[str, Any]:
    """Run the two extraction LLM passes and merge into one draft (no refinement)."""
    personal = extract_personal_info(ocr_text)
    accident = extract_accident_details(ocr_text)

    merged = _deep_merge(empty_form_template(), personal)
    merged = _deep_merge(merged, accident)
    
    logger.info("Layer 2 — two-pass merge completed (refinement is orchestrated separately)")
    return merged
