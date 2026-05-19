from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from config.auth import create_llm_gpt_4o_client, get_llm_gpt_4o_config
from config.logger import get_logger
from phase_2.src.constants import COLLECT_LLM_TEMPERATURE
from phase_2.src.models import (
    CollectInfoRequest,
    CollectInfoResponse,
    CollectLLMOutput,
)
from phase_2.src.prompts import COLLECT_SYSTEM_PROMPT, build_collect_user_message
from phase_2.src.validation import is_profile_complete, merge_profile

from openai import AsyncAzureOpenAI

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Shared AsyncAzureOpenAI client for the app lifetime."""
    config = get_llm_gpt_4o_config()
    app.state.openai_client = create_llm_gpt_4o_client(config, async_client=True)

    app.state.openai_model_name = config.model_name
    logger.info("Async OpenAI client ready (model=%s)", config.model_name)

    yield

    await app.state.openai_client.close()
    logger.info("Async OpenAI client closed")


app = FastAPI(
    title="Phase 2 — Health Fund Chatbot API",
    version="0.1.0",
    lifespan=lifespan,
)


async def _call_collect_llm(
    request: CollectInfoRequest,
    client: AsyncAzureOpenAI,
    model_name: str,
) -> CollectLLMOutput:
    """Async GPT-4o call returning conversational reply and profile patch."""
    user_content = build_collect_user_message(
        request.model_dump_json(indent=2, exclude_none=True)
    )

    messages = [
        {"role": "system", "content": COLLECT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info("Calling GPT-4o for /collect-info (model=%s)", model_name)

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=COLLECT_LLM_TEMPERATURE,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        logger.error("LLM returned an empty response")
        raise ValueError("LLM returned an empty response")

    try:
        payload = json.loads(content)
        return CollectLLMOutput.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc


def _build_response(
    request: CollectInfoRequest,
    llm_output: CollectLLMOutput,
) -> CollectInfoResponse:
    updated_profile = merge_profile(request.user_profile, llm_output.profile_patch)
    profile_valid = is_profile_complete(updated_profile)
    profile_confirmed = llm_output.profile_confirmed and profile_valid
    ready_for_qa = profile_confirmed

    return CollectInfoResponse(
        reply=llm_output.reply.strip(),
        user_profile=updated_profile,
        profile_confirmed=profile_confirmed,
        profile_valid=profile_valid,
        ready_for_qa=ready_for_qa,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    logger.info("GET /health")
    return {"status": "ok"}


@app.post("/collect-info", response_model=CollectInfoResponse)
async def collect_info(
    request: CollectInfoRequest,
    http_request: Request,
) -> CollectInfoResponse:
    """Collect user profile via one LLM call (reply + structured profile_patch)."""
    logger.info(
        "POST /collect-info — message=%s recent_messages=%s",
        request.message,
        request.recent_messages,
    )

    try:
        llm_output = await _call_collect_llm(
            request,
            http_request.app.state.openai_client,
            http_request.app.state.openai_model_name,
        )
        response = _build_response(request, llm_output)
    except ValueError as exc:
        logger.warning("Collect-info validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Collect-info LLM call failed")
        raise HTTPException(status_code=500, detail="Collect-info LLM call failed") from exc

    logger.info(
        "POST /collect-info — reply=%s profile_valid=%s ready_for_qa=%s user_profile_info = %s",
        response.reply,
        response.profile_valid,
        response.ready_for_qa,
        response.user_profile.model_dump(exclude_none=True),
    )
    return response
