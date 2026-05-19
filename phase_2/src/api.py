from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from openai import AsyncAzureOpenAI
from pydantic import ValidationError

from config.auth import create_llm_gpt_4o_client, get_llm_gpt_4o_config
from config.logger import get_logger
from phase_2.src.constants import (
    COLLECT_LLM_TEMPERATURE,
    DEFAULT_RETRIEVAL_TOP_K,
    QA_LLM_TEMPERATURE,
    QA_RECENT_MESSAGE_LIMIT,
)
from phase_2.src.indexing import load_faiss_knowledge_base, run_indexing_pipeline
from phase_2.src.models import (
    CollectInfoRequest,
    CollectInfoResponse,
    CollectLLMOutput,
    QAChatMessage,
    QARequest,
    QAResponse,
    QAToolCall,
    QAToolCallFunction,
    QAToolUsage,
)
from phase_2.src.prompts import (
    COLLECT_SYSTEM_PROMPT,
    build_collect_user_message,
    build_qa_messages,
)
from phase_2.src.retrieving import run_search_hmo_knowledge_tool
from phase_2.src.tools import QA_TOOLS
from phase_2.src.validation import is_profile_complete, merge_profile

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # OpenAI client
    config = get_llm_gpt_4o_config()
    app.state.openai_client = create_llm_gpt_4o_client(config, async_client=True)
    app.state.openai_model_name = config.model_name
    logger.info("Async OpenAI client ready (model=%s)", config.model_name)

 
    # Knowledge base
    try:
        loaded = await asyncio.to_thread(load_faiss_knowledge_base)

    except FileNotFoundError:
        await asyncio.to_thread(run_indexing_pipeline)
        loaded = await asyncio.to_thread(load_faiss_knowledge_base)

    app.state.faiss_index, app.state.kb_chunks, app.state.kb_manifest = loaded

    yield

    await app.state.openai_client.close()
    logger.info("Async OpenAI client closed")


app = FastAPI(
    title="Phase 2 — Health Fund Chatbot API",
    version="0.1.0",
    lifespan=lifespan,
)


async def _call_collect_llm(request: CollectInfoRequest, client: AsyncAzureOpenAI, model_name: str) -> CollectLLMOutput:
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


def _build_response(request: CollectInfoRequest, llm_output: CollectLLMOutput) -> CollectInfoResponse:
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


def _qa_message_to_api_dict(message: QAChatMessage) -> dict[str, object]:
    payload: dict[str, object] = {"role": message.role, "content": message.content}

    if message.tool_calls:
        payload["tool_calls"] = [tool_call.model_dump() for tool_call in message.tool_calls]

    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id

    return payload


def _qa_prior_messages(request: QARequest) -> list[dict[str, object]]:
    """Prior Q&A history capped by QA_RECENT_MESSAGE_LIMIT (includes tool messages)."""
    api_messages = [_qa_message_to_api_dict(message) for message in request.messages]

    if len(api_messages) > QA_RECENT_MESSAGE_LIMIT:
        api_messages = api_messages[-QA_RECENT_MESSAGE_LIMIT:]

    return api_messages


def _tool_calls_from_assistant(message: object) -> list[QAToolCall]:
    tool_calls = getattr(message, "tool_calls", None) or []
    return [
        QAToolCall(
            id=call.id,
            type=call.type,
            function=QAToolCallFunction(
                name=call.function.name,
                arguments=call.function.arguments,
            ),
        )
        for call in tool_calls
    ]


def _tool_usages_from_assistant(message: object) -> list[QAToolUsage]:
    return [
        QAToolUsage(
            order=index,
            name=call.function.name,
            arguments=call.function.arguments,
            tool_call_id=call.id,
        )
        for index, call in enumerate(getattr(message, "tool_calls", None) or [], start=1)
    ]


def _build_qa_turn_messages(*, user_message: str, assistant_with_tools: object | None, tool_result_messages: list[dict[str, object]], 
                            final_reply: str) -> list[QAChatMessage]:
    """Build Q&A turn messages from user message, assistant with tools, tool result messages, and final reply."""

    turn: list[QAChatMessage] = [QAChatMessage(role="user", content=user_message)]

    if assistant_with_tools is not None and getattr(assistant_with_tools, "tool_calls", None):
        turn.append(
            QAChatMessage(
                role="assistant",
                content=getattr(assistant_with_tools, "content", None) or "",
                tool_calls=_tool_calls_from_assistant(assistant_with_tools),
            )
        )
        for tool_message in tool_result_messages:
            turn.append(
                QAChatMessage(
                    role="tool",
                    tool_call_id=str(tool_message["tool_call_id"]),
                    content=str(tool_message["content"]),
                )
            )

    turn.append(QAChatMessage(role="assistant", content=final_reply))
    return turn


def _assistant_message_with_tool_calls(message: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "role": "assistant",
        "content": getattr(message, "content", None) or "",
    }

    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]

    return payload


async def _call_qa_llm(request: QARequest, client: AsyncAzureOpenAI, model_name: str, http_request: Request) -> QAResponse:
    """Async GPT-4o call returning Q&A response and turn messages."""
    profile_json = request.user_profile.model_dump_json(indent=2, exclude_none=True)
    prior_messages = _qa_prior_messages(request)

    messages: list[dict[str, object]] = build_qa_messages(
        profile_json=profile_json,
        prior_messages=prior_messages,
        latest_message=request.message,
    )

    logger.info(
        "Calling GPT-4o for /qa (model=%s, prior_len=%s, limit=%s)",
        model_name,
        len(prior_messages),
        QA_RECENT_MESSAGE_LIMIT,
    )

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=QA_TOOLS,
        tool_choice="auto",
        temperature=QA_LLM_TEMPERATURE,
    )
    assistant = response.choices[0].message

    if not assistant.tool_calls:
        reply = (assistant.content or "").strip()

        if not reply:
            raise ValueError("LLM returned an empty Q&A response")

        logger.info("POST /qa — direct reply (no tool call)")

        return QAResponse(
            reply=reply,
            turn_messages=_build_qa_turn_messages(
                user_message=request.message,
                assistant_with_tools=None,
                tool_result_messages=[],
                final_reply=reply,
            ),
            tool_calls=[],
        )

    tool_usages = _tool_usages_from_assistant(assistant)
    logger.info(
        "POST /qa — tool_calls=%s",
        [usage.name for usage in tool_usages],
    )

    messages.append(_assistant_message_with_tool_calls(assistant))
    tool_result_messages: list[dict[str, object]] = []

    for tool_call in assistant.tool_calls:
        if tool_call.function.name != "search_hmo_knowledge":
            logger.warning("Unknown tool requested: %s", tool_call.function.name)
            tool_content = f"Unknown tool: {tool_call.function.name}"

        else:
            tool_content = await asyncio.to_thread(
                run_search_hmo_knowledge_tool,
                tool_call.function.arguments,
                DEFAULT_RETRIEVAL_TOP_K,
                request.user_profile,
                http_request.app.state.faiss_index,
                http_request.app.state.kb_chunks,
            )

            logger.info(
                "POST /qa — search_hmo_knowledge query=%r",
                tool_call.function.arguments[:120],
            )

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_content,
        }
        tool_result_messages.append(tool_message)
        messages.append(tool_message)

    final = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=QA_TOOLS,
        tool_choice="none",
        temperature=QA_LLM_TEMPERATURE,
    )
    reply = (final.choices[0].message.content or "").strip()

    if not reply:
        raise ValueError("LLM returned an empty Q&A response after tool use")

    logger.info("POST /qa — reply after tool use")

    return QAResponse(
        reply=reply,
        turn_messages=_build_qa_turn_messages(
            user_message=request.message,
            assistant_with_tools=assistant,
            tool_result_messages=tool_result_messages,
            final_reply=reply,
        ),
        tool_calls=tool_usages,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    logger.info("GET /health")
    return {"status": "ok"}


@app.post("/collect-info", response_model=CollectInfoResponse)
async def collect_info(request: CollectInfoRequest, http_request: Request) -> CollectInfoResponse:
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


@app.post("/qa", response_model=QAResponse)
async def qa(request: QARequest, http_request: Request) -> QAResponse:
    """Answer member questions using the knowledge base (tool-based retrieval)."""
    logger.info(
        "POST /qa — message=%s profile_confirmed=%s hmo=%s tier=%s",
        request.message,
        request.profile_confirmed,
        request.user_profile.hmo,
        request.user_profile.insurance_tier,
    )

    if http_request.app.state.faiss_index is None:
        logger.error("POST /qa — knowledge base not loaded")
        raise HTTPException(status_code=503, detail="Knowledge base is not ready")

    if not request.profile_confirmed:
        logger.warning("POST /qa — profile not confirmed")
        raise HTTPException(
            status_code=400,
            detail="Profile must be confirmed before Q&A",
        )

    if not is_profile_complete(request.user_profile):
        logger.warning("POST /qa — incomplete profile")
        raise HTTPException(status_code=400, detail="User profile is incomplete")

    try:
        qa_result = await _call_qa_llm(
            request,
            http_request.app.state.openai_client,
            http_request.app.state.openai_model_name,
            http_request,
        )

    except ValueError as exc:
        logger.warning("Q&A validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Q&A LLM call failed")
        raise HTTPException(status_code=500, detail="Q&A LLM call failed") from exc

    logger.info(
        "POST /qa — reply=%s tool_calls=%s",
        qa_result.reply[:200],
        [usage.name for usage in qa_result.tool_calls],
    )
    
    return qa_result
