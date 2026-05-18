from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.auth import create_llm_gpt_4o_client, get_llm_gpt_4o_config
from config.logger import get_logger
from phase_2.src.constants import COLLECT_LLM_TEMPERATURE
from phase_2.src.prompts import COLLECT_SYSTEM_PROMPT, build_collect_user_message

logger = get_logger(__name__)

app = FastAPI(
    title="Phase 2 — Health Fund Chatbot API",
    version="0.1.0",
)


class CollectInfoRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Latest user message")


class CollectInfoResponse(BaseModel):
    reply: str


def _call_collect_llm(request: CollectInfoRequest) -> str:
    """Send CollectInfoRequest context and user message to GPT-4o."""
    config = get_llm_gpt_4o_config()
    client = create_llm_gpt_4o_client(config)

    user_content = build_collect_user_message(request.model_dump_json(indent=2))

    messages = [
        {"role": "system", "content": COLLECT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info("Calling GPT-4o for /collect-info (model=%s)", config.model_name)

    response = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=COLLECT_LLM_TEMPERATURE,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("LLM returned an empty response")

    return content.strip()


@app.get("/health")
def health() -> dict[str, str]:
    logger.info("GET /health")
    return {"status": "ok"}


@app.post("/collect-info", response_model=CollectInfoResponse)
def collect_info(request: CollectInfoRequest) -> CollectInfoResponse:
    """Collect user profile information via GPT-4o using the CollectInfoRequest payload."""
    logger.info("POST /collect-info — message length=%s", len(request.message))

    try:
        reply = _call_collect_llm(request)

    except ValueError as exc:
        logger.warning("Collect-info LLM validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
        
    except Exception as exc:
        logger.exception("Collect-info LLM call failed")
        raise HTTPException(status_code=500, detail="Collect-info LLM call failed") from exc

    logger.info("POST /collect-info — reply length=%s", len(reply))
    return CollectInfoResponse(reply=reply)
