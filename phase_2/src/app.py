from __future__ import annotations

import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Allow `streamlit run phase_2/src/app.py` without PYTHONPATH set.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import requests
import streamlit as st

from phase_2.src.constants import (
    CHARS_PER_LINE,
    COLLECT_INFO_URL,
    COLLECT_RECENT_MESSAGE_LIMIT,
    MAX_COMPOSER_HEIGHT,
    MAX_VISIBLE_LINES,
    MIN_COMPOSER_HEIGHT,
    STREAM_WORD_DELAY_SEC,
)


st.set_page_config(
    page_title="Health Fund Assistant",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
    h1.app-title {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    [data-testid="stBottom"] {
        width: 100%;
        padding: 0 1rem 0.5rem;
    }
    [data-testid="stBottom"] [data-testid="stVerticalBlockBorderWrapper"] {
        width: 100%;
    }
    [data-testid="stBottom"] textarea {
        border: none !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="app-title">Health Fund Assistant</h1>', unsafe_allow_html=True)


def _empty_user_profile() -> dict[str, Any]:
    return {
        "first_name": None,
        "last_name": None,
        "id_number": None,
        "gender": None,
        "age": None,
        "hmo": None,
        "hmo_card_number": None,
        "insurance_tier": None,
    }


if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, str]] = []

if "composer_nonce" not in st.session_state:
    st.session_state.composer_nonce = 0

if "pending_generation" not in st.session_state:
    st.session_state.pending_generation = False

if "show_empty_message_warning" not in st.session_state:
    st.session_state.show_empty_message_warning = False

if "user_profile" not in st.session_state:
    st.session_state.user_profile = _empty_user_profile()

if "profile_confirmed" not in st.session_state:
    st.session_state.profile_confirmed = False

if "ready_for_qa" not in st.session_state:
    st.session_state.ready_for_qa = False

if st.session_state.show_empty_message_warning:
    st.warning("Please enter a message before sending.")


def _composer_widget_key() -> str:
    return f"query_draft_{st.session_state.composer_nonce}"


def _on_send_click() -> None:
    """Handle send without mutating widget state after instantiation."""
    draft_key = _composer_widget_key()
    user_text = str(st.session_state.get(draft_key, "")).strip()
    if not user_text:
        st.session_state.show_empty_message_warning = True
        return
    st.session_state.show_empty_message_warning = False
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.pending_generation = True
    st.session_state.composer_nonce += 1


def _composer_height(text: str) -> int:
    """Estimate textarea height (px) from draft text length."""
    if not text.strip():
        return MIN_COMPOSER_HEIGHT
    line_count = max(1, text.count("\n") + 1, (len(text) // CHARS_PER_LINE) + 1)
    line_count = min(line_count, MAX_VISIBLE_LINES)
    return min(MAX_COMPOSER_HEIGHT, max(MIN_COMPOSER_HEIGHT, 20 + line_count * 22))


def _recent_messages_for_api() -> list[dict[str, str]]:
    """Last N chat turns (user + assistant) for collect-info context."""
    messages = st.session_state.messages
    if not messages:
        return []
    tail = messages[-COLLECT_RECENT_MESSAGE_LIMIT + 1:]
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in tail]


def _call_collect_info(
    message: str,
    user_profile: dict[str, Any],
    recent_messages: list[dict[str, str]],
) -> dict[str, Any]:
    """POST to /collect-info with message, profile, and recent chat context."""
    response = requests.post(
        COLLECT_INFO_URL,
        json={
            "message": message,
            "user_profile": user_profile,
            "recent_messages": recent_messages,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def _stream_words(text: str) -> Iterator[str]:
    """Yield the reply word-by-word for st.write_stream."""
    tokens = re.findall(r"\S+\s*|\n", text)
    if not tokens:
        if text:
            yield text
        return
    for token in tokens:
        yield token
        time.sleep(STREAM_WORD_DELAY_SEC)


def _last_user_message() -> str:
    for message in reversed(st.session_state.messages):
        if message["role"] == "user":
            return message["content"]
    return ""


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.pending_generation:
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                collect_result = _call_collect_info(
                    _last_user_message(),
                    st.session_state.user_profile,
                    _recent_messages_for_api(),
                )
            assistant_reply = str(collect_result.get("reply", ""))
            st.session_state.user_profile = collect_result.get(
                "user_profile",
                st.session_state.user_profile,
            )
            st.session_state.profile_confirmed = bool(
                collect_result.get("profile_confirmed", False)
            )
            st.session_state.ready_for_qa = bool(collect_result.get("ready_for_qa", False))
            st.write_stream(_stream_words(assistant_reply))
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_reply}
            )
            if st.session_state.ready_for_qa:
                st.success("Profile confirmed — Q&A phase will be available soon.")
        except requests.RequestException as exc:
            error_text = (
                f"Sorry, the assistant is unavailable ({exc}). "
                "Make sure the API is running: "
                "`uvicorn phase_2.src.api:app --reload --port 8000`"
            )
            st.markdown(error_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_text}
            )
        st.session_state.pending_generation = False
        st.rerun()

composer_key = _composer_widget_key()
draft_height = _composer_height(str(st.session_state.get(composer_key, "")))

with st._bottom:
    with st.container(border=True):
        input_col, button_col = st.columns([14, 1], gap="small", vertical_alignment="center")
        with input_col:
            st.text_area(
                "Your message",
                placeholder="Type your message...",
                label_visibility="collapsed",
                height=draft_height,
                key=composer_key,
                disabled=st.session_state.pending_generation,
            )
        with button_col:
            st.button(
                "↑",
                use_container_width=True,
                key="send_message",
                help="Send message",
                disabled=st.session_state.pending_generation,
                on_click=_on_send_click,
            )
