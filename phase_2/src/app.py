from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("PHASE_2_API_BASE_URL", "http://localhost:8000").rstrip("/")
COLLECT_INFO_URL = f"{API_BASE_URL}/collect-info"

MIN_COMPOSER_HEIGHT = 52
MAX_COMPOSER_HEIGHT = 140
CHARS_PER_LINE = 52
MAX_VISIBLE_LINES = 5
STREAM_WORD_DELAY_SEC = 0.04


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

if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, str]] = []

if "composer_nonce" not in st.session_state:
    st.session_state.composer_nonce = 0

if "pending_generation" not in st.session_state:
    st.session_state.pending_generation = False

if "show_empty_message_warning" not in st.session_state:
    st.session_state.show_empty_message_warning = False

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


def _call_collect_info(message: str) -> str:
    """POST the user message to the FastAPI /collect-info endpoint."""
    response = requests.post(
        COLLECT_INFO_URL,
        json={"message": message},
        timeout=120,
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return str(data.get("reply", ""))


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
                assistant_reply = _call_collect_info(_last_user_message())
            st.write_stream(_stream_words(assistant_reply))
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_reply}
            )
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
