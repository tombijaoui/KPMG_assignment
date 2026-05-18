from __future__ import annotations
from typing import Any
import streamlit as st
from phase_1.src.parsing import parsing_pipeline


def _upload_fingerprint(uploaded_file: Any) -> str:
    return f"{uploaded_file.name}:{uploaded_file.size}"


st.set_page_config(
    page_title="Bituach Leumi Accident Forms Parser",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    h1 {
        text-align: center;
    }
    [data-testid="stCaptionContainer"] {
        text-align: center;
    }
    [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
        min-height: 360px;
        padding: 3rem 1.5rem;
        border: 2px dashed #4a90d9;
        border-radius: 12px;
        background-color: #f0f7ff;
    }
    [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #2563eb;
        background-color: #e8f2ff;
    }
    [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] div {
        font-size: 1.05rem;
    }
    [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] small {
        font-size: 0.95rem;
    }
    button[data-testid="stBaseButton-primary"] {
        background-color: #1e3a8a !important;
        border: 1px solid #172e6e !important;
        color: #ffffff !important;
        font-weight: 600;
        border-radius: 8px;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #172e6e !important;
        border-color: #122454 !important;
        color: #ffffff !important;
    }
    button[data-testid="stBaseButton-primary"]:focus {
        box-shadow: 0 0 0 0.2rem rgba(30, 58, 138, 0.5) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Bituach Leumi Accident Forms Parser")

st.caption(
    "Upload one National Insurance work-injury form (PDF or image) to extract structured JSON."
)

uploaded_file = st.file_uploader(
    label="Drag and drop your document here",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=False,
    help="One document only. Supported formats: PDF, PNG, JPG, JPEG.",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    st.success(f"File loaded: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
    fingerprint = _upload_fingerprint(uploaded_file)
    if st.session_state.get("upload_fingerprint") != fingerprint:
        st.session_state["upload_fingerprint"] = fingerprint
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("parse_error", None)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
_, button_col = st.columns([5, 1])
with button_col:
    parse_clicked = st.button(
        "Parse Form",
        type="primary",
        use_container_width=True,
        key="parse_form",
    )

if parse_clicked:
    if uploaded_file is None:
        st.warning("Please upload a document before parsing.")
    else:
        try:
            with st.spinner("Parsing..."):
                st.session_state["pipeline_result"] = parsing_pipeline(
                    uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                )
            st.session_state.pop("parse_error", None)
        except Exception as exc:
            st.session_state.pop("pipeline_result", None)
            st.session_state["parse_error"] = str(exc)

if st.session_state.get("parse_error"):
    st.error(f"Parsing failed: {st.session_state['parse_error']}")

result = st.session_state.get("pipeline_result")
if result and result.extracted_fields is not None:
    st.divider()
    st.json(result.extracted_fields)
