# KPMG GenAI Assignment — Implementation Notes

This document describes how the two assignment phases were implemented: design choices, pipeline steps, and how to run each part locally. Content is added incrementally; sections marked below will be filled in as we go.

---

## Table of contents

1. [Shared configuration (`config/`)](#shared-configuration-config)
2. [Phase 1 — National Insurance accident form extraction](#phase-1--national-insurance-accident-form-extraction)
3. [Phase 2 — Health fund chatbot](#phase-2--health-fund-chatbot)

---

## Shared configuration (`config/`)

Both phases rely on a small shared package at the repository root. It centralizes Azure credentials, SDK client factories, and logging so phase-specific code stays focused on business logic.

Credentials are loaded once from a `.env` file at the project root (via `python-dotenv`). The `.env` file itself is not committed to Git.

| Module | Used by | Purpose |
|--------|---------|---------|
| `config/auth.py` | Phase 1 & 2 | Read Azure settings from the environment and build SDK clients |
| `config/logger.py` | Phase 1 & 2 | Consistent structured logging across all modules |

### `config/auth.py`

Loads Azure settings from `.env` (via `python-dotenv`), builds a small config object (endpoint, key, model name, etc.), and passes it when creating the SDK client (Document Intelligence, GPT-4o, GPT-4o mini, or embeddings). Phase code calls the `create_*_client()` helpers rather than reading env vars directly.

For GPT-4o, `create_llm_gpt_4o_client` can return an `AsyncAzureOpenAI` client (`async_client=True`). Phase 2 uses this in the FastAPI API so LLM calls do not block the event loop while waiting on Azure, allowing other users’ requests to be handled concurrently.

### `config/logger.py`

All modules use `get_logger(__name__)`. Log lines are written to stdout in this format:

`YYYY-MM-DD HH:MM:SS | module:function | LEVEL | message`

---

## Phase 1 — National Insurance accident form extraction

**Goal:** Extract structured JSON from ביטוח לאומי (National Insurance) work-injury forms (PDF or image), using Azure Document Intelligence and Azure OpenAI, with validation and a simple upload UI.

### 1.1 Overview

Phase 1 automates extraction of structured data from Israeli National Insurance (ביטוח לאומי) work-injury accident forms. The user uploads a single PDF or image (PNG/JPG) through a Streamlit app; the backend runs a fixed three-step pipeline and returns JSON aligned with the assignment schema (personal details, address, injury information, medical-institution fields, and dates).

**Layer 1 — OCR:** Azure Document Intelligence (`prebuilt-layout`) turns the document into plain text while preserving reading order. This text is the only input to the LLM steps.

**Layer 2 — Extraction:** GPT-4o mini fills the target JSON in three calls: two parallel schema-focused passes (personal/administrative fields, then accident/medical fields) merged into one draft, followed by a refinement pass on fields that are often misread (e.g. health fund, gender, nature of accident). Missing or unclear values are left as empty strings, as required.

**Layer 3 — Validation:** Rule-based checks normalize phones, dates, ID numbers, postal codes, and allowed Hebrew labels, then produce a validation report (logged with the extraction). The UI displays the final `extracted_fields` JSON.

The implementation lives under `phase_1/src/`. Azure access goes through the shared `config/` package (see above).

### 1.2 Modules (`phase_1/src/`)

| Module | Role |
|--------|------|
| `app.py` | Streamlit UI: file upload, run pipeline, show JSON result |
| `parsing.py` | Orchestrates the full pipeline; runs OCR (Layer 1) and wires layers 2–3 |
| `llm_extraction.py` | GPT-4o mini calls: personal fields, accident fields, merge, refinement |
| `validation.py` | Normalize extracted values and run validation rules; build validation report |
| `prompts.py` | System and user prompts for extraction and refinement passes |
| `constants.py` | JSON schemas, Document Intelligence settings, regex patterns, allowed field values |

### 1.3 Architecture

Phase 1 combines three building blocks:

| Component | Technology | Role |
|-----------|------------|------|
| UI | Streamlit (`app.py`) | Upload a single form (PDF/PNG/JPG) and display the extracted JSON |
| OCR | Azure Document Intelligence | Turn the document into plain text |
| Extraction | **GPT-4o mini** | Map OCR text to structured JSON |

For field extraction, a smaller model is enough: the task is structured parsing from short OCR output, not open-ended reasoning. GPT-4o mini keeps cost and latency low while remaining accurate on this use case.

**End-to-end flow** (orchestrated in `parsing.py`):

1. **Upload** — The file is received through Streamlit and passed to the pipeline as bytes.
2. **OCR** — Document Intelligence returns the full document text. We send the **entire text** to the LLM on every call. There is no chunking or retrieval step here—no cost/latency optimization is needed at this stage because of its short length.
3. **LLM extraction (3 calls)** — All calls use the full OCR text:
   - **Call 1** — Personal and administrative fields (name, ID, address, phones, etc.).
   - **Call 2** — Accident and medical-institution fields (injury date, description, health fund, etc.).
   - The two JSON objects are **merged** into one draft.
   - **Call 3** — Refinement on fields that are hard or ambiguous to read (e.g. health fund, gender, nature of accident), using the draft plus OCR context again.
4. **Validation** — The merged JSON goes through a rule-based pipeline (`validation.py`) that checks formats (dates, ID, phones, postal code, allowed Hebrew values), normalizes values where possible, and flags issues. This step verifies that parsing stayed consistent and fixes common OCR/LLM mistakes.

Conformance checks (valid ID length, phone patterns, allowed health-fund labels, etc.) are implemented as explicit functions, not as another LLM prompt. The model is used to read the form; deterministic rules decide whether a value is acceptable. That separation reduces the risk of the model “fixing” or inventing plausible-looking values that are not actually on the document.

**Why split the LLM work?** One monolithic prompt would ask the model to do too much at once. Separate passes let each call focus on a single schema and task, which reduces confusion and improves reliability on dense forms.

**Temperature (extraction vs. refinement):** The first two LLM calls use **temperature 0** (`TEMPERATURE` in `constants.py`). Those steps are pure extraction: the model should transcribe what appears on the form into JSON, not interpret or embellish. Zero temperature suppresses randomness and reduces hallucinated fields or “helpful” guesses when the OCR line is unclear.

The refinement call uses **temperature 0.25** (`REFINEMENT_TEMPERATURE`). That pass only revisits a small set of ambiguous fields (health fund, gender, nature of accident) where the draft and OCR may disagree or where handwriting/checkboxes are hard to read. A slightly higher temperature gives the model room to weigh conflicting cues in the full OCR text and pick the reading that best fits the document—without opening the door to creative rewriting of the whole form. Anything it returns still goes through the deterministic validation layer, so increased flexibility in refinement does not replace hard format checks.

### 1.4 How to run Phase 1

From the repository root:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create or edit `.env` at the project root and fill in all required Azure API keys and endpoints (Document Intelligence and GPT-4o mini for Phase 1). See `config/auth.py` for the variable names.
3. Start the Streamlit app:
   ```bash
   streamlit run phase_1/src/app.py
   ```

Upload a form in the browser, click **Parse Form**, and view the extracted JSON.

---

## Phase 2 — Health fund chatbot

*(This section will be written after Phase 1 is complete.)*
