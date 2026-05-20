# KPMG GenAI Assignment — Implementation Notes

This document describes how the two assignment phases were implemented: design choices, pipeline steps, and how to run each part locally. Content is added incrementally; sections marked below will be filled in as we go.

---

## Table of contents

1. [Shared configuration (`config/`)](#shared-configuration-config)
2. [Phase 1 — National Insurance accident form extraction](#phase-1--national-insurance-accident-form-extraction)
3. [Phase 2 — Health fund chatbot (HMO assistant)](#phase-2--health-fund-chatbot-hmo-assistant)

---

## Shared configuration (`config/`)

Both phases rely on a small shared package at the repository root. It centralizes Azure credentials, SDK client factories, and logging so phase-specific code stays focused on business logic.

Credentials are loaded once from a `.env` file at the project root (via `python-dotenv`). The `.env` file itself is committed to Git without the API keys, endpoints and API versions of the models. When you clone the repo, you have to fill by yourself the different API keys, API versions (available at the end of the endpoint as a parameter) and the different endpoints of the different models

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
2. Edit `.env` at the project root and fill in all required Azure API keys, API versions and endpoints (Document Intelligence and GPT-4o mini for Phase 1). See `config/auth.py` for the variable names.
3. Start the Streamlit app:
   ```bash
   streamlit run phase_1/src/app.py
   ```

Upload a form in the browser, click **Parse Form**, and view the extracted JSON.

---

## Phase 2 — Health fund chatbot (HMO assistant)

**Goal:** A conversational assistant for Israeli health-fund members. It first collects a member profile (name, ID, HMO, insurance tier, etc.), then answers questions about services, coverage, and contact details using a retrieval-augmented knowledge base built from the assignment HTML files.

### 2.1 Overview

Phase 2 is split into two chat modes, exposed by a FastAPI backend and used from a Streamlit chat UI.

**Onboarding (`/collect-info`):** GPT-4o conducts a short dialogue to fill a structured profile (מכבי / מאוחדת / כללית, tier זהב / כסף / ארד, 9-digit IDs, etc.). The model returns a natural-language reply plus a JSON patch; the server merges and validates fields with deterministic rules (same philosophy as Phase 1—rules enforce format, the LLM reads the conversation). The member must explicitly confirm the full profile before Q&A unlocks.

**Q&A (`/qa`):** After confirmation, GPT-4o answers in Hebrew or English. When factual HMO information is needed, it calls the `search_hmo_knowledge` tool. Retrieval runs over a **FAISS** index built from chunked HTML (`phase_2/data`): chunks are embedded with **text-embedding-ada-002**, indexed at startup (built automatically if missing), and filtered by the member’s HMO and insurance tier so answers stay scoped to their fund.

The API uses an **async** GPT-4o client so long LLM calls do not block other requests. The Streamlit app (`phase_2/src/app.py`) is a thin client: it keeps chat state and calls `http://localhost:8000` (`/collect-info` then `/qa`).

### 2.2 Modules (`phase_2/src/`)

| Module | Role |
|--------|------|
| `app.py` | Streamlit chat UI: onboarding then Q&A, calls the FastAPI backend |
| `api.py` | FastAPI app: `/collect-info`, `/qa`, `/health`; loads or build FAISS index + async OpenAI on startup |
| `models.py` | Pydantic models for requests, responses, profile, and tool-call payloads |
| `prompts.py` | System prompts and message builders for collect and Q&A phases |
| `validation.py` | Normalize profile fields, merge patches, check profile completeness |
| `constants.py` | API URLs, UI limits, allowed HMOs/tiers/genders, chunk types, retrieval defaults |
| `chunking.py` | Parse HTML knowledge files into typed chunks; batch embedding via ada-002 |
| `indexing.py` | Build, save, and load FAISS index + `chunks.json` under `phase_2/knowledge_base/` |
| `retrieving.py` | Semantic search, metadata filter (HMO + tier), tool implementation for the LLM |
| `tools.py` | OpenAI function schema for `search_hmo_knowledge` |

### 2.3 Architecture and pipeline

The **Q&A chatbot is a RAG system** (Retrieval-Augmented Generation): GPT-4o does not answer from memory alone. Factual replies are grounded in passages retrieved from the HMO knowledge base built from the assignment HTML files. Onboarding (profile collection) is a standard guided dialogue; **RAG applies only to the answer phase.**

**RAG architecture (high level):**

```text
                    ┌─────────────────────────┐
  Offline (once)    │  HTML → chunks → embed  │
                    │  → FAISS + chunks.json  │
                    └───────────┬─────────────┘
                                │
  Online (per question)         ▼
  User question ──► GPT-4o ──► needs facts? ──no──► direct reply
                        │ yes
                        ▼
              embed query + filter by profile (HMO, tier)
                        │
                        ▼
              top-3 chunks from vector index
                        │
                        ▼
              GPT-4o reads passages + profile → final answer
```

| Piece | Role |
|-------|------|
| **Index** | Stores embedded chunks from all HMO HTML sources (see [§2.4](#24-knowledge-base--scraping-chunking-embeddings-and-index)) |
| **Retriever** | Finds the best passages for the question, scoped to the member’s fund and tier (see [§2.5](#25-retrieval--personalized-search-over-the-knowledge-base)) |
| **Generator** | GPT-4o with the confirmed profile in the system prompt and retrieved text as tool context |
| **Orchestration** | The model **chooses** when to search (tool call), so simple messages skip retrieval; precise questions trigger lookup |

This is **agentic RAG**: retrieval is not run on every turn—only when the LLM decides it needs knowledge-base evidence. That keeps latency low for chit-chat and focuses context on real service or coverage questions.

#### When the API starts

On launch, the backend prepares two things before accepting chat traffic:

- An **asynchronous** connection to GPT-4o, so while one user waits for the model, others can still be served.
- The **vector index** over the HMO knowledge base. If it does not exist yet, it is built first; this work runs in a background thread so startup stays as responsive as possible.

#### How Streamlit and the API work together

The chat UI only talks to the REST API—it never calls Azure directly. The **member profile JSON is kept in the browser session**: after each onboarding reply, the UI stores the updated profile returned by the server. The API does not remember users between calls; every request carries everything needed (current profile, recent chat, latest message). That keeps the design **stateless on the server**.

**Moving from onboarding to Q&A** — The UI switches endpoints automatically using status returned by the API: whether the profile is complete and correctly formatted, whether the user has confirmed it, and whether Q&A is allowed. Until all conditions are met, messages go to the collect endpoint; afterwards, only the Q&A endpoint is used. Answer history is tracked separately from the onboarding conversation.

#### Phase A — Collecting the profile

1. The user types in Streamlit; the app calls the **collect-info** endpoint.
2. Each call sends the **latest message**, the **profile built so far**, and a **short window of recent dialogue** so the model can understand confirmations like “yes” or “כן” after reading back an ID or card number.
3. GPT-4o runs at **temperature 0**: the job is to understand and record data, not to improvise.
4. The server **merges** what the model extracted with the existing profile and runs **validation rules** (format of IDs, phones, allowed HMO names and tiers, etc.). Bad values are corrected or dropped.
5. **ID and card numbers** are read back to the user for immediate confirmation if they might be wrong. When all fields are filled, the assistant presents a **full summary** and waits for the user to approve it—human-in-the-loop before any Q&A.
6. The UI saves the new profile and status flags for the next turn. The new status flags allow the UI to only call the **qa** endpoint.

#### Phase B — Answering questions

1. Q&A opens only after the user has confirmed a complete, valid profile.
2. Each call sends the **new question**, the **confirmed profile** (so answers stay tied to the right HMO and tier), and **recent Q&A history** so follow-up questions make sense.
3. GPT-4o uses **temperature 0.2**—slightly more flexible wording than during collection, while still relying on facts rather than invention.
4. **Search is optional and model-driven.** The LLM can query the knowledge base when it needs concrete details (coverage, benefits, phone numbers). For greetings or generic chat, it can reply without searching.
5. When search runs, results are limited to the member’s **HMO and insurance tier**. Only the **top three** relevant passages are used—enough for a specific fact without overloading the prompt.
6. The model turns those passages into a final answer; the UI keeps the exchange for the next question.

```text
Startup:  prepare async LLM + load or build vector index
Collect:  UI → collect-info → GPT-4o (strict) → validate → update profile in UI
Q&A:      UI → qa → GPT-4o (slightly warmer) → optional knowledge search → answer
```

### 2.4 Knowledge base — scraping, chunking, embeddings, and index

The Q&A assistant does not read raw HTML at query time. Offline, we turn the assignment files in `phase_2/data/` into a searchable vector index under `phase_2/knowledge_base/`.

#### From HTML to chunks (scraping / parsing)

We parse the static HTML with BeautifulSoup.

For every file we build several kinds of **chunks** (short text passages), each with **metadata** used later to filter results by HMO and insurance tier:

| Chunk type | What it contains | Metadata (typical) |
|------------|------------------|--------------------|
| Intro | Opening paragraphs before the table | Applies to all HMOs and all tiers |
| Service overview | Bullet list of services covered in that domain | All HMOs, all tiers |
| Coverage | One benefit description for a specific service, HMO, and tier (זהב / כסף / ארד) | Domain, service name, single HMO, single tier |
| Contact phone | Customer-service numbers per HMO | Domain, single HMO, all tiers |
| Contact details | Addresses / extra contact info per HMO | Domain, single HMO, all tiers |

**Table handling** — The main table has HMO names in the header row (מכבי, מאוחדת, כללית). Each body row is a medical service; each HMO column cell is split into up to three tier sections (marked with bold labels). One chunk is created per non-empty tier benefit, with text that states domain, service, HMO, tier, and the benefit wording so retrieval and the LLM see full context.

**Contact sections** — Blocks under headings that mention phone numbers or contact details are parsed from lists; the HMO is inferred from the start of each line (Hebrew fund name).

Chunk text is plain, predictable prose—not raw HTML—so embeddings capture meaning rather than markup noise.

#### Embeddings

Every chunk’s text is sent to Azure **text-embedding-ada-002** in batches (default 64 texts per API call). Each chunk receives a fixed-size vector stored alongside its text until the index is written. If embedding fails or counts do not match, indexing stops rather than saving a broken base.

#### Building and storing the FAISS index

All vectors are stacked into a matrix, **normalized** (so similarity behaves like cosine similarity), and added to a **FAISS** flat inner-product index—one vector per chunk, in the same order as the chunk list.

On disk we keep three artifacts in `phase_2/knowledge_base/`:

| File | Contents |
|------|----------|
| `faiss.index` | Vector index for fast similarity search |
| `chunks.json` | Chunk id, text, and metadata (vectors are only in FAISS) |
| `manifest.json` | Embedding model name, dimension, chunk count, index settings |

**Loading at API startup** — When the backend starts, loading or building the index runs in a **separate worker thread** (using `asyncio.to_thread`). Parsing HTML, calling the embedding API, and writing FAISS are CPU- and I/O-heavy; moving that work off the event loop keeps the server from freezing while the index is prepared. This matters especially because the **collect-info** phase (profile onboarding) does **not** use the vector index at all—only **Q&A / RAG** does. The index is therefore an independent concern from onboarding: we prepare it once at startup for later retrieval, without tying it to how member data is collected.

If `phase_2/knowledge_base/` already exists, the files are loaded; otherwise the full pipeline runs once (parse → embed → build → save), then the API is ready to serve.

At answer time, the user’s question is embedded with the same model, compared against the index, and the best matches are filtered by the member’s HMO and tier before the top passages are sent to GPT-4o.

```text
phase_2/data/*.html  →  parse & chunk (+ metadata)
                    →  embed (ada-002, batched)
                    →  FAISS index + chunks.json + manifest
                    →  load at API startup (or build if missing)
```

### 2.5 Retrieval — personalized search over the knowledge base

When the Q&A model needs factual HMO information, it issues a **search query** (in the user’s language). Retrieval is not a blind search over the whole index: results are **scoped to the logged-in member** using the profile collected earlier.

#### Step 1 — Metadata filter (personalization)

Before similarity scoring, we keep only chunks whose metadata **matches the member’s HMO and insurance tier**:

- **Coverage chunks** are tagged with one fund (מכבי, מאוחדת, כללית) and one tier (זהב, כסף, ארד). A כללית / כסף member never sees מכבי-only benefits in retrieval.
- **Intro and service-overview** chunks are tagged as applying to **all** funds and tiers, so general domain context remains available to everyone.
- **Contact** chunks are tagged per HMO but all tiers, so phone numbers and addresses stay fund-specific without mixing competitors.

If the profile is missing HMO or tier, or no chunk passes the filter, search returns nothing and the model is told that no entries matched—avoiding generic or wrong-fund answers.

This filter is the main personalization layer: vector similarity ranks relevance **within** the member’s world, not across the entire market.

#### Step 2 — Semantic ranking

The search phrase is embedded with the **same model** used to build the index. We compare it only against vectors for the filtered chunks (cosine similarity via normalized vectors), then take the **top three** hits.

Each hit includes the chunk text, metadata (domain, service name when relevant), and a similarity score. Passages are formatted as numbered context blocks for GPT-4o, separated clearly so the model can cite grounded facts in its reply.

#### Why this two-step design?

| Approach | Risk |
|----------|------|
| Similarity only, no metadata filter | High-scoring chunk might describe another HMO’s coverage |
| Metadata only, no similarity | Cannot pick the best passage for the actual question |
| Filter then similarity | Answers stay **personal** (right fund and tier) and **relevant** (right topic) |

The member’s profile therefore drives both **what** can be retrieved and **which** passages rank highest—without storing separate indexes per HMO.

```text
User question → LLM chooses to search → embed query
              → keep chunks where metadata matches profile (HMO + tier)
              → rank by similarity → top 3 → context for final answer
```

### 2.6 How to run Phase 2

Phase 2 needs **two terminals** (API first, then the chat UI).

From the repository root:

**Terminal 1 — API**

1. Install dependencies (if not already done):
   ```bash
   pip install -r requirements.txt
   ```
2. Edit `.env` at the project root and fill in all Azure API keys, API versions and endpoints.
3. Start the backend:
   ```bash
   uvicorn phase_2.src.api:app --reload --port 8000
   ```
   On first startup the knowledge base may be built automatically if `phase_2/knowledge_base/` is missing—wait until the server is ready.

**Terminal 2 — Streamlit chat**

```bash
streamlit run phase_2/src/app.py
```

Complete onboarding in the browser until the profile is confirmed, then ask questions about your HMO and insurance tier. The UI calls `http://localhost:8000` by default (`PHASE_2_API_BASE_URL` in `.env` can override this).
