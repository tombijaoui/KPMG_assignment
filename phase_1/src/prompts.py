from __future__ import annotations

# --- LLM Call 1 & 2: layer 2a & 2b: extraction (shared system prompt) -----------------------------------

EXTRACTION_SYSTEM_PROMPT = """You extract data from Israeli National Insurance (Bituah Leumi) work-injury forms.
The OCR text may be in Hebrew or English. Checkbox markers appear as :selected: and :unselected:.
Rules:
- Return valid JSON only, matching the exact schema provided.
- Use empty string "" for any missing or unreadable field.
- For date objects, split values into day, month, year as strings (e.g. "16", "04", "2022").
- Do not invent values not supported by the OCR text.
- Health fund (healthFundMember): each fund name is paired with the marker immediately next to it on the fund row. Only the fund that has :selected: beside it counts (not a fund that only has :unselected: beside it).
- Map :selected: / :unselected: to the corresponding field values for other checkbox fields (e.g. gender)."""

PERSONAL_EXTRACTION_TASK = (
    "Extract only personal information and form dates/signature from the OCR text."
)

ACCIDENT_EXTRACTION_TASK = (
    "Extract only accident-related and medical-institution fields from the OCR text."
)


def build_extraction_user_message(task: str, schema_json: str, ocr_text: str) -> str:
    """User message for pass 1 or 2 (schema slice + OCR)."""
    return (
        f"{task}\n\n"
        f"JSON schema to fill:\n{schema_json}\n\n"
        f"OCR text:\n{ocr_text}"
    )


# --- LLM Call 3: layer 2c: refinement ------------------------------------------------------------

REFINEMENT_SYSTEM_PROMPT = (
    "You output only corrected JSON for Israeli work-injury forms. No prose."
)

REFINEMENT_RULES = """You refine a draft JSON extracted from an Israeli National Insurance (Bituah Leumi) work-injury form.
The draft may contain errors on delicate fields. Use the OCR text as ground truth and correct the draft.

Health fund row (medicalInstitutionFields.healthFundMember):
- Allowed values: exactly one of כללית, מאוחדת, מכבי, לאומית, or "" if none can be determined.
- Each of the four fund names is paired only with the checkbox marker that sits immediately next to it (before or after the name on the same row). The selected fund is the one whose adjacent marker is :selected:.
- Disambiguation example — OCR may contain a substring like:
  5 למילוי ע״י המוסד הרפואי :unselected: לאומית :unselected: מכבי :selected: מאוחדת :unselected: כללית :unselected: הנפגע חבר בקופת חולים :unselected: הנפגע אינו חבר בקופת חולים :unselected: מהות התאונה (אבחנות רפואיות)
  In that fragment, healthFundMember must be מאוחדת (the :selected: marker is next to מאוחדת). It must NOT be מכבי: מכבי is next to :unselected:. Text after the four funds (e.g. הנפגע חבר בקופת חולים) is a different question; do not use those markers to infer the fund row.
- If no fund among the four has :selected: beside it, use "".

Other refinements:
- Gender: only זכר or נקבה when :selected: clearly pairs with that word in the OCR; otherwise align with draft if consistent.
- natureOfAccident / accident type lines: prefer the option whose label is clearly paired with :selected: in the OCR.
- If OCR is ambiguous, keep the draft value; only change when the OCR clearly contradicts the draft.

Output: valid JSON only. Include ONLY the keys and nesting shown in the refinement target schema (no other top-level fields). Use "" when a value cannot be determined."""


REFINEMENT_USER_INTRO = (
    "Refine the draft JSON using the OCR text. Fix errors on delicate fields "
    "(health fund checkboxes, gender, accident type lines) using the rules below."
)


def build_refinement_user_message(
    focus_schema_json: str, focus_draft_json: str, ocr_text: str
) -> str:
    """User message for pass 3 (refinement target schema + draft slice for those fields + OCR)."""
    return (
        f"{REFINEMENT_USER_INTRO}\n\n"
        f"Rules:\n{REFINEMENT_RULES}\n\n"
        f"Refinement target schema (output ONLY these keys and nesting):\n{focus_schema_json}\n\n"
        f"Current values for those fields from the draft:\n{focus_draft_json}\n\n"
        f"OCR text:\n{ocr_text}"
    )
