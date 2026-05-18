from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# --- Document Intelligence constants -------------------------------------------------

LAYOUT_MODEL_ID = "prebuilt-layout"

CONTENT_TYPE_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

# --- LLM extraction ----------------------------------------------------------------

TEMPERATURE = 0
REFINEMENT_TEMPERATURE = 0.25

EMPTY_DATE: dict[str, str] = {"day": "", "month": "", "year": ""}

PERSONAL_INFO_SCHEMA: dict[str, Any] = {
    "lastName": "",
    "firstName": "",
    "idNumber": "",
    "gender": "",
    "dateOfBirth": dict(EMPTY_DATE),
    "address": {
        "street": "",
        "houseNumber": "",
        "entrance": "",
        "apartment": "",
        "city": "",
        "postalCode": "",
        "poBox": "",
    },
    "landlinePhone": "",
    "mobilePhone": "",
    "jobType": "",
    "formFillingDate": dict(EMPTY_DATE),
    "formReceiptDateAtClinic": dict(EMPTY_DATE),
    "signature": "",
}

ACCIDENT_DETAILS_SCHEMA: dict[str, Any] = {
    "dateOfInjury": dict(EMPTY_DATE),
    "timeOfInjury": "",
    "accidentLocation": "",
    "accidentAddress": "",
    "accidentDescription": "",
    "injuredBodyPart": "",
    "medicalInstitutionFields": {
        "healthFundMember": "",
        "natureOfAccident": "",
        "medicalDiagnoses": "",
    },
}

REFINEMENT_FOCUS_SCHEMA: dict[str, Any] = {
    "gender": "",
    "medicalInstitutionFields": {
        "healthFundMember": "",
        "natureOfAccident": "",
    },
}

# --- Validation / normalization ----------------------------------------------------

DIGITS_ONLY = re.compile(r"\D")
MOBILE_PHONE_PATTERN = re.compile(r"^05\d{8}$")
POSTAL_CODE_PATTERN = re.compile(r"^\d{6,7}$")
ID_PATTERN = re.compile(r"^\d{9,10}$")

ALLOWED_GENDERS_HE = frozenset({"זכר", "נקבה"})
ALLOWED_GENDER_VALUES = frozenset({"זכר", "נקבה", ""})
ALLOWED_HEALTH_FUNDS_HE = frozenset({"כללית", "מאוחדת", "מכבי", "לאומית"})
ALLOWED_HEALTH_FUND_VALUES = frozenset({"כללית", "מאוחדת", "מכבי", "לאומית", ""})
HEALTH_FUNDS_BY_LENGTH = sorted(ALLOWED_HEALTH_FUNDS_HE, key=len, reverse=True)
