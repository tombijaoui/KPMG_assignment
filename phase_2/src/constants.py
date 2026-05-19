from __future__ import annotations

import os

COLLECT_LLM_TEMPERATURE = 0.0

# Streamlit UI / API client
API_BASE_URL = os.getenv("PHASE_2_API_BASE_URL", "http://localhost:8000").rstrip("/")
COLLECT_INFO_URL = f"{API_BASE_URL}/collect-info"

MIN_COMPOSER_HEIGHT = 52
MAX_COMPOSER_HEIGHT = 140
CHARS_PER_LINE = 52
MAX_VISIBLE_LINES = 5
STREAM_WORD_DELAY_SEC = 0.04

# Recent chat turns sent only when short replies need context (yes / corrections).
COLLECT_RECENT_MESSAGE_LIMIT = 6

COLLECT_LLM_MAX_RETRIES = 2
COLLECT_LLM_RETRY_DELAY_SEC = 1.0

ALLOWED_HMO = frozenset({"מכבי", "מאוחדת", "כללית"})
ALLOWED_INSURANCE_TIERS = frozenset({"זהב", "כסף", "ארד"})
# Canonical gender labels stored in the profile.
GENDER_MALE_HE = "זכר"
GENDER_FEMALE_HE = "נקבה"

# Hebrew synonyms normalized to זכר / נקבה (e.g. גבר, אישה).
GENDER_HEBREW_SYNONYMS_TO_CANONICAL: dict[str, str] = {
    "זכר": GENDER_MALE_HE,
    "גבר": GENDER_MALE_HE,
    "איש": GENDER_MALE_HE,
    "נקבה": GENDER_FEMALE_HE,
    "אישה": GENDER_FEMALE_HE,
    "אשה": GENDER_FEMALE_HE,
}

GENDER_ENGLISH_TO_CANONICAL: dict[str, str] = {
    "male": GENDER_MALE_HE,
    "female": GENDER_FEMALE_HE,
    "masculine": GENDER_MALE_HE,
    "feminine": GENDER_FEMALE_HE,
    "m": GENDER_MALE_HE,
    "f": GENDER_FEMALE_HE,
    "masc": GENDER_MALE_HE,
    "fem": GENDER_FEMALE_HE,
    "man": GENDER_MALE_HE,
    "woman": GENDER_FEMALE_HE,
    "boy": GENDER_MALE_HE,
    "girl": GENDER_FEMALE_HE,
    "homme": GENDER_MALE_HE,
    "femme": GENDER_FEMALE_HE,
}

GENDER_TO_HEBREW = {**GENDER_HEBREW_SYNONYMS_TO_CANONICAL, **GENDER_ENGLISH_TO_CANONICAL}

ALLOWED_GENDERS = frozenset(GENDER_TO_HEBREW.keys())
