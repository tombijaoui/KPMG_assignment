from __future__ import annotations

import re
from typing import Any

from config.logger import get_logger
from phase_2.src.constants import (
    ALLOWED_GENDERS,
    ALLOWED_HMO,
    ALLOWED_INSURANCE_TIERS,
    GENDER_TO_HEBREW,
)
from phase_2.src.models import ProfilePatch, UserProfile

logger = get_logger(__name__)

_DIGITS_ONLY = re.compile(r"\D")
_PROFILE_FIELDS = (
    "first_name",
    "last_name",
    "id_number",
    "gender",
    "age",
    "hmo",
    "hmo_card_number",
    "insurance_tier",
)


def _normalize_digits(value: str, *, length: int) -> str | None:
    digits = _DIGITS_ONLY.sub("", value.strip())
    if len(digits) != length:
        return None
    return digits


def _normalize_name(value: str) -> str | None:
    cleaned = " ".join(value.strip().split())
    if len(cleaned) < 2:
        return None
    return cleaned


def _normalize_gender(value: str) -> str | None:
    raw = value.strip()
    if not raw:
        return None
    if raw in GENDER_TO_HEBREW:
        return GENDER_TO_HEBREW[raw]
    lowered = raw.lower()
    if lowered in GENDER_TO_HEBREW:
        return GENDER_TO_HEBREW[lowered]
    return None


def _normalize_hmo(value: str) -> str | None:
    raw = value.strip()
    if raw in ALLOWED_HMO:
        return raw
    for name in ALLOWED_HMO:
        if name in raw:
            return name
    return None


def _normalize_tier(value: str) -> str | None:
    raw = value.strip()
    if raw in ALLOWED_INSURANCE_TIERS:
        return raw
    for tier in ALLOWED_INSURANCE_TIERS:
        if tier in raw:
            return tier
    return None


def validate_profile_field(field: str, value: Any) -> Any | None:
    """Return a normalized value or None if the field fails validation."""
    if value is None:
        return None

    if field in {"first_name", "last_name"}:
        if not isinstance(value, str):
            logger.info(
                "Validation rejected field=%s raw=%r reason=expected string",
                field,
                value,
            )
            return None
        result = _normalize_name(value)
        if result is None:
            logger.info(
                "Validation rejected field=%s raw=%r reason=name too short",
                field,
                value,
            )
            return None
        logger.info("Validation accepted field=%s value=%r", field, result)
        return result

    if field in {"id_number", "hmo_card_number"}:
        if isinstance(value, int):
            value = str(value)
        if not isinstance(value, str):
            logger.info(
                "Validation rejected field=%s raw=%r reason=expected string",
                field,
                value,
            )
            return None
        result = _normalize_digits(value, length=9)
        if result is None:
            digit_count = len(_DIGITS_ONLY.sub("", value.strip()))
            logger.info(
                "Validation rejected field=%s raw=%r reason=expected 9 digits got %s",
                field,
                value,
                digit_count,
            )
            return None
        logger.info("Validation accepted field=%s value=%r", field, result)
        return result

    if field == "gender":
        if not isinstance(value, str):
            logger.info(
                "Validation rejected field=%s raw=%r reason=expected string",
                field,
                value,
            )
            return None
        result = _normalize_gender(value)
        if result is None:
            logger.info(
                "Validation rejected field=%s raw=%r reason=expected זכר or נקבה",
                field,
                value,
            )
            return None
        logger.info("Validation accepted field=%s value=%r", field, result)
        return result

    if field == "age":
        try:
            age = int(value)
        except (TypeError, ValueError):
            logger.info(
                "Validation rejected field=%s raw=%r reason=expected integer age",
                field,
                value,
            )
            return None
        if 0 <= age <= 120:
            logger.info("Validation accepted field=%s value=%s", field, age)
            return age
        logger.info(
            "Validation rejected field=%s raw=%r reason=age must be between 0 and 120",
            field,
            value,
        )
        return None

    if field == "hmo":
        if not isinstance(value, str):
            logger.info(
                "Validation rejected field=%s raw=%r reason=expected string",
                field,
                value,
            )
            return None
        result = _normalize_hmo(value)
        if result is None:
            logger.info(
                "Validation rejected field=%s raw=%r reason=expected one of %s",
                field,
                value,
                sorted(ALLOWED_HMO),
            )
            return None
        logger.info("Validation accepted field=%s value=%r", field, result)
        return result

    if field == "insurance_tier":
        if not isinstance(value, str):
            logger.info(
                "Validation rejected field=%s raw=%r reason=expected string",
                field,
                value,
            )
            return None
        result = _normalize_tier(value)
        if result is None:
            logger.info(
                "Validation rejected field=%s raw=%r reason=expected one of %s",
                field,
                value,
                sorted(ALLOWED_INSURANCE_TIERS),
            )
            return None
        logger.info("Validation accepted field=%s value=%r", field, result)
        return result

    logger.warning("Validation skipped unknown field=%s raw=%r", field, value)
    return None


def merge_profile(current: UserProfile, patch: ProfilePatch) -> UserProfile:
    """Apply validated patch fields onto the current profile."""
    data = current.model_dump()
    patch_data = patch.model_dump(exclude_none=True)

    if not patch_data:
        logger.info("merge_profile: empty patch, profile unchanged")
        return UserProfile.model_validate(data)

    logger.info("merge_profile: patch fields to apply=%s", list(patch_data.keys()))

    applied: list[str] = []
    rejected: list[str] = []

    for field, raw_value in patch_data.items():
        if field not in _PROFILE_FIELDS:
            logger.warning("merge_profile: ignoring unknown patch field=%s", field)
            continue
        before = data.get(field)
        normalized = validate_profile_field(field, raw_value)
        if normalized is None:
            rejected.append(field)
            continue
        data[field] = normalized
        applied.append(field)
        if before != normalized:
            logger.info(
                "merge_profile: updated field=%s from %r to %r",
                field,
                before,
                normalized,
            )

    logger.info(
        "merge_profile: done applied=%s rejected=%s",
        applied or "(none)",
        rejected or "(none)",
    )
    return UserProfile.model_validate(data)


def is_profile_complete(profile: UserProfile) -> bool:
    """True when every required field is present and valid."""
    missing: list[str] = []
    for field in _PROFILE_FIELDS:
        value = getattr(profile, field)
        if value is None:
            missing.append(field)
            continue
        if validate_profile_field(field, value) is None:
            missing.append(field)

    complete = not missing
    if complete:
        logger.info("Profile completeness check: complete=True (all %s fields valid)", len(_PROFILE_FIELDS))
    else:
        logger.info(
            "Profile completeness check: complete=False missing=%s",
            missing,
        )
    return complete
