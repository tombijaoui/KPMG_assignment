from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config.logger import get_logger
from phase_1.src.constants import (
    ALLOWED_GENDER_VALUES,
    ALLOWED_GENDERS_HE,
    ALLOWED_HEALTH_FUND_VALUES,
    ALLOWED_HEALTH_FUNDS_HE,
    DIGITS_ONLY,
    HEALTH_FUNDS_BY_LENGTH,
    ID_PATTERN,
    MOBILE_PHONE_PATTERN,
    POSTAL_CODE_PATTERN,
)

logger = get_logger(__name__)


# --- Normalization -----------------------------------------------------------------


def normalize_phone_number(value: str) -> str:
    """
    Normalize an Israeli phone number to digits only.

    OCR may read a leading 0 as 6 when the box has a pre-printed arc.
    If the number does not start with 0, the first digit is replaced with 0.
    """
    if not value or not str(value).strip():
        return ""

    digits = DIGITS_ONLY.sub("", str(value).strip())
    if not digits:
        return ""

    if not digits.startswith("0"):
        corrected = "0" + digits[1:]

        logger.info(
            "Phone number normalized: leading digit corrected to 0 (was %s)",
            digits[0],
        )
        digits = corrected

    return digits


def normalize_gender(value: str) -> str:
    """
    Keep only Hebrew male (זכר) or female (נקבה); anything else becomes empty.

    Unknown English labels, OCR noise, or free text are cleared to "".
    """
    raw = str(value or "").strip()

    if not raw:
        return ""

    if raw in ALLOWED_GENDERS_HE:
        return raw

    logger.info("Gender normalized to empty (value was not זכר or נקבה)")
    return ""


def normalize_health_fund_member(value: str) -> str:
    """
    Keep only a recognized Israeli health fund name in Hebrew; else empty.

    If the string contains a known fund name as a substring (e.g. checkbox noise),
    that canonical name is returned; otherwise "".
    """
    raw = str(value or "").strip()

    if not raw:
        return ""

    if raw in ALLOWED_HEALTH_FUNDS_HE:
        return raw

    for fund in HEALTH_FUNDS_BY_LENGTH:
        if fund in raw:
            logger.info("Health fund normalized from noisy OCR text to canonical name")
            return fund

    logger.info("Health fund normalized to empty (unrecognized value)")
    return ""


def normalize_extracted_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Apply field-level normalization to the extracted JSON payload."""
    result = dict(fields)

    for phone_field in ("landlinePhone", "mobilePhone"):
        if phone_field in result:
            result[phone_field] = normalize_phone_number(str(result.get(phone_field, "")))

    if "gender" in result:
        result["gender"] = normalize_gender(str(result.get("gender", "")))

    medical = result.get("medicalInstitutionFields")
    if isinstance(medical, dict) and "healthFundMember" in medical:
        medical = dict(medical)
        medical["healthFundMember"] = normalize_health_fund_member(
            str(medical.get("healthFundMember", ""))
        )
        result["medicalInstitutionFields"] = medical

    return result


# --- Validation types --------------------------------------------------------------


@dataclass
class FieldCheckResult:
    """Result of a single field validation."""
    valid: bool
    field_path: str
    message: str = ""


@dataclass
class ValidationPipelineResult:
    """Aggregated validation output."""
    checks: list[FieldCheckResult] = field(default_factory=list)
    is_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "checks": [
                {
                    "valid": c.valid,
                    "field_path": c.field_path,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }


def check_date_validity(date_obj: Any, *, field_path: str, allow_all_empty: bool = True) -> FieldCheckResult:
    """
    Validate a date object with day, month, year string fields.

    If all three are empty and ``allow_all_empty`` is True, the date is considered valid (unknown).
    If any part is filled, all three must be filled and form a plausible calendar date.
    """
    if not isinstance(date_obj, dict):
        return FieldCheckResult(
            valid=False,
            field_path=field_path,
            message="Expected an object with day, month, year",
        )

    day = str(date_obj.get("day", "")).strip()
    month = str(date_obj.get("month", "")).strip()
    year = str(date_obj.get("year", "")).strip()

    if not day and not month and not year:
        if allow_all_empty:
            return FieldCheckResult(valid=True, field_path=field_path, message="Empty date (optional)")
        return FieldCheckResult(valid=False, field_path=field_path, message="Date is required")

    if not (day and month and year):
        return FieldCheckResult(
            valid=False,
            field_path=field_path,
            message="Date must have day, month, and year when partially filled",
        )

    if not (day.isdigit() and month.isdigit() and year.isdigit()):
        return FieldCheckResult(
            valid=False,
            field_path=field_path,
            message="Date parts must be numeric strings",
        )

    d, m, y = int(day), int(month), int(year)
    if not (1 <= m <= 12):
        return FieldCheckResult(valid=False, field_path=field_path, message="Invalid month")
    if not (1 <= d <= 31):
        return FieldCheckResult(valid=False, field_path=field_path, message="Invalid day")
    if not (1900 <= y <= 2100):
        return FieldCheckResult(valid=False, field_path=field_path, message="Invalid year")

    days_in_month = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[m - 1]
    if d > days_in_month:
        return FieldCheckResult(valid=False, field_path=field_path, message="Day out of range for month")

    return FieldCheckResult(valid=True, field_path=field_path, message="OK")


def check_id_validity(value: str, *, field_path: str = "idNumber") -> FieldCheckResult:
    """Israeli ID: 9 or 10 digits after stripping non-digits. Empty is allowed (readme)."""
    if not value or not str(value).strip():
        return FieldCheckResult(
            valid=True,
            field_path=field_path,
            message="Empty ID (optional)",
        )

    digits = DIGITS_ONLY.sub("", str(value).strip())
    if ID_PATTERN.fullmatch(digits):
        return FieldCheckResult(valid=True, field_path=field_path, message="OK")

    return FieldCheckResult(
        valid=False,
        field_path=field_path,
        message=f"ID must be 9 or 10 digits, got {len(digits)} digits",
    )


def check_gender_validity(value: str, *, field_path: str = "gender") -> FieldCheckResult:
    """Gender must be Hebrew male (זכר), female (נקבה), or empty (after normalization)."""
    normalized = normalize_gender(value)
    if normalized in ALLOWED_GENDER_VALUES:
        return FieldCheckResult(valid=True, field_path=field_path, message="OK")

    return FieldCheckResult(
        valid=False,
        field_path=field_path,
        message="Gender must be זכר, נקבה, or empty",
    )


def check_health_fund_member_validity(
    value: str,
    *,
    field_path: str = "medicalInstitutionFields.healthFundMember",
) -> FieldCheckResult:
    """Health fund must be one of the four Israeli funds in Hebrew, or empty."""
    normalized = normalize_health_fund_member(value)
    if normalized in ALLOWED_HEALTH_FUND_VALUES:
        return FieldCheckResult(valid=True, field_path=field_path, message="OK")

    return FieldCheckResult(
        valid=False,
        field_path=field_path,
        message="Health fund must be כללית, מאוחדת, מכבי, לאומית, or empty",
    )


def check_mobile_phone_validity(value: str, *, field_path: str = "mobilePhone") -> FieldCheckResult:
    """Mobile: must start with 05 and contain exactly 10 digits."""
    if not value or not str(value).strip():
        return FieldCheckResult(
            valid=True,
            field_path=field_path,
            message="Empty mobile (optional)",
        )

    digits = DIGITS_ONLY.sub("", str(value).strip())
    if MOBILE_PHONE_PATTERN.fullmatch(digits):
        return FieldCheckResult(valid=True, field_path=field_path, message="OK")

    return FieldCheckResult(
        valid=False,
        field_path=field_path,
        message="Mobile must start with 05 and be exactly 10 digits",
    )


def check_landline_phone_validity(value: str, *, field_path: str = "landlinePhone") -> FieldCheckResult:
    """Landline: empty is OK; if set, digits only, starts with 0, 9–10 digits (Israel)."""
    if not value or not str(value).strip():
        return FieldCheckResult(
            valid=True,
            field_path=field_path,
            message="Empty landline (optional)",
        )

    digits = DIGITS_ONLY.sub("", str(value).strip())
    if re.fullmatch(r"0\d{8,9}", digits):
        return FieldCheckResult(valid=True, field_path=field_path, message="OK")

    return FieldCheckResult(
        valid=False,
        field_path=field_path,
        message="Landline must start with 0 and have 9 or 10 digits",
    )


def check_postal_code_validity(value: str, *, field_path: str = "address.postalCode") -> FieldCheckResult:
    """Postal code: empty OK; if set, exactly 6 or 7 digits."""
    if not value or not str(value).strip():
        return FieldCheckResult(
            valid=True,
            field_path=field_path,
            message="Empty postal code (optional)",
        )

    digits = DIGITS_ONLY.sub("", str(value).strip())
    if POSTAL_CODE_PATTERN.fullmatch(digits):
        return FieldCheckResult(valid=True, field_path=field_path, message="OK")

    return FieldCheckResult(
        valid=False,
        field_path=field_path,
        message="Postal code must be 6 or 7 digits",
    )


def run_validation_pipeline(fields: dict[str, Any]) -> ValidationPipelineResult:
    """
    Run all field validators on the extracted readme-shaped JSON.

    Returns aggregated results; ``is_valid`` is False if any check fails.
    """
    checks: list[FieldCheckResult] = []

    date_paths = (
        ("dateOfBirth", fields.get("dateOfBirth")),
        ("dateOfInjury", fields.get("dateOfInjury")),
        ("formFillingDate", fields.get("formFillingDate")),
        ("formReceiptDateAtClinic", fields.get("formReceiptDateAtClinic")),
    )
    for path, obj in date_paths:
        checks.append(check_date_validity(obj, field_path=path))

    checks.append(check_id_validity(str(fields.get("idNumber", "")), field_path="idNumber"))
    checks.append(check_gender_validity(str(fields.get("gender", "")), field_path="gender"))

    medical = fields.get("medicalInstitutionFields") or {}
    health_fund = ""
    if isinstance(medical, dict):
        health_fund = str(medical.get("healthFundMember", ""))
    checks.append(check_health_fund_member_validity(health_fund))

    checks.append(check_landline_phone_validity(str(fields.get("landlinePhone", "")), field_path="landlinePhone"))
    checks.append(check_mobile_phone_validity(str(fields.get("mobilePhone", "")), field_path="mobilePhone"))

    address = fields.get("address") or {}
    if isinstance(address, dict):
        checks.append(
            check_postal_code_validity(str(address.get("postalCode", "")), field_path="address.postalCode")
        )

    is_valid = all(c.valid for c in checks)
    for c in checks:
        if not c.valid:
            logger.warning("Validation failed: %s — %s", c.field_path, c.message)

    if is_valid:
        logger.info("Validation pipeline completed: all checks passed")

    return ValidationPipelineResult(checks=checks, is_valid=is_valid)


def validation_result_to_dict(result: ValidationPipelineResult) -> dict[str, Any]:
    """Serialize validation pipeline output for APIs / UI."""
    return result.to_dict()
