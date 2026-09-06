import re


EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d[ -]*?){13,16}\b"
)

PHONE_PATTERN = re.compile(
    r"(?<!\w)"
    r"(?:\+?\d{1,3}[\s.-]?)?"
    r"(?:\(?\d{3}\)?[\s.-]?)"
    r"\d{3}[\s.-]?\d{4}"
    r"(?!\w)"
)


def replace_pattern(
    text: str,
    pattern: re.Pattern,
    replacement: str,
) -> tuple[str, int]:
    """Replace occurrences of one sensitive pattern."""

    matches = pattern.findall(text)

    masked_text = pattern.sub(
        replacement,
        text,
    )

    return masked_text, len(matches)


def mask_sensitive_data(
    text: str,
) -> tuple[str, dict]:
    """
    Mask sensitive information before sending
    text to an external AI service.
    """

    redaction_counts = {
        "emails": 0,
        "phone_numbers": 0,
        "ssn_numbers": 0,
        "credit_cards": 0,
    }

    masked_text, count = replace_pattern(
        text,
        EMAIL_PATTERN,
        "[EMAIL]",
    )

    redaction_counts["emails"] += count

    masked_text, count = replace_pattern(
        masked_text,
        SSN_PATTERN,
        "[SSN]",
    )

    redaction_counts["ssn_numbers"] += count

    masked_text, count = replace_pattern(
        masked_text,
        CREDIT_CARD_PATTERN,
        "[CREDIT_CARD]",
    )

    redaction_counts["credit_cards"] += count

    masked_text, count = replace_pattern(
        masked_text,
        PHONE_PATTERN,
        "[PHONE]",
    )

    redaction_counts["phone_numbers"] += count

    return masked_text, redaction_counts