import re


SENSITIVE_PATTERNS = [
    (
        "email",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL REDACTED]",
    ),
    (
        "ssn",
        r"\b\d{3}-\d{2}-\d{4}\b",
        "[SSN REDACTED]",
    ),
    (
        "credit_card",
        r"\b(?:\d[ -]?){13,19}\b",
        "[CARD NUMBER REDACTED]",
    ),
    (
        "phone",
        r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)",
        "[PHONE REDACTED]",
    ),
]


def mask_sensitive_data(
    text: str,
) -> tuple[str, dict[str, int]]:
    masked_text = text
    redaction_counts = {}

    for category, pattern, replacement in SENSITIVE_PATTERNS:
        masked_text, count = re.subn(
            pattern,
            replacement,
            masked_text,
        )

        redaction_counts[category] = count

    return masked_text, redaction_counts