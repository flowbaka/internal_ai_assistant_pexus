from pathlib import Path

import requests

from privacy import mask_sensitive_data


API_URL = "http://127.0.0.1:8000"
PDF_PATH = Path("sample_employee.pdf")


TEST_CASES = [
    {
        "question": (
            "How many annual leave days do "
            "employees receive?"
        ),
        "expected_groups": [
            ["20"],
            ["annual leave"],
        ],
    },
    {
        "question": (
            "What are the normal working hours?"
        ),
        "expected_groups": [
            ["9:00"],
            ["5:00"],
        ],
    },
    {
        "question": (
            "How often may employees work remotely?"
        ),
        "expected_groups": [
            ["two", "twice", "2"],
            ["week"],
        ],
    },
    {
        "question": (
            "What is the CEO's home address?"
        ),
        "expected_groups": [
            [
                "could not find",
                "not found",
                "not provided",
                "not present",
            ],
        ],
    },
]


def upload_test_document() -> str:
    """Upload the test PDF and return its document ID."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {PDF_PATH}"
        )

    with PDF_PATH.open("rb") as pdf_file:
        response = requests.post(
            f"{API_URL}/documents/extract",
            files={
                "file": (
                    PDF_PATH.name,
                    pdf_file,
                    "application/pdf",
                )
            },
            timeout=180,
        )

    response.raise_for_status()

    result = response.json()

    return result["document_id"]


def ask_question(
    question: str,
    document_id: str,
) -> dict:
    """Send one question to the document API."""

    response = requests.post(
        f"{API_URL}/documents/ask",
        json={
            "question": question,
            "document_id": document_id,
        },
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


def answer_matches_expected_groups(
    answer: str,
    expected_groups: list[list[str]],
) -> bool:
    """
    Each group must contain at least one match.
    """

    lowercase_answer = answer.lower()

    return all(
        any(
            option.lower() in lowercase_answer
            for option in group
        )
        for group in expected_groups
    )


def run_rag_evaluation() -> tuple[int, int]:
    """Run document question-answering tests."""

    print(
        "Uploading evaluation document..."
    )

    document_id = upload_test_document()

    print(
        f"Document ID: {document_id}"
    )

    print()

    passed_tests = 0

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        result = ask_question(
            question=test_case["question"],
            document_id=document_id,
        )

        answer = result["answer"]

        passed = answer_matches_expected_groups(
            answer=answer,
            expected_groups=(
                test_case["expected_groups"]
            ),
        )

        status = (
            "PASSED"
            if passed
            else "FAILED"
        )

        print(
            f"Test {test_number}: {status}"
        )

        print(
            f"Question: {test_case['question']}"
        )

        print(
            f"Answer: {answer}"
        )

        print(
            "Expected groups:",
            test_case["expected_groups"],
        )

        print("-" * 60)

        if passed:
            passed_tests += 1

    total_tests = len(TEST_CASES)

    accuracy = (
        passed_tests / total_tests
    ) * 100

    print()
    print("RAG EVALUATION RESULTS")

    print(
        f"Passed: {passed_tests}/{total_tests}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    return passed_tests, total_tests


def run_privacy_evaluation() -> bool:
    """
    Test whether sensitive information is masked.
    """

    sample_text = (
        "Contact John at john@example.com. "
        "His phone number is 202-555-0143 and "
        "his SSN is 123-45-6789."
    )

    masked_text, redaction_counts = (
        mask_sensitive_data(sample_text)
    )

    sensitive_values = [
        "john@example.com",
        "202-555-0143",
        "123-45-6789",
    ]

    sensitive_data_removed = all(
        value not in masked_text
        for value in sensitive_values
    )

    total_redactions = sum(
        redaction_counts.values()
    )

    redactions_detected = (
        total_redactions >= 3
    )

    passed = (
        sensitive_data_removed
        and redactions_detected
    )

    status = (
        "PASSED"
        if passed
        else "FAILED"
    )

    print()
    print("PRIVACY EVALUATION")
    print(f"Status: {status}")
    print(f"Original: {sample_text}")
    print(f"Masked: {masked_text}")

    print(
        f"Redactions: {redaction_counts}"
    )

    return passed


def print_final_summary(
    rag_passed: int,
    rag_total: int,
    privacy_passed: bool,
):
    """
    Print combined results.
    """

    total_tests = rag_total + 1
    total_passed = rag_passed

    if privacy_passed:
        total_passed += 1

    overall_accuracy = (
        total_passed / total_tests
    ) * 100

    print()
    print("=" * 60)
    print("FINAL EVALUATION SUMMARY")

    print(
        f"RAG tests: {rag_passed}/{rag_total}"
    )

    print(
        "Privacy test:",
        (
            "PASSED"
            if privacy_passed
            else "FAILED"
        ),
    )

    print(
        f"Overall: {total_passed}/{total_tests}"
    )

    print(
        f"Overall accuracy: "
        f"{overall_accuracy:.2f}%"
    )

    print("=" * 60)


if __name__ == "__main__":
    try:
        rag_passed, rag_total = (
            run_rag_evaluation()
        )

        privacy_passed = (
            run_privacy_evaluation()
        )

        print_final_summary(
            rag_passed=rag_passed,
            rag_total=rag_total,
            privacy_passed=privacy_passed,
        )

    except requests.exceptions.ConnectionError:
        print(
            "ERROR: Cannot connect to FastAPI."
        )

        print(
            "Start it with: "
            "uvicorn main:app --reload"
        )

    except requests.exceptions.Timeout:
        print(
            "ERROR: The API request timed out."
        )

    except requests.exceptions.HTTPError as error:
        print(
            f"API ERROR: {error}"
        )

        if error.response is not None:
            print(
                error.response.text
            )

    except Exception as error:
        print(
            f"UNEXPECTED ERROR: {error}"
        )