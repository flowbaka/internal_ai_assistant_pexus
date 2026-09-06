import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Internal Document Assistant",
    page_icon="📄",
)

st.title("Privacy-Aware Document Assistant")

st.write(
    "Upload an internal PDF and ask questions about its contents."
)


st.header("1. Upload a document")

uploaded_file = st.file_uploader(
    "Select a PDF",
    type=["pdf"],
)


if st.button("Process document"):
    if uploaded_file is None:
        st.warning("Please select a PDF first.")

    else:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        }

        try:
            with st.spinner("Processing document..."):
                response = requests.post(
                    f"{API_BASE_URL}/documents/extract",
                    files=files,
                    timeout=180,
                )

            if response.status_code == 200:
                result = response.json()

                st.success("Document processed successfully.")

                st.write({
                    "filename": result["filename"],
                    "pages": result["page_count"],
                    "chunks": result["stored_chunks"],
                    "document_id": result["document_id"],
                })

            else:
                error = response.json()
                st.error(
                    error.get(
                        "detail",
                        "Document processing failed.",
                    )
                )

        except requests.RequestException:
            st.error(
                "Could not connect to the FastAPI backend."
            )


st.header("2. Ask a question")

question = st.text_input(
    "Question about the uploaded documents"
)


if st.button("Ask document"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching documents..."):
                response = requests.post(
                    f"{API_BASE_URL}/documents/ask",
                    json={
                        "question": question,
                    },
                    timeout=180,
                )

            if response.status_code == 200:
                result = response.json()

                st.subheader("Answer")
                st.write(result["answer"])

                privacy = result.get("privacy", {})

                st.caption(
                    "Sensitive values masked: "
                    f"{privacy.get('total_redactions', 0)}"
                )

                with st.expander("View sources"):
                    for source in result["sources"]:
                        st.write(
                            f"File: {source['filename']} | "
                            f"Chunk: {source['chunk_number']} | "
                            f"Distance: {source['distance']}"
                        )

                        st.write(source["text"])
                        st.divider()

            else:
                error = response.json()
                st.error(
                    error.get(
                        "detail",
                        "Question answering failed.",
                    )
                )

        except requests.RequestException:
            st.error(
                "Could not connect to the FastAPI backend."
            )