import requests
import streamlit as st


# Address of our FastAPI backend
API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Internal Document Assistant",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Privacy-Aware Document Assistant")

st.write(
    "Upload a PDF and ask questions based only on that document."
)


# Store the active document information across Streamlit reruns
if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "filename" not in st.session_state:
    st.session_state.filename = None


# ----------------------------
# PDF UPLOAD SECTION
# ----------------------------

st.header("1. Upload a document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
)


if st.button(
    "Upload and process document",
    disabled=uploaded_file is None,
):
    try:
        with st.spinner("Extracting and indexing the document..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            }

            response = requests.post(
                f"{API_URL}/documents/extract",
                files=files,
                timeout=180,
            )

        if response.status_code == 200:
            result = response.json()

            # Remember which document is currently active
            st.session_state.document_id = result["document_id"]
            st.session_state.filename = result["filename"]

            st.success("Document uploaded successfully!")

            st.write("Filename:", result["filename"])
            st.write("Pages:", result["page_count"])
            st.write("Characters:", result["character_count"])
            st.write("Chunks created:", result["chunk_count"])
            st.write("Chunks stored:", result["stored_chunks"])

        else:
            try:
                error_message = response.json().get(
                    "detail",
                    "Document upload failed.",
                )
            except ValueError:
                error_message = response.text

            st.error(error_message)

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. Make sure Uvicorn is running."
        )

    except requests.exceptions.Timeout:
        st.error(
            "The document took too long to process. Please try again."
        )

    except Exception as error:
        st.error(f"Unexpected error: {error}")


# Show the currently active document
if st.session_state.document_id:
    st.info(
        f"Active document: {st.session_state.filename}"
    )


# ----------------------------
# QUESTION-ANSWERING SECTION
# ----------------------------

st.header("2. Ask a question")

question = st.text_area(
    "What would you like to know?",
    placeholder="For example: How many annual leave days are provided?",
    height=100,
)


ask_button = st.button("Ask the document")


if ask_button:
    if not st.session_state.document_id:
        st.warning("Please upload and process a PDF first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching the document..."):
                response = requests.post(
                    f"{API_URL}/documents/ask",
                    json={
                        "question": question,
                        "document_id": st.session_state.document_id,
                    },
                    timeout=180,
                )

            if response.status_code == 200:
                result = response.json()

                st.subheader("Answer")
                st.write(result["answer"])

                # Display privacy-masking information if returned
                redactions = result.get(
                    "redaction_counts",
                    result.get("redactions", {}),
                )

                if redactions and any(redactions.values()):
                    st.subheader("Privacy protection")
                    st.write(
                        "Sensitive information was masked before "
                        "sending context to the AI provider."
                    )
                    st.json(redactions)

                # Display the retrieved source chunks
                sources = result.get("sources", [])

                if sources:
                    st.subheader("Sources")

                    for index, source in enumerate(sources, start=1):
                        source_name = source.get(
                            "filename",
                            st.session_state.filename,
                        )

                        with st.expander(
                            f"Source {index}: {source_name}"
                        ):
                            if "text" in source:
                                st.write(source["text"])

                            if "chunk_index" in source:
                                st.caption(
                                    f"Chunk: {source['chunk_index']}"
                                )

                            if "distance" in source:
                                st.caption(
                                    f"Vector distance: "
                                    f"{source['distance']:.4f}"
                                )

            else:
                try:
                    error_message = response.json().get(
                        "detail",
                        "The question could not be answered.",
                    )
                except ValueError:
                    error_message = response.text

                st.error(error_message)

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. Make sure Uvicorn is running."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The AI response took too long. Please try again."
            )

        except Exception as error:
            st.error(f"Unexpected error: {error}")