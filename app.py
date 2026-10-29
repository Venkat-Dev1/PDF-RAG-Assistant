import streamlit as st
from rag import process_pdf, ask_question


st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("📚 PDF RAG Assistant")
st.caption("Upload a PDF and ask questions about its contents.")


# Initialize session state
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar: PDF upload
with st.sidebar:
    st.header("Upload your PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="pdf_uploader",
    )

    if st.button(
        "Process PDF",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    ):
        try:
            with st.spinner(
                "Reading PDF and creating embeddings..."
            ):
                result = process_pdf(uploaded_file.getvalue())

            st.session_state.collection_name = result[
                "collection_name"
            ]
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []

            st.success("PDF processed successfully!")
            st.write(f"Pages: {result['pages']}")
            st.write(f"Text chunks: {result['chunks']}")

        except Exception as exc:
            st.error(f"Could not process PDF: {exc}")

    if st.session_state.pdf_name:
        st.divider()
        st.write("**Active document**")
        st.write(st.session_state.pdf_name)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept questions only after a PDF has been processed
question = st.chat_input(
    "Ask a question about your PDF...",
    disabled=not bool(st.session_state.collection_name),
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding the answer..."):
                answer = ask_question(
                    question,
                    st.session_state.collection_name,
                )

            # Display only the answer
            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception as exc:
            st.error(f"Could not generate an answer: {exc}")

