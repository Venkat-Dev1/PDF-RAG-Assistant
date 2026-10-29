
# rag.py

import hashlib
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# Load .env from this project's directory
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

DB_DIR = str(PROJECT_DIR / "chroma_db")


def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


def get_llm():
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY is missing. Check your project's .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        max_output_tokens=1024,
        timeout=60,
        max_retries=2,
    )


def get_vectorstore(collection_name):
    return Chroma(
        collection_name=collection_name,
        persist_directory=DB_DIR,
        embedding_function=get_embeddings(),
    )


def process_pdf(pdf_bytes):
    """Index an uploaded PDF and return its collection information."""

    # Identify the PDF by its contents.
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    collection_name = "pdf_" + pdf_hash[:24]

    # Save uploaded bytes temporarily so PyPDFLoader can read them.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        documents = PyPDFLoader(temp_path).load()

        if not documents:
            raise ValueError("No pages could be read from this PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "No readable text was found. "
                "This may be a scanned PDF without selectable text."
            )

        vectorstore = get_vectorstore(collection_name)
        existing_count = vectorstore._collection.count()

        # Reuse a complete index; rebuild an incomplete one.
        if existing_count != len(chunks):
            if existing_count > 0:
                vectorstore.delete_collection()
                vectorstore = get_vectorstore(collection_name)

            batch_size = 50

            for start in range(0, len(chunks), batch_size):
                vectorstore.add_documents(
                    chunks[start:start + batch_size]
                )

        return {
            "collection_name": collection_name,
            "pages": len(documents),
            "chunks": len(chunks),
            "pdf_hash": pdf_hash,
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def ask_question(question, collection_name):
    """Retrieve relevant chunks and return only Gemini's text answer."""

    vectorstore = get_vectorstore(collection_name)

    if vectorstore._collection.count() == 0:
        raise ValueError(
            "This PDF has no indexed content. Please upload and process it again."
        )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )

    documents = retriever.invoke(question)

    if not documents:
        return "I don't know based on the uploaded PDF."

    context = "\n\n".join(
        document.page_content for document in documents
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You answer questions using only the supplied PDF context.
If the context does not contain the answer, respond:
"I don't know based on the uploaded PDF."
Give a direct, clear answer. Do not output metadata, signatures,
internal notes, or information unrelated to the question."""
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ])

    response = get_llm().invoke(
        prompt.format_messages(
            context=context,
            question=question,
        )
    )

    # Return answer text, not the complete response object.
    answer = response.text

    if not answer:
        raise ValueError("Gemini returned no text answer.")

    return answer.strip()
