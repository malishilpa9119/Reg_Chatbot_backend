from __future__ import annotations

import os
import threading
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200
_DEFAULT_K = 4
_DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

_VECTOR_STORE_CACHE: dict[str, FAISS] = {}
_CACHE_LOCK = threading.Lock()

_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
_LLM: Optional[ChatGroq] = None

_PROMPT = PromptTemplate(
    template=(
        "You are a helpful assistant.\n"
        "Answer ONLY from the provided transcript context.\n"
        "If the context is insufficient, just say you don't know.\n\n"
        "{context}\n"
        "Question: {question}\n"
    ),
    input_variables=["context", "question"],
)


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        model_name = os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL)
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=model_name)
    return _EMBEDDINGS


def _get_llm() -> ChatGroq:
    global _LLM
    if _LLM is None:
        model_name = os.getenv("GROQ_MODEL", _DEFAULT_GROQ_MODEL)
        _LLM = ChatGroq(model=model_name)
    return _LLM


def _fetch_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en"]
        )

        transcript_text = " ".join(chunk.text for chunk in transcript_list)
        return transcript_text

    except TranscriptsDisabled:
        raise ValueError("Transcript not available for this video")

    except Exception as e:
        raise ValueError(str(e))



def _get_vector_store(video_id: str, transcript_text: str) -> FAISS:
    with _CACHE_LOCK:
        cached = _VECTOR_STORE_CACHE.get(video_id)
        if cached is not None:
            return cached

    chunk_size = int(os.getenv("CHUNK_SIZE", _DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", _DEFAULT_CHUNK_OVERLAP))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = splitter.create_documents([transcript_text])
    vector_store = FAISS.from_documents(documents, _get_embeddings())

    with _CACHE_LOCK:
        _VECTOR_STORE_CACHE[video_id] = vector_store

    return vector_store


def answer_question(video_id: str, question: str) -> str:
    if not video_id or not question:
        return "Please provide both a video_id and a question."

    transcript_text = _fetch_transcript(video_id)
    if transcript_text is None:
        return "Transcript not available for this video."

    vector_store = _get_vector_store(video_id, transcript_text)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": _DEFAULT_K},
    )

    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs:
        return "I don't know."

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = _PROMPT.format(context=context_text, question=question)
    response = _get_llm().invoke(final_prompt)

    return response.content.strip()
