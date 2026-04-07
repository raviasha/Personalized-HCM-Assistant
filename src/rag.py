"""
rag.py — Retrieval-Augmented Generation pipeline.

Retrieves relevant corpus chunks from ChromaDB and generates an answer
via GPT-4o, given a user message and the current conversation history.
"""

from __future__ import annotations

import re
from pathlib import Path

import openai
from dotenv import load_dotenv

from . import vector_store as vs
from .prompts import build_rag_messages

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHAT_MODEL = "gpt-4o"
_RETRIEVE_TOP_K = 4
_MAX_USER_MESSAGE_CHARS = 2000  # guard against excessively long inputs
_CORPUS_DIR = Path(__file__).parent.parent / "corpus"


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------

def _sanitise(text: str) -> str:
    """Basic input sanitisation to mitigate prompt-injection attempts.

    - Strips leading/trailing whitespace.
    - Truncates to _MAX_USER_MESSAGE_CHARS.
    - Removes null bytes.
    """
    text = text.replace("\x00", "").strip()
    return text[:_MAX_USER_MESSAGE_CHARS]


# ---------------------------------------------------------------------------
# Citation post-processing
# ---------------------------------------------------------------------------

def _inject_citations(text: str, doc_map: dict) -> str:
    """Replace [DOC-N] labels in *text* with markdown links to corpus files.

    The links use Gradio's /file= scheme so they resolve when the corpus
    directory is added to ``allowed_paths`` in ``launch()``.
    """
    for label, info in doc_map.items():
        title = info["title"]
        source = info["source"]
        url = f"/file={_CORPUS_DIR / (source + '.md')}"
        text = text.replace(f"[{label}]", f"[{title}]({url})")
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer(user_message: str, conversation_history: list[dict], employee_state: dict | None = None) -> str:
    """Generate a grounded answer to *user_message* using RAG.

    Args:
        user_message: The raw text typed by the employee.
        conversation_history: The Gradio chatbot history so far (list of
            ``{"role": ..., "content": ...}`` dicts), **not** including the
            current user message.
        employee_state: Optional employee state dict for personalisation.

    Returns:
        The assistant's reply as a plain string with citation links injected.
    """
    clean_message = _sanitise(user_message)
    if not clean_message:
        return "I didn't receive a message. Could you please rephrase your question?"

    # 1. Retrieve relevant corpus chunks
    chunks = vs.query(clean_message, n_results=_RETRIEVE_TOP_K)

    # 2. Build the message list for the completion request
    messages, doc_map = build_rag_messages(chunks, conversation_history, clean_message, employee_state)

    # 3. Call GPT-4o
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=700,
    )

    # 4. Replace [DOC-N] citation labels with markdown links
    reply = response.choices[0].message.content
    return _inject_citations(reply, doc_map)
