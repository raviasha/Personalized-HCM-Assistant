"""
vector_store.py — ChromaDB initialization, chunking, embedding, and query.

Loads corpus markdown files, splits them into sections, embeds them using
OpenAI text-embedding-3-small, stores them in an ephemeral in-memory ChromaDB
collection, and caches the raw embeddings to disk to avoid re-calling the
OpenAI API on each restart.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import chromadb
import openai
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).parent.parent
CORPUS_DIR = _BASE_DIR / "corpus"
EMBEDDINGS_CACHE = _BASE_DIR / "embeddings" / "corpus_embeddings.pkl"
COLLECTION_NAME = "benefits_corpus"
EMBEDDING_MODEL = "text-embedding-3-small"
# Maximum characters that ChromaDB / the embedding API will handle per chunk
_MAX_CHUNK_CHARS = 6000

# Module-level singletons
_collection: chromadb.Collection | None = None
_source_title_map: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_markdown(text: str, source: str) -> list[dict]:
    """Split a markdown document into chunks at H1/H2/H3 section boundaries.

    Each chunk contains the section header plus its body text so that the
    chunk is self-contained when retrieved.
    """
    header_re = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)
    parts = header_re.split(text)

    chunks: list[dict] = []
    current_header = source.replace("_", " ").title()
    current_body = ""

    for part in parts:
        if header_re.fullmatch(part.strip()):
            if current_body.strip():
                chunk_text = f"{current_header}\n\n{current_body.strip()}"
                # Hard-cap in case a section is extremely long
                if len(chunk_text) > _MAX_CHUNK_CHARS:
                    chunk_text = chunk_text[:_MAX_CHUNK_CHARS]
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": source,
                        "header": current_header,
                    }
                )
            current_header = part.strip()
            current_body = ""
        else:
            current_body += part

    # Flush last section
    if current_body.strip():
        chunk_text = f"{current_header}\n\n{current_body.strip()}"
        if len(chunk_text) > _MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:_MAX_CHUNK_CHARS]
        chunks.append(
            {
                "text": chunk_text,
                "source": source,
                "header": current_header,
            }
        )

    return chunks


def _load_corpus() -> list[dict]:
    """Read all markdown files from the corpus directory and return chunks."""
    chunks: list[dict] = []
    for md_file in sorted(CORPUS_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        chunks.extend(_chunk_markdown(text, md_file.stem))
    return chunks


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Call the OpenAI Embeddings API and return a list of embedding vectors."""
    client = openai.OpenAI()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # API guarantees the response order matches input order
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def _load_or_build_embeddings(chunks: list[dict]) -> list[list[float]]:
    """Return embeddings for the given chunks, using disk cache when possible."""
    texts = [c["text"] for c in chunks]

    if EMBEDDINGS_CACHE.exists():
        try:
            with open(EMBEDDINGS_CACHE, "rb") as f:
                cache = pickle.load(f)
            if cache.get("texts") == texts:
                print(f"[VectorStore] Loaded {len(texts)} embeddings from cache.")
                return cache["embeddings"]
            print("[VectorStore] Corpus has changed — rebuilding embeddings.")
        except Exception:
            print("[VectorStore] Cache unreadable — rebuilding embeddings.")

    print(f"[VectorStore] Requesting embeddings for {len(texts)} chunks via OpenAI…")
    embeddings = _embed_texts(texts)

    EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump({"texts": texts, "embeddings": embeddings}, f)
    print("[VectorStore] Embeddings cached to disk.")

    return embeddings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_vector_store() -> None:
    """Build the in-memory ChromaDB collection from the corpus.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _collection
    if _collection is not None:
        return  # Already initialised

    chunks = _load_corpus()
    if not chunks:
        raise RuntimeError(f"No markdown files found in {CORPUS_DIR}")

    embeddings = _load_or_build_embeddings(chunks)

    client = chromadb.Client()  # ephemeral, in-process
    _collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    _collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "header": c["header"]} for c in chunks],
    )
    print(f"[VectorStore] Initialised with {len(chunks)} chunks.")


def query(query_text: str, n_results: int = 4) -> list[dict]:
    """Embed *query_text* and return the *n_results* most similar corpus chunks.

    Each returned dict has keys: ``text``, ``source``, ``distance``.
    """
    if _collection is None:
        raise RuntimeError("Vector store not initialised. Call init_vector_store() first.")

    client = openai.OpenAI()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query_text])
    query_embedding = response.data[0].embedding

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "text": doc,
            "source": meta["source"],
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def get_source_title_map() -> dict[str, str]:
    """Return a cached mapping of corpus file stems to their H1 document titles.

    E.g. ``{"health_insurance": "Health Insurance Benefits", ...}``
    """
    global _source_title_map
    if _source_title_map is not None:
        return _source_title_map
    result: dict[str, str] = {}
    for md_file in sorted(CORPUS_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if match:
            title = re.sub(r"^Acme Corp\s+", "", match.group(1).strip())
        else:
            title = md_file.stem.replace("_", " ").title()
        result[md_file.stem] = title
    _source_title_map = result
    return result


def get_corpus_topics() -> list[dict]:
    """Return one entry per corpus file with its display label and a starter question.

    Reads the H1 heading from each markdown file to produce a human-friendly
    label, so the list stays accurate as the corpus evolves.

    Returns:
        List of dicts with keys ``"label"`` (short display name) and
        ``"question"`` (a sensible starter question for that topic).
    """
    topics: list[dict] = []
    for md_file in sorted(CORPUS_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        # Extract the first H1 heading as the topic title
        match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if match:
            full_title = match.group(1).strip()
            # Strip the "Acme Corp " prefix if present for a shorter label
            label = re.sub(r"^Acme Corp\s+", "", full_title)
        else:
            label = md_file.stem.replace("_", " ").title()

        topics.append({
            "label": label,
            "question": f"Give me an overview of {label}.",
        })
    return topics
