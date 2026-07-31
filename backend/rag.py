"""
rag.py
------
Retrieval-Augmented Generation layer for the counseling chatbot.

Knowledge source: plain-text files under knowledge_base/ (counseling
guidelines, psycho-education material, career-advice snippets). Each
file is chunked and embedded, then stored in a local Chroma collection
(chroma_store/) so no external vector DB or extra API key is needed.

Embeddings come from Ollama's embedding endpoint (nomic-embed-text) —
same local server that's already running Llama 3.2 for llm.py, so this
stays free and fully local.

One-time setup:
    pip install chromadb
    ollama pull nomic-embed-text
    python rag.py            # builds the index from knowledge_base/*.txt

Usage elsewhere:
    import rag
    chunks = rag.retrieve("dealing with exam anxiety", k=3)
"""
import glob
import os
from typing import List

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "counseling_kb"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 80

# chromadb and requests are imported lazily inside the functions below
# (see _get_client, _embed) rather than at module level. This means
# `import rag` -- and therefore `import chatbot_engine`, and therefore
# the whole app -- still works even before `pip install chromadb` has
# been run. Only the actual RAG calls (build_index/retrieve) need it.
_client = None


def _get_client():
    global _client
    if _client is None:
        import chromadb
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def _embed(text: str) -> List[float]:
    import requests
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _chunk_text(text: str) -> List[str]:
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def build_index() -> int:
    """
    Reads every .txt file under knowledge_base/ - including files in
    category subfolders (stress_psychology/, career_employment/,
    university_info/, matching the architecture diagram's three KBs) -
    chunks each, embeds each chunk, and (re)builds the Chroma
    collection from scratch. Files directly in knowledge_base/ (not in
    a subfolder) get category "general".

    Run this once at setup, and again any time knowledge_base/
    content changes.

    Returns the number of chunks indexed.
    """
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    files = glob.glob(os.path.join(KB_DIR, "**", "*.txt"), recursive=True)
    if not files:
        raise FileNotFoundError(
            f"No .txt files found under {KB_DIR}. Add counseling-guideline / "
            "career-advice / university-info documents there before building "
            "the index - see knowledge_base/<category>/ subfolders."
        )

    ids, embeddings, documents, metadatas = [], [], [], []
    for filepath in files:
        source = os.path.basename(filepath)
        rel_dir = os.path.relpath(os.path.dirname(filepath), KB_DIR)
        category = "general" if rel_dir == "." else rel_dir.replace(os.sep, "/")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        for i, chunk in enumerate(_chunk_text(text)):
            ids.append(f"{category}-{source}-{i}")
            embeddings.append(_embed(chunk))
            documents.append(chunk)
            metadatas.append({"source": source, "category": category})

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(ids)


def retrieve(query: str, k: int = 3, category: str | None = None) -> List[str]:
    """
    Returns the top-k most relevant knowledge-base chunks for a query.

    If `category` is given (e.g. "career_employment", "stress_psychology",
    "university_info" - matching the knowledge_base/ subfolder names),
    only chunks from that category are considered. Leave it None to
    search across all categories, which is still the default used
    where chatbot_engine.py doesn't have a strong signal either way.

    Never raises - if the index hasn't been built yet, or the KB is
    empty, returns []. Callers (prompt_builder.build_counseling_prompt)
    already treat RAG context as optional, so a missing index degrades
    gracefully instead of breaking the chatbot.
    """
    try:
        collection = _get_client().get_collection(COLLECTION_NAME)
    except Exception:
        return []

    try:
        query_embedding = _embed(query)
        where = {"category": category} if category else None
        results = collection.query(
            query_embeddings=[query_embedding], n_results=k, where=where
        )
    except Exception:
        return []

    docs = results.get("documents", [[]])[0]
    return docs


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks from {KB_DIR}")