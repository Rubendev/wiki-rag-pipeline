import faiss
import json
import os
import numpy as np
from fastembed import TextEmbedding

from config import FASTEMBED_MODEL_NAME, INDEX_DIR, INDEX_PATH, CHUNKS_PATH


def build_index(chunks: list[dict]) -> tuple[faiss.Index, TextEmbedding]:
    """
    Claude Sonnet 4.6 generated code to 
    encode chunks of texts into dense embeddings and build a FAISS index
    using cosine similarity after L2 norm
    Persists the index and chunks to disk for reuse.

    Args:
        chunks: list of dicts with at least a 'text' key (output of chunk_articles)

    Returns:
        (faiss_index, model) tuple for later retrieval
    """
    model = TextEmbedding(FASTEMBED_MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]
    print(f"Encoding {len(texts)} chunks with '{FASTEMBED_MODEL_NAME}'...")
    embeddings = np.array(list(model.embed(texts)), dtype=np.float32)

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine similarity after L2 norm
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors (dim={dimension}).")

    # Persist to disk
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Index saved to '{INDEX_PATH}'.")
    print(f"Chunks saved to '{CHUNKS_PATH}'.")

    return index, model


def load_index() -> tuple[faiss.Index, list[dict], TextEmbedding]:
    """
    Claude Sonnet 4.6 generated code to 
    load previously built FAISS index and chunks from disk,
    and initialize the embedding model.

    Returns:
        (faiss_index, chunks, model) tuple
    
    Raises:
        FileNotFoundError: if index or chunks file does not exist
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"No saved index found at '{INDEX_DIR}'. Run build_index() first."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    model = TextEmbedding(FASTEMBED_MODEL_NAME)
    print(f"Loaded index ({index.ntotal} vectors) and {len(chunks)} chunks from disk.")
    return index, chunks, model


def retrieve(
    query: str,
    chunks: list[dict],
    index: faiss.Index,
    model: TextEmbedding,
    top_k: int = 5,
) -> list[dict]:
    """
    Claude Sonnet 4.6 generated code to 
    retrieve the top-k most relevant chunks for a given query
    using cosine similarity after L2 norm
    and includes relevance scores in output.

    Args:
        query: natural language question
        chunks: original list of chunk dicts (same order as used in build_index)
        index: FAISS index built from those chunks
        model: TextEmbedding model used to encode the chunks
        top_k: number of results to return

    Returns:
        List of top-k chunk dicts with an added 'score' key
    
    Example:
        print(f"\nTop results for: '{query}'\n")
        for i, result in enumerate(results):
            print(f"[{i+1}] score={result['score']:.4f} | source='{result['source']}' | chunk={result['chunk_index']}")
            print(f"    {result['text'][:200]}...")
            print()
    """
    query_embedding = np.array(list(model.embed([query])), dtype=np.float32)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results
