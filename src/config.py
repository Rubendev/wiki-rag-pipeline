import os

RECURSIVE_CHUCK_SIZE = 1000
RECURSIVE_CHUNK_OVERLAP = 200

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")

FASTEMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_LLM_NAME = "llama-3.3-70b-versatile"

WIKI_ARTICLES = [
    # "Attention Is All You Need",
    # "Large language model",
    # "Generative artificial intelligence",
    "Recursive self-improvement",
    "Artificial general intelligence",
    "AI alignment",
    "Regulation of artificial intelligence",

]
