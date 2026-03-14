import os
import wikipedia
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import WIKI_ARTICLES, RECURSIVE_CHUCK_SIZE, RECURSIVE_CHUNK_OVERLAP


def download_wikipedia_articles(titles=WIKI_ARTICLES, cache=True, store_dir="data/wiki/", use_cache=True):
    """
    Downloads Wikipedia articles by title, with optional caching to avoid redundant downloads.
    Args:
        titles: list of Wikipedia article titles to download
        cache: whether to save downloaded articles to disk for future reuse
        store_dir: directory to save cached articles
        use_cache: whether to check for and load from cache before downloading
    Returns:
        dict mapping article title -> full text
    """
    articles = {}
    for title in titles:
        if use_cache:
            filename = os.path.join(store_dir, f"{title.replace(' ', '_')}.txt")
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    articles[title] = f.read()
                    print(f"Loaded '{title}' from cache.")
                    continue
        try:
            content = wikipedia.page(title).content
            articles[title] = content
            if cache:
                os.makedirs(store_dir, exist_ok=True)
                filename = os.path.join(store_dir, f"{title.replace(' ', '_')}.txt")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
        except Exception as e:
            print(f"Failed to download '{title}': {e}")
    return articles


def chunk_articles(articles: dict, chunk_size=RECURSIVE_CHUCK_SIZE, chunk_overlap=RECURSIVE_CHUNK_OVERLAP) -> list[dict]:
    """
    Claude Sonnet 4.6 generated code to 
    chunk multiple articles while preserving source metadata
    using RecursiveCharacterTextSplitter
    and use Wikipedia-friendly separators.
    Common chunk size of 512–1000 characters with an overlap of 100–200 characters

    Args:
        articles: dict mapping article title -> full text
        chunk_size: max characters per chunk
        chunk_overlap: overlap between consecutive chunks to preserve context

    Returns:
        List of dicts with 'text', 'source', and 'chunk_index' keys
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # Wikipedia-friendly separators
    )

    all_chunks = []
    for title, text in articles.items():
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": title,
                "chunk_index": i,
            })
        print(f"'{title}' split into {len(chunks)} chunks.")

    return all_chunks


if __name__ == "__main__":
    print('Running ingest')
    titles = ["Retrieval-augmented generation", "Recursive self-improvement"]
    articles = download_wikipedia_articles(titles, cache=True)
    chunks = chunk_articles(articles)
    print(f"Total chunks across all articles: {len(chunks)}")
    print(chunks[0])
