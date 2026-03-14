import sys

from ingest import download_wikipedia_articles, chunk_articles
from retrieval import build_index, load_index, retrieve
from generate import generate_answer
from evaluate import run_evaluation, EVAL_QUESTIONS, EVAL_DATA


def setup_rag():
    # Ingest
    articles = download_wikipedia_articles(cache=True)
    chunks = chunk_articles(articles)
    # Build dense retrieval index
    build_index(chunks)


def run_genai(questions):
    # Dense retrieval loading from disk
    index, chunks, model = load_index()
    results = []
    for question in questions:
        # Retrieve from index
        retrieved_chunks = retrieve(question, chunks, index, model, top_k=5)
        # Generate answer with LLM
        answer = generate_answer(question, retrieved_chunks, constraint_prompt=True)
        results.append({
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
        })
    return results


if __name__ == "__main__":
    # Download data, download models, build index
    setup_rag()
    # Manual evaluation (Faithfulness)
    if "-evaluate" in sys.argv or "--evaluate" in sys.argv:
        import pdb; pdb.set_trace()
        results = run_genai(EVAL_QUESTIONS)
        for i, result in enumerate(results):
            result['ground_truth'] = EVAL_DATA[i]['ground_truth']
        run_evaluation(results)
