import os
from groq import Groq
from dotenv import load_dotenv

from config import GROQ_LLM_NAME

load_dotenv()

EVAL_DATA = [
    # Specific fact
    {
        "question": "When did google first create a coding agent that could evolve on its own and what was it called?",
        "ground_truth": "According to Wikipedia, in May 2025, Google DeepMind unveiled AlphaEvolve",
    },
    # Info not in context:
    {
        "question": "What is the best AWS service to easily use fully managed LLM agents?",
        "ground_truth": "I cannot answer this question based on the provided context.",
    },
    # Comparison across different chunks
    {
        "question": "What is the difference between AI regulation and AI alignment?",
        "ground_truth": "According to Wikipedia, regulation of artificial intelligence is the development of public sector policies and laws, while AI alignment aims to steer AI systems toward a person's or group's intended goals, preferences, or ethical principles.",
    },
    # Out of scope
    {
        "question": "Which country currently has the strictest AI regulation framework?",
        "ground_truth": "I cannot answer this question based on the provided context.",
    },
    # Fact trap
    {
        "question": "Who created the AlphaEvolve system and what company developed it?",
        "ground_truth": "According to Wikipedia, the provided context states that Google DeepMind unveiled AlphaEvolve in May 2025, but it does not specify the individual creators.",
    },
]

EVAL_QUESTIONS = [item["question"] for item in EVAL_DATA]

JUDGE_PROMPT = \
"""Score how faithful the generated answer is compared to the ground truth.

Score the answer between 0 and 1 based on:
- 1.0: Fully grounded in stated sources, correctly refuses when information is unavailable, no hallucinations.
- 0.5: Partially grounded but makes minor unsupported claims.
- 0.0: Hallucinates facts, ignores the context, or fails to acknowledge missing information.

Return ONLY a single number between 0 and 1.

Ground truth:
{ground_truth}

Generated answer:
{answer}
"""


def judge_answer(ground_truth: str, answer: str) -> float:
    """
    Use an LLM to score a RAG answer for groundedness and hallucination avoidance.

    Args:
        question: the original evaluation question
        answer: the generated answer from the pipeline

    Returns:
        Float score between 0 and 1
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

    client = Groq(api_key=api_key)
    prompt = JUDGE_PROMPT.format(ground_truth=ground_truth, answer=answer)
    response = client.chat.completions.create(
        model=GROQ_LLM_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return float(response.choices[0].message.content.strip())


def run_evaluation(results) -> None:
    scores = []

    for result in results:
        question = result["question"]
        ground_truth = result["ground_truth"]
        answer = result["answer"]
        score = judge_answer(ground_truth, answer)
        scores.append(score)

        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"Ground truth: {ground_truth}")
        print(f"Score: {score:.2f}")

    avg = sum(scores) / len(scores)
    print(f"\nAverage Faithfulness Score: {avg:.2f}")
