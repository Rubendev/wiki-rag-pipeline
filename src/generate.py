import os
from groq import Groq
from dotenv import load_dotenv

from config import GROQ_LLM_NAME

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided Wikipedia context.
Every answer must start with "According to Wikipedia, ...".
If the answer cannot be found in the context, say so clearly."""
# Always cite the source article(s) you used.

SYSTEM_PROMPT_CONSTRAINT = \
"""You are a strict fact-checking assistant. Your ONLY knowledge source is the context provided by the user.

Rules you MUST follow:
1. ONLY use information explicitly stated in the provided context.
2. If the context does not contain enough information to answer, respond exactly with: "I cannot answer this question based on the provided context."
3. Do NOT use any prior knowledge, assumptions, or information outside the context.
4. Do NOT speculate or infer beyond what is clearly written.
5. If you are uncertain whether the context supports a claim, do NOT include it.
6. Every answer must start with "According to Wikipedia, ..."."""
# 6. At the end of your answer, always list which sources you used under a "Sources:" section.


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Claude Sonnet 4.6 generated code to 
    build a constrained prompt from the query and retrieved context chunks.
    Each chunk is clearly delimited to prevent the model from blending sources.

    Args:
        query: the user's question
        retrieved_chunks: top-k chunks returned by retrieve()

    Returns:
        Formatted prompt string
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        context_blocks.append(
            f"[CONTEXT BLOCK {i+1}]\n"
            f"Source: {chunk['source']} (chunk {chunk['chunk_index']}, relevance score: {chunk['score']:.4f})\n"
            f"{chunk['text']}\n"
            f"[END BLOCK {i+1}]"
        )
    context = "\n\n".join(context_blocks)

    return (
        "Below are the ONLY context blocks you are allowed to use. Do not use any other information.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Important: Base your answer strictly on the context blocks above. "
        "If the answer is not in the context, say so."
    )


def generate_answer(query: str, retrieved_chunks: list[dict], constraint_prompt: bool = True) -> str:
    """
    Claude Sonnet 4.6 generated code to 
    pass retrieved context and query to lama via Groq and returns the answer.
    Uses a constrained prompt to reduce hallucinations.

    Args:
        query: the user's question
        retrieved_chunks: top-k chunks returned by retrieve()
        constraint_prompt: whether to use the constrained prompt

    Returns:
        Generated answer string
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

    client = Groq(api_key=api_key)
    prompt = build_prompt(query, retrieved_chunks) if constraint_prompt else query
    messages = [
            {"role": "system", "content": SYSTEM_PROMPT_CONSTRAINT if constraint_prompt else SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    response = client.chat.completions.create(
        model=GROQ_LLM_NAME,
        messages=messages,
        temperature=0.0,  # Zero temperature = fully deterministic, no creative hallucination
    )

    return response.choices[0].message.content
