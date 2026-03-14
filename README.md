# Wiki RAG Pipeline

RAG-Powered Q&A System over a focused Wikipedia subset

---

## 1) Project Overview

This project implements a production-relevant GenAI pipeline:

1. Download Wikipedia articles
2. Chunk text with `RecursiveCharacterTextSplitter`
3. Build dense embeddings with `fastembed`
4. Store/search vectors using `FAISS`
5. Pass retrieved context to cloud based LLM
6. Generate constrained answers

---

## 2) Architecture

```text
Wikipedia articles
      │
      ▼
 [ingest.py]     download + chunk
      │
      ▼
 [retrieval.py]  dense embeddings + FAISS index
      │
      ▼
 [generate.py]   constraint prompt + LLM API call
       │
      ▼
 [evaluate.py]   evaluate faithfulness with LLM API call
      │
      ▼
 [pipeline.py]   orchestrates ingest → index → retrieve → generate → evaluate
```

---

## 3) Design Choices

### Dataset choice
**Option B: Wikipedia subset covering AGI**  
Wikipedia has a standard API for data retrieval and doesn’t contain complex content (like code extracts) requiring more advanced RAG and format chunking.

### Chunking approach
**Recursive chunking** (`RecursiveCharacterTextSplitter`)  
A lightweight, robust all-rounder and safe starting point. It preserves paragraph-level context for Wikipedia-style text to reduce hallucinations and improve attention.

### Retrieval approach
**Dense retrieval**  
Prioritizing semantic understanding since Wikipedia is linguistically diverse and has concept based content. 

- Model: `sentence-transformers/all-MiniLM-L6-v2` (via `fastembed`)
- Vector DB: FAISS `IndexFlatIP` with L2 normalization (cosine similarity)

### Generation
**Cloud-based LLM:**  `llama-3.3-70b-versatile` via Groq API  

### Frameworks:
**None**  
For a lightweight, flexible approach. Langchain/other frameworks are overkill for Simple RAG (not data heavy, no chains or complex agents required).

---

## 4) Hallucination Mitigation

### Constraint Prompt
Very easy to implement for moderate to high reliability.
The effectiveness of constraints varies heavily depending on the model and the system prompt must be explicit. Over-constraint and conflicting instructions reduce creativity, but creativity is not important for Wikipedia factual information.

---

## 5) Evaluation

Evaluated both manually and using the same Cloud-based LLM used for generation to score answer faithfulness.

### Faithfulness

| Question ID | Question (short) | Manual Score (0-1) | Auto Score (0-1) | Notes |
|---|---|---:|---:|---|
| Q1 | Self-evolving coding agent | 1 | 1.00 | Able to find specific facts |
| Q2 | Best AWS managed agent service | 1 | 1.00 | Refuses to answer out of context |
| Q3 | AI regulation vs AI alignment | 1 | 0.5 | Good comparison between chunks |
| Q4 | Strictest AI regulation | 1 | 1 | No answer out of context |
| Q5 | Creators of AlphaEvolve | 1 | 1 | Answers in context and refuses out of context |
| **Average (manual)** |  | **1** | **-** | Baseline with constraint prompt |

### Interpretation
The constraint promt is very strict and refuses any out of context information.
The system showcases strong faithfulness and strong in-context relevance, but
very weak out of context relevance. Relevance could be improved with more balanced prompting and fallback "general answer" strategies.

---

## 6) Setup

### Environment (`pyproject.toml`)
Create a Python environment and install requirements via `pyproject.toml` with pip, poetry or uv

### API key (`.env`)
Add Groq API key to `.env` file

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 7) Run

Run pipeline for inital setup:
```bash
python src/pipeline.py
```
Optionally, you can run with evaluation enabled:
```bash
python src/pipeline.py --evaluate
```

To start interactive Q&A CLI session: 
```bash
python src/app.py
```

---

### [Example queries with outputs](sample_queries.md)

---

## 8) Future improvements on limitations

Addressing current limitations and pipeline improvements
- Very strict constraint prompt - doesn't allow knowledge outside of context
- No filtering is being done on vector similarity scores to ensure retrieval quality before generation
- More challenging evaluation queries should be studied
- Very basic ingestion and data processing (no provision for Wiki formulas, title separation, references)
- The system only ingests a small subset of Wikipedia articles, if relevant information is not present in the indexed documents, retrieval will fail and the model may either refuse the query.

For production readiness
- Add grounding checks/citations
- Add retry/timeout policies for LLM calls
- Take LLM API limits / token usage into consideration
- Add retrieval metrics
- Add chunking diagnostics
- Upgrade logging
- Add exception handling
- Add unit testing for future consistency
- Add Dockerfile and update code based on deployment architecture
