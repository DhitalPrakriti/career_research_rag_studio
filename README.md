# Career + Research RAG Studio

A production-grade RAG (Retrieval-Augmented Generation) system for querying your personal
career documents — resumes, job descriptions, class notes, and research papers.
Built by Prakriti Dhital as a portfolio project covering the full RAG lifecycle:
ingestion → retrieval → evaluation → agentic routing → deployment.

## What it does

Point it at your resumes, job descriptions, cover letters, class notes, and research
papers, then ask questions like:

- "How should I tailor my resume for this job description?"
- "Compare these two job descriptions — which one fits my skills better?"
- "What skills from my resume are missing for this role?"
- "What binary F1 score was achieved in the capstone project?"

You get grounded answers with citations back to the source document and page. Retrieval
quality is measured against a golden test set, and the agent exposes a full trace of every
decision it made along the way.

## Build status

| Week | Focus | Status |
| --- | --- | --- |
| 1 | Ingestion pipeline (loader, chunker, embeddings, FAISS) | ✅ Done |
| 2 | Answer generation + citations | ✅ Done |
| 3 | Hybrid retrieval + reranking, parent/child, multi-query, HyDE | ✅ Done |
| 4 | Golden test set + deterministic eval + failure analysis | ✅ Done |
| 5 | Agentic routing with LangGraph, grading, rewrite, tracing | ✅ Done |
| 6 | FastAPI + React UI + Docker + Cloud Run | ⬜ Not started |

Week 6 has not been started: there is no `backend/api`, no `frontend/`, and no
`Dockerfile` yet. Everything today is driven through the CLIs below.

## Architecture

The package is organised by pipeline stage. Each stage is a subpackage that re-exports
its public names, so `from rag_studio.agents import CareerResearchAgent` works as well as
the full module path.

```
docs/                       ← source documents (PDFs) live here
evaluation/
  golden_set.jsonl          ← 14 golden examples, incl. 3 negative controls
  runs/                     ← eval outputs (gitignored)
backend/rag_studio/
  schema.py                 ← core types: Document, Chunk, RetrievedChunk, Citation, RagAnswer
  config.py                 ← RagConfig, read from environment
  pipeline.py               ← RagPipeline: composition root wiring the stages together

  ingestion/                ← documents in, indexed chunks out
    loader.py               ← PDF/txt/md loading + metadata (title, page, doc_type)
    chunker.py              ← recursive word chunking
    parent_child.py         ← small child chunks for precision, parents for context
    embeddings.py           ← sentence-transformers embeddings
    vector_store.py         ← FAISS dense index

  retrieval/                ← ways of finding relevant chunks
    bm25.py                 ← sparse retrieval
    hybrid.py               ← dense + sparse reciprocal rank fusion
    reranker.py             ← cross-encoder reranking
    multi_query.py          ← query variants + fusion across them
    hyde.py                 ← hypothetical document embeddings

  generation/
    generator.py            ← grounded prompt, citations, Ollama/OpenAI/extractive fallback

  agents/                   ← decisions the agent makes about retrieval
    graph.py                ← CareerResearchAgent, the LangGraph (see below)
    router.py               ← rule-based route selection
    grader.py               ← relevance grading + reranking of retrieved context
    rewriter.py             ← query expansion for retry attempts
    trace.py                ← per-node trace events
    langsmith.py            ← LangSmith run configuration

  evaluation/
    golden_set.py           ← golden set loading + deterministic metrics
    agent_eval.py           ← agent-level eval with route/grade/rewrite columns
    failure_analysis.py     ← weakest-example inspection

  cli.py, agent_cli.py, eval_cli.py, agent_eval_cli.py,
  ragas_eval_cli.py, failure_cli.py     ← entry points, kept at the root so the
                                          documented `python -m rag_studio.<name>`
                                          commands stay stable

  tests/                    ← one test module per implementation module
```

Routing, grading and rewriting live under `agents/` rather than `retrieval/` because they
are decisions the agent makes *about* retrieval, not retrieval strategies themselves.

### The agent graph

`CareerResearchAgent` in `agents/graph.py` compiles this LangGraph:

```
route_query ──┬─→ direct_answer ─────────────────────────────→ END
              │
              └─→ retrieve_answer ─→ grade_retrieval ──┬─→ generate ─→ END
                        ↑                              │
                        └───────── rewrite_query ←─────┘
                                  (bounded by max_retries)
```

- **route_query** picks the retriever and flags (parent context, multi-query, HyDE) from
  the question shape, and may rewrite the query *before* the first retrieval for vague
  memory/database questions.
- **grade_retrieval** scores question↔context token overlap. Below the threshold the agent
  rewrites the query and retries, up to `max_retries` (default 1), then answers anyway
  with an explicit low-confidence caveat rather than silently guessing.
- Every node appends an `AgentTraceEvent`, surfaced via `--show-trace` and used as
  evaluation columns.

## Tech stack

| Layer | Technology |
| --- | --- |
| Document parsing | pypdf |
| Chunking | custom recursive + parent/child |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | FAISS |
| Sparse retrieval | BM25 |
| Reranker | cross-encoder (ms-marco-MiniLM-L-6-v2) |
| LLM | Ollama (local) / OpenAI, with extractive fallback |
| Orchestration | LangGraph |
| Evaluation | deterministic metrics + RAGAS |
| Observability | local trace + LangSmith |
| Backend / Frontend / Deployment | not yet built (Week 6) |

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate          # macOS/Linux: source venv/bin/activate

# 2. Install the package and extras
pip install -e ".[dev,eval,agent]"

# 3. Run the tests
pytest backend/rag_studio/tests -q
```

Generation falls back to extractive answers when no LLM is configured, so the pipeline
runs end-to-end with zero API keys. To use a local LLM, set `OLLAMA_MODEL` (and
optionally `OLLAMA_NUM_GPU`, `OLLAMA_NUM_THREAD`, `OLLAMA_NUM_PREDICT`). To use OpenAI,
set `OPENAI_MODEL` and `OPENAI_API_KEY`.

Retrieval and chunking are configured from the environment via `RagConfig.from_env()` —
`EMBEDDING_MODEL`, `RERANKER_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`,
`PARENT_CHUNK_SIZE`/`CHILD_CHUNK_SIZE` (and their overlaps), `MAX_CONTEXT_CHARS`, `TOP_K`.

## Usage

Baseline (non-agentic) pipeline:

```powershell
python -m rag_studio.cli "What AI skills does my resume show?" --docs docs\Prakriti_Dhital_Resume_AI_ML.pdf
```

Agentic pipeline, showing the route it chose and the full decision trace:

```powershell
python -m rag_studio.agent_cli "What binary F1 score was achieved in the capstone project?" `
  --docs docs\Prakriti_Dhital_Resume_AI_ML.pdf --show-route --show-trace
```

Add `--langsmith` (requires `LANGSMITH_API_KEY`) to ship the run to LangSmith;
`--langsmith-project` overrides the default project `career-research-rag-studio`.

## Evaluation

The golden set has 14 examples, including 3 **negative controls** (GPA, salary, work
experience) whose answers are deliberately absent from the documents.

Deterministic retrieval metrics — no LLM judge, fully reproducible:

```powershell
python -m rag_studio.eval_cli --output evaluation\runs\baseline.jsonl
python -m rag_studio.agent_eval_cli --output evaluation\runs\agent_latest.jsonl
python -m rag_studio.failure_cli --input evaluation\runs\agent_latest.jsonl
```

Latest results (n=14):

| Run | Term recall | Doc title hit | Mean retrieval grade | Rewrite rate |
| --- | --- | --- | --- | --- |
| Baseline pipeline (`all_resumes_baseline`) | 1.000 | 1.000 | — | — |
| LangGraph agent (`agent_latest`) | 1.000 | 1.000 | 0.617 | 0.071 |

Term recall and document-title hit are saturated at 1.000 on this golden set, so they no
longer discriminate between configurations — the mean retrieval grade (range 0.250–0.857)
is the metric with headroom. Exactly one example (`tutoring_memory`) triggered a rewrite
and retry. Both facts are the main argument for a harder golden set before tuning further.

### RAGAS with a local Ollama judge

First generate RAGAS-compatible records from the golden set, then run **one metric at a
time** — local judging is slow, so start with `--limit 1`:

```powershell
$env:OLLAMA_MODEL="llama3"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python -m rag_studio.eval_cli --output evaluation\runs\all_resumes_baseline.jsonl
python -m rag_studio.ragas_eval_cli --input evaluation\runs\all_resumes_baseline.jsonl `
  --output evaluation\runs\ragas_ollama_context_recall_limit1.jsonl `
  --limit 1 --judge-model llama3 --num-thread 4 --metrics context_recall
```

Supported metrics: `faithfulness`, `answer_relevancy`, `context_precision`,
`context_recall`.

**RAGAS results so far are incomplete.** Only `context_recall` has produced a value:

| Metric | Value | n |
| --- | --- | --- |
| context_recall | 1.00 | 1 |
| faithfulness | not obtained (null) | 1 |
| answer_relevancy | not obtained (null) | 1 |
| context_precision | not obtained (null) | 1 |

The three null metrics returned `None` from the local llama3 judge rather than a score,
and the graded runs were made against *extractive* answers (no LLM was configured at the
time), which is not a fair target for faithfulness or answer relevancy. Getting real
numbers here requires re-running with an LLM-backed generator and a judge that reliably
returns parseable output — that is the open Week 4 thread.

## Key learning outcomes

- End-to-end RAG pipeline design
- Chunking strategy tradeoffs (fixed vs recursive vs parent/child)
- Dense vs sparse retrieval, and fusion of both
- Cross-encoder reranking
- Deterministic evaluation, negative controls, and metric saturation
- Agentic query routing, retrieval grading, and self-correcting retry loops
- Production deployment with Docker + Cloud Run (Week 6, pending)
