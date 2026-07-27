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

## Current state

Working end to end today: document ingestion, hybrid retrieval with reranking,
parent/child chunking, multi-query and HyDE, grounded generation with citations, a golden
test set with deterministic metrics and failure analysis, and the LangGraph agent that
routes, grades its own retrieval and retries with a rewritten query.

Not built yet: the HTTP API, the web UI, and containerised deployment — there is no
`backend/api`, no `frontend/` and no `Dockerfile`. Everything is driven through the CLIs
below.

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
| Backend / Frontend / Deployment | not yet built |

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate          # macOS/Linux: source venv/bin/activate

# 2. Install the package and extras
pip install -e ".[dev,eval,agent]"

# 3. Configure your keys
copy .env.example .env           # then fill in GEMINI_API_KEY

# 4. Run the tests
pytest backend/rag_studio/tests -q
```

### Choosing a generation backend

`AnswerGenerator` picks the first provider that is configured, in this order:

| Order | Provider | Environment |
| --- | --- | --- |
| 1 | Gemini (default) | `GEMINI_API_KEY`, optional `GEMINI_MODEL` |
| 2 | OpenAI-compatible | `OPENAI_API_KEY` + `OPENAI_MODEL`, optional `OPENAI_BASE_URL` |
| 3 | Ollama (local) | `OLLAMA_MODEL`, optional `OLLAMA_BASE_URL`/`OLLAMA_TIMEOUT`/`OLLAMA_NUM_*` |
| 4 | Extractive fallback | nothing configured |

`GEMINI_MODEL` defaults to `gemini-3.6-flash`; `gemini-3.5-flash-lite` is a cheaper
option for bulk work like RAGAS judging.

Two behaviours worth knowing:

- **A configured provider that fails raises instead of falling back.** Silent degradation
  is how a whole evaluation run can end up scoring extractive text dumps as if they were
  generated answers — which is exactly what happened to the earlier RAGAS numbers below.
- **The extractive fallback is not generation.** It is there so the retrieval pipeline runs
  with zero API keys, but never score those runs for faithfulness or answer relevancy.

Because the OpenAI path honours `OPENAI_BASE_URL`, it doubles as the way to route through
a LiteLLM proxy — set the base URL to the proxy and `OPENAI_MODEL` to a proxy alias. That
keeps the proxy's per-model budget caps and response caching, which calling Gemini
directly does not.

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
Context precision and recall below are pinned at 1.000 for the same reason: retrieval on
this set is not the bottleneck, generation quality is.

### RAGAS

Generate records **with a real generation backend configured**, then judge them. With
`GEMINI_API_KEY` set, both steps use Gemini:

```powershell
python -m rag_studio.eval_cli --output evaluation\runs\gemini_baseline.jsonl
python -m rag_studio.ragas_eval_cli --input evaluation\runs\gemini_baseline.jsonl `
  --output evaluation\runs\ragas_gemini.jsonl
```

Judging is embarrassingly parallel, so the Gemini judge defaults to 4 concurrent calls and
the full golden set is practical in one pass. Useful flags:

- `--judge-provider ollama` to judge locally instead (needs `ollama serve`; defaults to 1
  worker because local judging is the bottleneck)
- `--judge-model gemini-3.5-flash-lite` to cut judging cost
- `--metrics faithfulness` to run one metric at a time
- `--limit N` to judge only the first N examples

Supported metrics: `faithfulness`, `answer_relevancy`, `context_precision`,
`context_recall`.

Results with Gemini generating the answers and `gemini-3.6-flash` judging, over the full
golden set (n=14, 56 judge calls in about 70 seconds):

| Metric | Score |
| --- | --- |
| faithfulness | 0.979 |
| answer_relevancy | 0.787 (0.910 excluding negative controls — see below) |
| context_precision | 1.000 |
| context_recall | 1.000 |

Faithfulness is the metric with real signal: 11 of 14 examples score a perfect 1.00, and
the floor is 0.83 (`software_frontend`). Being LLM-judged, it moves a little between runs —
an earlier pass over the same records gave 0.952.

**Answer relevancy penalises correct refusals.** The two lowest scores are `negative_gpa`
and `negative_salary` at exactly 0.00 — RAGAS classifies "that information is not in the
sources" as *noncommittal* and floors the score. Those are negative controls, so refusing
is precisely the behaviour we want, and the metric marks it as failure. Read the 0.910
positives-only figure alongside the 0.787 headline, and treat per-example answer relevancy
on negative controls as uninformative.

Three separate bugs had to be fixed before any of these numbers existed, all of which
produced silent nulls rather than errors:

1. `ragas` 0.4.3 imports `langchain_community.chat_models.vertexai`, removed in
   langchain-community 0.4, with no upper bound in its own dependencies. Any clean install
   made `import ragas` fail outright. Pinned to `<0.4`.
2. The local embeddings wrapper exposed `model` as a `SentenceTransformer` object. RAGAS
   reads that attribute for telemetry and validates it as `Optional[str]`, so every
   embedding-based metric — that is, answer relevancy — died inside a pydantic
   `ValidationError` and returned null. Now a string, with a regression test.
3. Gemini flash models reject `n > 1`, which RAGAS uses to reverse-generate several
   questions per answer. Default `--answer-relevancy-strictness` is now 1.

The earlier numbers in this file were also produced against *extractive* answers, since no
LLM was configured at the time — never a fair target for faithfulness or answer relevancy.

## Key learning outcomes

- End-to-end RAG pipeline design
- Chunking strategy tradeoffs (fixed vs recursive vs parent/child)
- Dense vs sparse retrieval, and fusion of both
- Cross-encoder reranking
- Deterministic evaluation, negative controls, and metric saturation
- Agentic query routing, retrieval grading, and self-correcting retry loops
- Production deployment with Docker + Cloud Run (pending)
