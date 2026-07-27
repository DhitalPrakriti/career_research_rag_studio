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

Also working: a FastAPI service and a React + TypeScript UI that surfaces the routing
decision, the retrieval grade, the trace and the retrieved chunks for every answer.

Not built yet: containerised deployment. There is no `Dockerfile` and nothing is deployed
to Cloud Run.

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

  llm.py                    ← provider selection: litellm, gemini, openai, ollama

  generation/
    generator.py            ← grounded prompt, citations, extractive fallback

  agents/                   ← decisions the agent makes about retrieval
    graph.py                ← CareerResearchAgent, the LangGraph (see below)
    router.py               ← rule-based route selection
    grader.py               ← relevance grading + reranking of retrieved context
    rewriter.py             ← query expansion for retry attempts
    trace.py                ← per-node trace events
    langsmith.py            ← LangSmith run configuration

  evaluation/
    golden_set.py           ← golden set loading, metrics, negative-control handling
    agent_eval.py           ← agent-level eval with route/grade/rewrite columns
    failure_analysis.py     ← weakest-example inspection

  api/
    app.py                  ← FastAPI app; ingests once at startup, serves the built UI
    models.py               ← request/response schemas

  cli.py, agent_cli.py, eval_cli.py, agent_eval_cli.py,
  ragas_eval_cli.py, failure_cli.py     ← entry points, kept at the root so the
                                          documented `python -m rag_studio.<name>`
                                          commands stay stable

  tests/                    ← one test module per implementation module
frontend/
  src/
    api.ts                  ← typed client mirroring api/models.py
    App.tsx                 ← page composition
    components/             ← StatRow, RouteCard, TraceTimeline, SourceList, Badge
    styles.css              ← light/dark theme from the validated palette
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

Set `LLM_PROVIDER` to one of `litellm`, `gemini`, `openai`, `ollama` or `extractive`:

| Provider | Environment | Notes |
| --- | --- | --- |
| `litellm` | `LITELLM_API_KEY`, `LITELLM_BASE_URL`, `LITELLM_MODEL` | Through your proxy, so its per-model budget caps and response cache stay in the path |
| `gemini` | `GEMINI_API_KEY`, `GEMINI_MODEL` | Direct API call — **no budget cap in front of it** |
| `openai` | `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` | Any OpenAI-compatible endpoint |
| `ollama` | `OLLAMA_MODEL`, `OLLAMA_BASE_URL` | Fully local, needs `ollama serve` |
| `extractive` | — | No LLM; returns source text |

Leaving `LLM_PROVIDER` unset auto-detects from whichever credentials are present,
preferring `litellm` so the capped path wins by default. That is convenient locally but
ambiguous in a deployment, so **always name the provider in production**. Naming a
provider whose credentials are missing is an error rather than a silent switch to another.

`GEMINI_MODEL` defaults to `gemini-3.6-flash`; `gemini-3.5-flash-lite` is cheaper for bulk
work like RAGAS judging. `LITELLM_MODEL` takes a proxy alias such as `gemini-flash`.

Two behaviours worth knowing:

- **A configured provider that fails raises instead of falling back.** Silent degradation
  is how a whole evaluation run can end up scoring extractive text dumps as if they were
  generated answers — which is exactly what happened to the earlier RAGAS numbers below.
- **The extractive fallback is not generation.** It is there so the retrieval pipeline runs
  with zero API keys, but never score those runs for faithfulness or answer relevancy. The
  API and UI both label it explicitly so it cannot be mistaken for a real answer.

## Web UI

FastAPI serves the API and, when the frontend has been built, the UI from the same origin.

```powershell
# once
pip install -e ".[dev,eval,agent,api]"
cd frontend; npm install; npm run build; cd ..

# run
python -m uvicorn --factory rag_studio.api.app:create_app --port 8000
```

Then open <http://127.0.0.1:8000>. For frontend work, `npm run dev` serves on port 5173
and proxies `/api` to port 8000, so hot reload works against the live backend.

The UI is built to make the agent legible rather than to look like a chat app: each answer
carries the route the agent chose and why, its own relevance grade, how many rewrite
retries it needed, the ordered trace of graph nodes it executed, and every retrieved chunk
behind a disclosure so a citation can be checked against the exact text the model saw. Ask
a negative-control question such as "What is my GPA?" to watch the rewrite-and-retry path
fire and the agent decline to answer.

Endpoints: `GET /api/health`, `POST /api/query`, and OpenAPI docs at `/docs`.

Documents are ingested once at startup from `docs/` (override with `RAG_DOCS_DIR`), because
ingestion builds an embedding index in seconds while a query takes milliseconds.

There is deliberately no module-level `app = create_app()`; that would run `load_dotenv()`
on import and leak real credentials into any process that imports the module, including the
test suite, which would then make live billed calls. Hence `--factory`.

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

The golden set has 14 examples: 11 answerable questions and 3 **negative controls** (GPA,
salary, work experience) whose answers are deliberately absent from the documents.

### Why the two classes are scored separately

A negative control exists to test one thing: does the system admit it does not know instead
of inventing an answer? So the metrics that apply to it are different, and averaging the
two classes together is misleading in both directions.

- **Retrieval metrics do not apply to a negative control.** There is nothing to retrieve.
  Worse, `term_recall` returns 1.0 for an empty `expected_terms` list, so each negative
  control is a free 1.0 that inflates the headline figure. The CLIs now report
  `answerable_term_recall` and `answerable_doc_title_hit` alongside the blended numbers.
- **RAGAS `answer_relevancy` actively punishes correct behaviour on them.** It classifies
  "that information is not in the sources" as *noncommittal* and floors the score to 0.00.
  Both `negative_gpa` and `negative_salary` score exactly 0.00 for refusing correctly.
- **What a negative control should be scored on is `refusal_accuracy`** — the share of them
  where the answer actually declined. That is now reported by the eval CLIs, and each record
  carries `is_negative_control` and `refusal_correct`.

Refusal detection is a deterministic phrase match rather than another LLM call, for the same
reason the retrieval grader is: the thing being measured should not depend on the thing
being tested. It recognises the agent's own low-confidence message too.

Negative controls turn out to come in two shapes, and the metric accepts both because what
it really measures is *did not fabricate*:

- **Unanswerable value questions** — "What is Prakriti's GPA?" The documents are silent, so
  the right answer is "not mentioned".
- **Absence-verification questions** — "Has Prakriti worked at Google or Microsoft?" The
  documents contain the full history, so a confident "there is no record of that" is the
  ideal answer, not a hedge.

Current results (n=3): **refusal accuracy 1.000** — every negative control was handled
without fabricating. Answerable term recall and doc-title hit are 1.000 (n=11), so the free
1.0s from the negative controls were not hiding anything, but the split is now explicit.

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

**Read the 0.910 answerable-only figure, not the 0.787 headline** — see the negative
controls section above for why. The RAGAS CLI now prints both side by side and appends a
note whenever negative controls are in the input, so the blended number cannot be quoted
without its caveat.

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
