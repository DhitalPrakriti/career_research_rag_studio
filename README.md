# Career + Research RAG Studio

A production-grade RAG (Retrieval-Augmented Generation) system for querying your personal
career documents — resumes, job descriptions, class notes, and research papers.
Built by Prakriti Dhital as a portfolio project covering the full RAG lifecycle:
ingestion → retrieval → evaluation → agentic routing → deployment.

Live app: https://career-rag-studio-1079009244955.us-central1.run.app/ 

## What it does

**Paste a job description and get your resume rewritten against it.** The tool pulls the
requirements out of the posting, retrieves evidence for each one from your own resumes, and
rewrites that evidence into bullets aimed at the job.

The part that matters: **it never invents experience.** A requirement with no supporting
evidence is reported as a gap and is never written about — structurally, not by asking a
model to behave. Paste a posting asking for Kubernetes and eight years of team leadership
and it will tell you those are gaps rather than quietly claiming them.

It also answers questions over the same documents, with citations back to the source page —
"What binary F1 score was achieved in the capstone project?" — which is how the retrieval
stack is evaluated.

Every result shows its work: which requirements matched, the exact resume text each bullet
came from, and the ordered trace of what the system did.

## Current state

Working end to end today: job-description tailoring with gap analysis, document ingestion,
hybrid retrieval with reranking, parent/child chunking, multi-query and HyDE, grounded
generation with citations, a golden test set with deterministic metrics and failure
analysis, and the LangGraph agent that routes, grades its own retrieval and retries with a
rewritten query.

Also working: a FastAPI service and a React + TypeScript UI that surfaces the routing
decision, the retrieval grade, the trace and the retrieved chunks for every answer.

Deployed to Google Cloud Run from a single container that serves both the API and the UI.

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
    router.py               ← LLM route classification, with rule fallback
    grader.py               ← relevance grading + reranking of retrieved context
    rewriter.py             ← query expansion for retry attempts
    trace.py                ← per-node trace events
    langsmith.py            ← LangSmith run configuration

  tailoring/
    matching.py             ← requirement extraction, rarity-weighted scoring, LLM audit
    service.py              ← orchestration; only supported requirements reach the rewriter

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

### Tailoring to a job description

```
job description ─→ extract requirements ─→ retrieve evidence per requirement
                                                      ↓
        tailored bullets ←─ rewrite ←─ hide gaps ←─ verify with an LLM
                                                      ↓
                                              gap list (never written about)
```

1. **Extract** the requirements from the posting. An LLM does this; bullet-list parsing is
   the fallback so it works with no API key.
2. **Retrieve** evidence for each requirement separately, with precise child-chunk settings.
   A requirement is a narrow claim, so wide retrieval would blur which chunk supports it.
3. **Score** the evidence by term overlap **weighted by rarity across your resumes**. Flat
   overlap is dangerous here: "Experience using Kubernetes for infrastructure automation"
   scored 0.67 — "matched" — against a resume with no Kubernetes, because the generic words
   carried it while the one decisive term was absent and counted for nothing. Under rarity
   weighting an unseen specialist term gets the maximum weight and can never be matched.
4. **Verify** every requirement against its evidence in one batched LLM call. Weighted
   overlap still cannot judge "8 years leading a platform engineering team" — there is no
   rare term to key on, only a claim about seniority — so the final status comes from
   reading. Gaps are audited too, not just matches: a one-way ratchet could strip a false
   match but never restore a false gap, which is how "Proficiency in Python" came back as a
   gap against a resume covered in Python.
5. **Rewrite** only the supported requirements. The rewriter is never given a gap's text, so
   there is no path by which one can become a bullet.

Thresholds are deliberately strict (0.6 matched, 0.35 partial) because the errors are not
symmetric: a false gap costs you a bullet you could have had, a false match puts an
unsupported claim in front of an employer.

`POST /api/tailor` returns the per-requirement statuses, the bullets with the source ids
they were written from, the gap list, a recommendation of which of your resumes to start
from, and the trace.

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

- **route_query** classifies the question into one of four categories and maps that to
  retrieval settings. See below.
- **grade_retrieval** scores question↔context token overlap. Below the threshold the agent
  rewrites the query and retries, up to `max_retries` (default 1), then answers anyway
  with an explicit low-confidence caveat rather than silently guessing.
- Every node appends an `AgentTraceEvent`, surfaced via `--show-trace` and used as
  evaluation columns.

### Routing

One question, four categories, and a single mapping from category to retrieval settings so
the two classifiers cannot disagree about what a category means:

| Category | Retrieval |
| --- | --- |
| `direct_answer` | none — greeting or a question about the assistant |
| `exact_fact` | child chunks, no expansion: precision over recall |
| `broad_comparison` | parent context + multi-query + HyDE: maximum recall |
| `balanced` | parent context + multi-query |

`ROUTER_MODE` selects the classifier: `llm`, `rules`, or `auto` (LLM when a provider is
configured, else rules). A failed LLM call logs a warning and falls back to the rules — a
wrong route only costs some retrieval quality, so it is not worth failing a query over,
unlike a generation failure. Every decision carries `decided_by`, so the trace and the UI
always show which classifier actually ran.

**The rules key on question form, never on corpus vocabulary.** They previously matched
literals lifted straight from the golden set — `"binary f1"`, `"macro f1"`, `"score did"`,
`"what database"` — which is tuning the router on the test set. It scored well on those 14
sentences and generalised to nothing: "What binary F1 score was achieved?" took the
precision route while "How accurate was the model?" fell through to the default. A test now
greps the module for those literals so the shortcut cannot come back.

The same cleanup fixed a worse bug. `direct` means *skip retrieval entirely*, and the old
rule matched the bare word "help" anywhere in a question, so "Which resume would help me
for an AI engineering role?" and "Can you help me tailor my resume?" — core use cases —
were answered with boilerplate and no retrieval at all. Greeting matching is now anchored
to the whole question.

Both classifiers were checked against that failure mode, because the LLM reproduced it
independently: with a loose prompt it read "Can you help me tailor my resume?" as a question
about the assistant. The prompt now states that politeness phrasing does not make a request
meta, with examples.

Replacing the overfitted rules changed no measured outcome — answerable term recall 1.000,
doc-title hit 1.000, refusal accuracy 1.000, mean grade 0.617 across the overfitted rules,
the general rules and the LLM classifier alike — and the rewrite rate fell from 0.071 to
0.000, so the special case it existed for was not needed. The LLM classifier does make
better-reasoned choices that the current golden set cannot reward: it treats "What is
Prakriti's GPA?" as a single-value question and takes the precision route, where the rules
fall back to balanced. That the numbers cannot tell these three apart is itself the
argument for a harder golden set.

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

Endpoints: `GET /api/health`, `POST /api/tailor`, `POST /api/query`, the document
endpoints below, and OpenAPI docs at `/docs`.

The UI has three modes: **Tailor to a job** (paste a posting), **Ask a question**, and
**My documents**.

### Keeping your documents current

Ingestion runs at startup because building the index takes seconds while a query takes
milliseconds. That is right for serving and wrong for a tool whose premise is that you keep
editing your resume, so documents can be managed at runtime instead of needing a restart:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/documents` | What is on disk, with the number of indexed chunks per file |
| `POST /api/documents` | Upload one or more files, then reindex automatically |
| `POST /api/documents/reindex` | Rebuild the index from whatever is on disk |
| `DELETE /api/documents/{name}` | Remove a file, then reindex |

Uploading a file with an existing name **replaces** it, because replacing a resume with its
newer version is the main thing this is for — accumulating "resume (1).pdf" variants that
all get indexed would quietly corrupt the evidence. The listing shows chunks per document,
so a file sitting at 0 chunks is visibly on disk but not yet indexed.

Reindexing swaps the agent and the index in one step behind a lock, so a concurrent request
never sees a half-built index.

### Login

Set `APP_PASSWORD` and every data endpoint requires a session:

| Endpoint | Auth |
| --- | --- |
| `GET /api/live` | open — liveness probe, reveals nothing. Not `/healthz`: Cloud Run reserves that path and answers it itself |
| `GET /api/auth/session` | open — tells the UI whether to show the login screen |
| `POST /api/auth/login` / `logout` | open |
| everything else, including `/api/health` | session required |

Leaving `APP_PASSWORD` unset disables auth, which keeps local development frictionless and
is exactly the wrong thing to deploy — startup logs a warning when it happens.

The session is a signed, `httpOnly`, `SameSite=Lax` cookie. It is a cookie rather than a
bearer token in `localStorage` because `localStorage` is readable by any injected script and
an `httpOnly` cookie is not. `APP_SECRET_KEY` signs it; rotating that value invalidates every
existing session. Prefer `APP_PASSWORD_HASH` (pbkdf2-sha256, salted) over the plaintext
variable in a deployment:

```powershell
python -c "from rag_studio.api.auth import hash_password; print(hash_password('your-password'))"
```

Failed logins are throttled per client — five attempts, then a five-minute lockout — so the
password cannot be guessed online. Password comparison is constant-time.

**Document writes still need thought even behind login.** Uploads are restricted to `.pdf`, `.txt` and
`.md`, capped at 10 MB, and filenames are reduced to a bare validated name so a path like
`../../.env` cannot escape the documents directory — but anyone who can reach the API can
still add or delete documents. Set `ALLOW_DOCUMENT_WRITES=false` for any deployment that is
publicly reachable, or put auth in front of it. The health and documents responses both
report `writes_enabled` so the UI hides the controls when it is off.

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

The golden set has 33 examples: 25 answerable questions and 8 **negative controls** whose
answers are deliberately absent from the documents. 24 of them search all three resumes
rather than the one known to hold the answer.

The answerable questions come in shapes chosen to break different parts of retrieval:

- **Paraphrase-only** — no keyword overlap with the source, so BM25 cannot shortcut it and
  the dense side has to carry it. "How does the tutoring system remember what a student
  said earlier in the conversation?" never says *database* or *Firestore*. One is a
  deliberate pair with `tutoring_memory`, which asks the same fact with the keywords in.
- **Multi-hop** — two facts, so one lucky chunk is not enough.
- **Distractor-sensitive** — a near-duplicate section in another resume offers a plausible
  wrong answer. The tutoring system is Cloud Run + Docker on one resume and Vertex AI Agent
  Engine on another; only one of those answers "which cloud service hosts it".
- **Aggregation** — counting or naming across a list.

### Why the two classes are scored separately

A negative control exists to test one thing: does the system admit it does not know instead
of inventing an answer? So the metrics that apply to it are different, and averaging the
two classes together is misleading in both directions.

- **Retrieval metrics do not apply to a negative control.** There is nothing to retrieve.
  Worse, `term_recall` returns 1.0 for an empty `expected_terms` list, so each negative
  control is a free 1.0 that inflates the headline figure. The CLIs now report
  `answerable_term_recall`, `answerable_doc_title_hit` and `answerable_doc_precision`
  alongside the blended numbers.
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

#### Negative controls with a plausible wrong answer in reach

GPA and salary are easy, and it took a while to see why: nothing in the documents resembles
an answer, so refusing needs no more than noticing the topic is absent. The five added
controls each sit next to an answer the documents *do* contain.

The sharpest is `negative_cnn_bilstm_f1`. CNN-BiLSTM is named as one of eight architectures,
but only two models have reported scores — 94.28% Binary F1 for Late Fusion and 85.98% Macro
F1 for Transformer. A number is right there to be misattributed. The answer:

> The provided sources mention that the CNN-BiLSTM model was one of eight deep learning
> models evaluated [1, 2, 3]. However, the sources do not provide the specific F1 score
> achieved by the CNN-BiLSTM model; they only report the F1 scores for the Late Fusion model
> (94.28% Binary F1) and the Transformer model (85.98% Macro F1) [1, 2, 3].

That is the ideal answer — it declines *and* attributes both real scores to the right models.
`negative_aws_services` behaves the same way, quoting "AWS & Azure familiarity" and declining
to name services that are not there.

**And it was scored as a fabrication.** The marker list had `not provided` but not
`do not provide`, so refusal accuracy read 0.875 while the system was behaving perfectly.
Enumerating literals per tense was never going to hold, so one pattern now covers the
negated-verb family — do/does/did, contracted or not, against fifteen verbs — which closes
every tense at once and stays deterministic. Of the negated verbs that appear in real
refusals across every run recorded so far (`contain`, `mention`, `state`, `specify`,
`include`, `provide`, `list`), the last two were uncovered.

The tradeoff is unchanged and worth stating: a phrase match cannot tell a refusal from a
fabrication that happens to contain a hedge. It stays a phrase match anyway, because the
alternative is asking an LLM to grade the LLM.

**Current results: refusal accuracy 1.000 (n=8), on all five retrieval configurations
below** — including all five hard controls, and unchanged when the retrieved context is cut
to a fifth of its size.

### Why the retrieval metrics were saturated, which was not the golden set's fault

Term recall and document-title hit both sat at 1.000 and could not tell two configurations
apart. The obvious diagnosis was that the questions were too easy, so the set was rebuilt
around the harder shapes above — and both metrics stayed at exactly 1.000. That turned out to
be the more useful result.

**The whole corpus is 14,174 characters.** Three one-page resumes. Parent-context expansion
returns whole pages, so `top_k=3` hands back a mean of 8,668 characters — 61% of every
character available, up to 84% on individual questions — and 9 of the 25 answerable examples
retrieve all three resumes. A recall metric cannot fail when the context is most of the
corpus, and document-title hit cannot miss a document that is nearly always included. The
saturation was a property of the retrieval budget, not the questions.

Two metrics were added because of this, and both are reported by the CLIs:

- **`doc_precision`** — the share of retrieved chunks drawn from a document expected to hold
  the answer. This is the cost that recall metrics hide: returning all three resumes for a
  question one of them answers scores 0.333 and pushes the work of ignoring two irrelevant
  resumes onto the generator.
- **`context_chars`** — a diagnostic, not a score, so a future run whose context has quietly
  grown to swallow the corpus is visible instead of looking like a perfect result.

Deterministic retrieval metrics — no LLM judge, fully reproducible:

```powershell
python -m rag_studio.eval_cli --output evaluation\runs\hard_baseline.jsonl
python -m rag_studio.agent_eval_cli --output evaluation\runs\agent_hard.jsonl
python -m rag_studio.failure_cli --input evaluation\runs\agent_hard.jsonl
```

Latest results (n=33: 25 answerable, 8 negative controls). All five rows scored refusal
accuracy 1.000:

| Run | Term recall | Doc title hit | Doc precision | Context chars |
| --- | --- | --- | --- | --- |
| Baseline, parent context, `top_k=3` (default) | 1.000 | 1.000 | 0.707 | 8,668 |
| Baseline, child chunks, `top_k=3` | 1.000 | 1.000 | 0.707 | 5,470 |
| Baseline, parent context, `top_k=1` | 0.903 | 0.880 | 0.880 | 3,505 |
| Baseline, child chunks, `top_k=1` | 0.843 | 0.960 | 0.960 | 1,750 |
| LangGraph agent, `top_k=3` | 1.000 | 1.000 | 0.720 | 6,546 |

The agent adds a mean retrieval grade of 0.547 and a rewrite rate of 0.030 — one example in
33 triggered a rewrite and retry. The grade fell from 0.617 on the old 14-example set, which
is the harder questions showing up in the one metric that already had headroom.

**At `top_k=1` the metrics finally discriminate**, and the reading is a real tradeoff rather
than a ranking. Expanding a child chunk to its parent page recovers terms the child did not
contain (term recall 0.843 → 0.903) and costs document precision (0.960 → 0.880), because a
whole page pulls in neighbouring sections that belong to other questions. That is the
parent/child design working as intended, and it is only visible once the retrieval budget is
tight enough for a wrong choice to cost something.

At `top_k=3` — the default, and what the deployed app uses — the honest summary is that
retrieval on a corpus this small is over-provisioned. Turning parent context off cuts the
context by 37% with no measurable loss on any metric. Nothing here is a bottleneck worth
tuning: `doc_precision` at 0.707 is the only number with room, and the way to move it is a
corpus large enough that returning most of it is no longer an option.

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

Results with Gemini generating the answers and `gemini-3.6-flash` judging, **measured on the
previous 14-example golden set** (56 judge calls in about 70 seconds). These have not been
re-run against the 33-example set — `context_precision` and `context_recall` are pinned at
1.000 here for the same over-provisioned-retrieval reason described above, so re-judging them
would cost calls to confirm a ceiling that is already understood:

| Metric | Score |
| --- | --- |
| faithfulness | 0.979 |
| answer_relevancy | 0.787 (0.910 excluding negative controls — see below) |
| context_precision | 1.000 |
| context_recall | 1.000 |

Faithfulness is the metric with real signal: 11 of those 14 examples score a perfect 1.00,
and the floor is 0.83 (`software_frontend`). Being LLM-judged, it moves a little between
runs — an earlier pass over the same records gave 0.952.

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
- Deterministic evaluation, negative controls, and metric saturation — including the case
  where a saturated metric is measuring the retrieval budget rather than the questions, and
  where a metric is under-counting correct behaviour instead of the system regressing
- Agentic query routing, retrieval grading, and self-correcting retry loops
- Production deployment with Docker + Cloud Run

### Evaluating the tailoring, not just the retrieval

The unit tests cover the mechanics; they say nothing about whether the output is right on a
real posting. `evaluation/tailoring_set.jsonl` holds 6 postings — AI engineer, ML engineer,
backend, frontend, IT support, data platform — labelled with 33 skills the resumes do
contain and 18 they do not, deliberately spanning all three resume files.

```powershell
python -m rag_studio.tailor_eval_cli
```

| Metric | Latest run |
| --- | --- |
| Invented claims (present in no retrieved evidence) | 0.000 — 26 of 26 bullets grounded |
| Misattributed (real, but another requirement's evidence) | 0.077 — 2 of 26 |
| Classification accuracy (51 labels) | 1.000 |
| Extraction recall | 1.000 |

**Invented and misattributed are counted separately, and the distinction changed the
headline.** The rewriter receives every supported requirement and its evidence in one
prompt, so a bullet can cite a real fact belonging to a neighbouring requirement. Checking
each bullet only against its own requirement's evidence reported 4.5% "fabrication"; almost
all of it was untidy attribution rather than invention. Only a claim absent from *all*
retrieved evidence is a fabrication.

**These numbers move between runs.** The verification step is an LLM call, so identical
inputs do not always produce identical statuses — classification accuracy read 0.941 on one
run and 1.000 on the next two, with `PyTorch` and `Flask` marked absent once and matched
afterwards. Quote the metric as a range, not a fixed figure, and re-run before trusting a
change.

**One genuine invention has been observed.** In one run a bullet expanded "BFRB" from the
resume into "Body-Focused Repetitive Behaviors" — factually correct, but text appearing
nowhere in the evidence, produced from the model's own knowledge. It did not recur, so the
honest claim is that invention is rare rather than impossible, which is exactly why the
check exists rather than resting on the structural argument alone.

The fabrication check needs no hand labelling: it extracts every number and every
named-entity-shaped token from a bullet and asserts each appears in the evidence. Its first
version only caught digits, acronyms and inner capitals like `PyTorch`, which silently
skipped the case that matters — `Kubernetes` and `Terraform` are plainly capitalised, so an
invented technology would have passed while a legitimate one was checked. Mid-sentence
capitalisation is now the signal, with sentence-initial words skipped so the verb a bullet
opens with does not fire.

Label correctness was verified against the extracted PDF text before the set shipped, which
caught `Tailwind` being labelled absent when it is on the software developer resume, and
ruled out short substring-prone keywords such as `rag` (inside "storage") and `git` (inside
"digit").

## Deployment

One container, built by the two-stage `Dockerfile`: node builds the frontend, then a Python
image serves those assets alongside the API, so there is no cross-origin request in
production and nothing to host separately.

```powershell
gcloud run deploy career-rag-studio --source . `
  --region=us-central1 --service-account=rag-studio-run@PROJECT.iam.gserviceaccount.com `
  --execution-environment=gen2 `
  --add-volume="name=docsvol,type=cloud-storage,bucket=PROJECT-documents" `
  --add-volume-mount="volume=docsvol,mount-path=/mnt/docs" `
  --set-env-vars="RAG_DOCS_DIR=/mnt/docs,LLM_PROVIDER=gemini,GEMINI_MODEL=gemini-3.6-flash" `
  --set-secrets="APP_PASSWORD_HASH=app-password-hash:latest,APP_SECRET_KEY=app-secret-key:latest,GEMINI_API_KEY=gemini-api-key:latest" `
  --memory=2Gi --cpu=2 --min-instances=1 --allow-unauthenticated
```

### Uploaded documents have to outlive the container

Cloud Run's filesystem is ephemeral and not shared between instances, so documents uploaded
through the UI would vanish on the next restart. A GCS bucket is mounted at the documents
directory instead, which needs `--execution-environment=gen2`. `RAG_DOCS_DIR` already
existed, so this took no code change — the upload path writes straight through to the
bucket.

Verified rather than assumed: a document uploaded through the deployed API appeared in the
bucket, survived a forced new revision (a genuinely different container), was still indexed
afterwards, and deleting it through the API removed it from the bucket too.

### Two things that only show up in the container

- **The UI 404s if the frontend path is resolved by walking up from the package file.** That
  works under an editable install and breaks once the package is pip-installed, where the
  walk lands in site-packages. `FRONTEND_DIST` plus a cwd-relative fallback covers both.
- **`/healthz` is reserved on Cloud Run.** Google's frontend answers it with its own HTML
  404 and never forwards it, so a probe there reports a healthy service as down. The
  liveness route is `/api/live`. The give-away was that `/nonexistent-path` returned
  FastAPI's JSON 404 while `/healthz` returned Google's HTML one.

### Running costs and knobs

`--min-instances=1` keeps one instance warm, because a cold start reloads the embedding
model *and* reindexes every document. That is roughly $8–12/month for 2 vCPU and 2 GiB
always on. Drop to `--min-instances=0` to pay almost nothing and accept 20–30 seconds on the
first request after idle.

`--memory=2Gi` is not optional: torch, FAISS and the embedding model do not fit in the
512Mi default. Torch is installed from the CPU index, since the CUDA build is gigabytes and
useless here, and the embedding model is baked into the image so a cold start never depends
on Hugging Face being reachable.

Secrets come from Secret Manager as environment variables; `.dockerignore` keeps `.env` out
of the image. The runtime service account holds `secretAccessor` on each secret
individually and `objectAdmin` on only the documents bucket, rather than anything
project-wide.
