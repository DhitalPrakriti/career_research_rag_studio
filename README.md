Career + Research RAG Studio
A production-grade RAG (Retrieval-Augmented Generation) system for querying
your personal career documents — resumes, job descriptions, class notes, and research papers.
Built by Prakriti Dhital as a portfolio project covering the full RAG lifecycle:
ingestion → retrieval → evaluation → agentic routing → deployment.

What it does

Upload your resumes, job descriptions, cover letters, class notes, and research papers
Ask questions like:

"How should I tailor my resume for this job description?"
"Compare these two job descriptions — which one fits my skills better?"
"What skills from my resume are missing for this role?"
"Summarize what I've learned in my networking course"


Get grounded, cited answers with source references
Evaluate retrieval quality automatically using RAGAS


Architecture
data/uploads/          ← your documents go here
      ↓
backend/ingestion/     ← PDF loading, chunking, embedding, FAISS index
      ↓
backend/retrieval/     ← hybrid search (dense + BM25), reranking, HyDE
      ↓
backend/generation/    ← prompt grounding, citation, LLM call
      ↓
backend/agents/        ← LangGraph router: decides which retriever to use
      ↓
backend/api/           ← FastAPI endpoints
      ↓
frontend/              ← React UI with chat + citation display
      ↓
evaluation/            ← RAGAS metrics, golden test set, comparison dashboard

6-Week Build Plan
WeekFocusStatus1Ingestion pipeline (loader, chunker, embeddings, FAISS)🚧 In progress2Answer generation + citations + basic UI⬜3Hybrid retrieval + reranking⬜4RAGAS evaluation + golden test set⬜5Agentic routing with LangGraph⬜6Deploy to Google Cloud Run + writeup⬜

Tech Stack
LayerTechnologyDocument parsingPyMuPDFChunkingLangChain splitters + custom recursiveEmbeddingssentence-transformers → OpenAIVector storeFAISS → QdrantSparse retrievalBM25 (rank-bm25)Rerankercross-encoder (sentence-transformers)LLMGoogle Gemini / GPT-4o-miniOrchestrationLangChain + LangGraphEvaluationRAGASBackendFastAPIFrontendReact + TypeScriptObservabilityLangSmithDeploymentDocker + Google Cloud Run

Setup
bash# 1. Clone and enter project
git clone <your-repo>
cd career-rag-studio

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Add your documents
cp your-resume.pdf data/uploads/
cp job-description.pdf data/uploads/

# 6. Build the index
cd backend/ingestion
python embeddings.py

# 7. Start the API
cd ../api
uvicorn main:app --reload

RAGAS Evaluation Results
ConfigFaithfulnessAnswer RelevancyContext PrecisionBaseline (fixed chunks)---Recursive chunking---+ Hybrid search---+ Reranking---+ HyDE---
(Results filled in during Week 4)

Local RAGAS With Ollama

First generate RAGAS-compatible records from the golden set:

```powershell
$env:OLLAMA_MODEL="llama3"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python -m rag_studio.eval_cli --output evaluation\runs\all_resumes_baseline.jsonl
```

Then run one RAGAS metric at a time with Ollama. Local judging is slow, so start with
`--limit 1`:

```powershell
python -m rag_studio.ragas_eval_cli --input evaluation\runs\all_resumes_baseline.jsonl --output evaluation\runs\ragas_ollama_context_recall_limit1.jsonl --limit 1 --judge-model llama3 --num-thread 4 --metrics context_recall
```

Supported metrics are `faithfulness`, `answer_relevancy`, `context_precision`, and
`context_recall`.

Key Learning Outcomes

End-to-end RAG pipeline design
Chunking strategy tradeoffs
Dense vs sparse retrieval
Cross-encoder reranking
RAGAS evaluation framework
Agentic query routing with LangGraph
Production deployment with Docker + Cloud Run
