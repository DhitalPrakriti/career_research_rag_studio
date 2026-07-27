"""FastAPI application exposing the career research agent.

Documents are ingested once at startup, not per request: ingestion loads an embedding
model and builds a FAISS index, which takes seconds, while a query takes milliseconds.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_studio.agents import CareerResearchAgent
from rag_studio.api.models import (
    BulletOut,
    CitationOut,
    ContextOut,
    DocumentOut,
    GradeOut,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RequirementOut,
    RouteOut,
    TailorRequest,
    TailorResponse,
    TraceEventOut,
)
from rag_studio.evaluation.golden_set import answer_refuses
from rag_studio.llm import resolve_provider
from rag_studio.tailoring import MATCHED, MISSING, PARTIAL, ResumeTailor

logger = logging.getLogger(__name__)

DEFAULT_DOCS_DIR = "docs"
EXTRACTIVE_MARKER = "No LLM model is configured"


class AgentService:
    """Holds the one ingested agent shared by all requests."""

    def __init__(self) -> None:
        self.agent: CareerResearchAgent | None = None
        self.tailor: ResumeTailor | None = None
        self.documents: list[DocumentOut] = []
        self.chunks = 0

    def load(self, docs_dir: Path) -> None:
        paths = sorted(
            path
            for path in docs_dir.glob("*")
            if path.suffix.lower() in {".pdf", ".txt", ".md"}
        )
        if not paths:
            logger.warning("No documents found in %s; queries will return no context.", docs_dir)
            return

        logger.info("Ingesting %d document(s) from %s", len(paths), docs_dir)
        started = time.perf_counter()
        agent = CareerResearchAgent()
        agent.ingest(list(paths))

        self.agent = agent
        # Shares the one ingested pipeline; tailoring needs the same index.
        self.tailor = ResumeTailor(pipeline=agent.pipeline)
        self.chunks = len(agent.pipeline.chunks)
        self.documents = [DocumentOut(title=path.name) for path in paths]
        logger.info(
            "Ingested %d chunks in %.1fs",
            self.chunks,
            time.perf_counter() - started,
        )

    def require_agent(self) -> CareerResearchAgent:
        if self.agent is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No documents are loaded. Put PDFs in {DEFAULT_DOCS_DIR}/ "
                    "(or set RAG_DOCS_DIR) and restart."
                ),
            )
        return self.agent

    def require_tailor(self) -> ResumeTailor:
        if self.tailor is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No documents are loaded. Put your resume PDFs in {DEFAULT_DOCS_DIR}/ "
                    "(or set RAG_DOCS_DIR) and restart."
                ),
            )
        return self.tailor


def _provider_description() -> tuple[str, bool]:
    """Provider label plus whether it produces real generation."""
    try:
        config = resolve_provider()
    except RuntimeError as exc:
        return f"misconfigured: {exc}", False
    return config.describe(), config.is_llm


def create_app(docs_dir: str | Path | None = None) -> FastAPI:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    service = AgentService()
    resolved_docs = Path(docs_dir or os.getenv("RAG_DOCS_DIR") or DEFAULT_DOCS_DIR)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.load(resolved_docs)
        yield

    app = FastAPI(
        title="Career Research RAG Studio",
        description="Agentic RAG over personal career documents, with a visible trace.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # The dev frontend runs on a different port under Vite. In production the built
    # assets are served by this same app, so no cross-origin request happens.
    dev_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in dev_origins.split(",") if origin.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        provider, is_generated = _provider_description()
        return HealthResponse(
            status="ok" if service.agent is not None else "no_documents",
            provider=provider,
            is_generated=is_generated,
            documents=service.documents,
            chunks=service.chunks,
        )

    @app.post("/api/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        agent = service.require_agent()
        started = time.perf_counter()
        try:
            state = agent.run(request.question, top_k=request.top_k)
        except RuntimeError as exc:
            # Generation failures are deliberately loud rather than silently extractive,
            # so surface them as a real error instead of a plausible-looking answer.
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        answer = state["answer"]
        provider, provider_is_llm = _provider_description()
        is_generated = provider_is_llm and EXTRACTIVE_MARKER not in answer.answer

        return QueryResponse(
            question=answer.question,
            answer=answer.answer,
            provider=provider,
            is_generated=is_generated,
            refused=answer_refuses(answer.answer),
            route=_route_out(state.get("decision")),
            grade=_grade_out(state.get("retrieval_grade")),
            retry_count=int(state.get("retry_count", 0)),
            citations=[
                CitationOut(
                    source_id=citation.source_id,
                    title=citation.title,
                    location=citation.location,
                    score=citation.score,
                )
                for citation in answer.citations
            ],
            contexts=[
                ContextOut(
                    source_id=index,
                    text=result.chunk.text,
                    score=result.score,
                    title=str(result.chunk.metadata.get("title") or "Untitled"),
                    page=_as_int(result.chunk.metadata.get("page")),
                    chunk_index=_as_int(result.chunk.metadata.get("chunk_index")),
                )
                for index, result in enumerate(answer.contexts, start=1)
            ],
            trace=[
                TraceEventOut(
                    step=step,
                    node=event.node,
                    message=event.message,
                    details={key: str(value) for key, value in event.details.items()},
                )
                for step, event in enumerate(state.get("trace", []), start=1)
            ],
            elapsed_ms=elapsed_ms,
        )

    @app.post("/api/tailor", response_model=TailorResponse)
    def tailor(request: TailorRequest) -> TailorResponse:
        tailor_service = service.require_tailor()
        started = time.perf_counter()
        try:
            result = tailor_service.tailor(
                request.job_description,
                max_requirements=request.max_requirements,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        provider, _ = _provider_description()
        source_ids = {
            context.chunk.text: index
            for index, context in enumerate(result.contexts, start=1)
        }

        return TailorResponse(
            provider=provider,
            is_generated=result.is_generated,
            coverage=result.coverage,
            recommended_resume=result.recommended_resume,
            matched_count=sum(1 for m in result.matches if m.status == MATCHED),
            partial_count=sum(1 for m in result.matches if m.status == PARTIAL),
            missing_count=sum(1 for m in result.matches if m.status == MISSING),
            requirements=[
                RequirementOut(
                    id=match.requirement.id,
                    text=match.requirement.text,
                    status=match.status,
                    score=match.score,
                    evidence=[
                        _context_out(item, source_ids.get(item.chunk.text, 0))
                        for item in match.evidence
                    ],
                )
                for match in result.matches
            ],
            bullets=[
                BulletOut(
                    requirement_id=bullet.requirement_id,
                    text=bullet.text,
                    source_ids=bullet.source_ids,
                )
                for bullet in result.bullets
            ],
            citations=[
                CitationOut(
                    source_id=citation.source_id,
                    title=citation.title,
                    location=citation.location,
                    score=citation.score,
                )
                for citation in result.citations
            ],
            contexts=[
                _context_out(context, index)
                for index, context in enumerate(result.contexts, start=1)
            ],
            trace=[
                TraceEventOut(
                    step=step,
                    node=event.node,
                    message=event.message,
                    details={key: str(value) for key, value in event.details.items()},
                )
                for step, event in enumerate(result.trace, start=1)
            ],
            elapsed_ms=elapsed_ms,
        )

    _mount_frontend(app)
    return app


def _context_out(result: object, source_id: int) -> ContextOut:
    chunk = getattr(result, "chunk")
    metadata = chunk.metadata
    return ContextOut(
        source_id=source_id,
        text=chunk.text,
        score=float(getattr(result, "score", 0.0)),
        title=str(metadata.get("title") or "Untitled"),
        page=_as_int(metadata.get("page")),
        chunk_index=_as_int(metadata.get("chunk_index")),
    )


# Intentionally no module-level `app = create_app()`. That would run load_dotenv() at
# import time, leaking real credentials into any process that merely imports this module
# — including the test suite, which would then make live billed API calls. Run the server
# with the factory instead:
#     uvicorn --factory rag_studio.api.app:create_app


def _route_out(decision: object) -> RouteOut | None:
    if decision is None:
        return None
    return RouteOut(
        route=getattr(decision, "route", ""),
        retriever=getattr(decision, "retriever", ""),
        parent_context=bool(getattr(decision, "parent_context", False)),
        multi_query=bool(getattr(decision, "multi_query", False)),
        hyde=bool(getattr(decision, "hyde", False)),
        rewrite_before_retrieval=bool(getattr(decision, "rewrite_before_retrieval", False)),
        reason=getattr(decision, "reason", ""),
    )


def _grade_out(grade: object) -> GradeOut | None:
    if grade is None:
        return None
    return GradeOut(
        is_relevant=bool(getattr(grade, "is_relevant", False)),
        score=float(getattr(grade, "score", 0.0)),
        reason=getattr(grade, "reason", ""),
    )


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built frontend when it exists, so one container serves both."""
    dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if not (dist / "index.html").exists():
        return

    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(dist / "index.html")

