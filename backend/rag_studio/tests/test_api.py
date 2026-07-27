"""API tests with a stubbed agent, so no model loading or network calls happen."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from rag_studio.api import app as app_module
from rag_studio.api.app import create_app
from rag_studio.schema import Chunk, Citation, RagAnswer, RetrievedChunk


@dataclass(frozen=True)
class _Decision:
    route: str = "retrieve"
    retriever: str = "hybrid"
    parent_context: bool = False
    multi_query: bool = False
    hyde: bool = False
    rewrite_before_retrieval: bool = False
    reason: str = "Specific fact question."


@dataclass(frozen=True)
class _Grade:
    is_relevant: bool = True
    score: float = 0.833
    reason: str = "Token overlap 0.83."


@dataclass(frozen=True)
class _TraceEvent:
    node: str
    message: str
    details: dict[str, object]


class _StubAgent:
    def __init__(self, answer_text: str = "94.28% Binary F1 [1].") -> None:
        self.answer_text = answer_text
        self.raise_on_run: RuntimeError | None = None

    def run(self, question: str, top_k: int | None = None) -> dict[str, object]:
        if self.raise_on_run is not None:
            raise self.raise_on_run
        context = RetrievedChunk(
            chunk=Chunk(
                id="c1",
                text="Achieved 94.28% Binary F1 (Late Fusion).",
                metadata={"title": "resume.pdf", "page": 1, "chunk_index": 2},
            ),
            score=0.42,
        )
        return {
            "answer": RagAnswer(
                question=question,
                answer=self.answer_text,
                citations=[Citation(source_id=1, title="resume.pdf", location="page 1", score=0.42)],
                contexts=[context],
            ),
            "decision": _Decision(),
            "retrieval_grade": _Grade(),
            "retry_count": 0,
            "trace": [
                _TraceEvent("route_query", "Selected route.", {"route": "retrieve"}),
                _TraceEvent("generate", "Produced final answer.", {"citations": 1}),
            ],
        }


@pytest.fixture
def stub_agent() -> _StubAgent:
    return _StubAgent()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, stub_agent: _StubAgent):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_load(self, docs_dir):  # noqa: ANN001
        self.agent = stub_agent
        self.chunks = 15
        self.documents = [app_module.DocumentOut(title="resume.pdf")]

    monkeypatch.setattr(app_module.AgentService, "load", fake_load)
    # Must be a context manager: lifespan is what populates the agent.
    with TestClient(create_app("docs")) as test_client:
        yield test_client


def test_health_reports_provider_and_corpus(client: TestClient) -> None:
    body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["provider"] == "gemini / gemini-3.6-flash"
    assert body["is_generated"] is True
    assert body["chunks"] == 15


def test_query_returns_answer_route_grade_and_trace(client: TestClient) -> None:
    response = client.post("/api/query", json={"question": "What binary F1 was achieved?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "94.28% Binary F1 [1]."
    assert body["is_generated"] is True
    assert body["refused"] is False
    assert body["route"]["retriever"] == "hybrid"
    assert body["grade"]["score"] == pytest.approx(0.833)
    assert [event["node"] for event in body["trace"]] == ["route_query", "generate"]
    assert body["contexts"][0]["page"] == 1
    assert body["elapsed_ms"] >= 0


def test_query_flags_extractive_answers_as_not_generated(
    client: TestClient, stub_agent: _StubAgent
) -> None:
    """A keyless run must never be presented in the UI as real generation."""
    stub_agent.answer_text = "No LLM model is configured, so this is an extractive answer."

    body = client.post("/api/query", json={"question": "Anything?"}).json()

    assert body["is_generated"] is False


def test_query_marks_refusals(client: TestClient, stub_agent: _StubAgent) -> None:
    stub_agent.answer_text = "Prakriti's GPA is not mentioned in the sources."

    body = client.post("/api/query", json={"question": "What is my GPA?"}).json()

    assert body["refused"] is True


def test_generation_failure_surfaces_as_an_error_not_a_plausible_answer(
    client: TestClient, stub_agent: _StubAgent
) -> None:
    stub_agent.raise_on_run = RuntimeError("Gemini request failed for model x: 401")

    response = client.post("/api/query", json={"question": "Anything?"})

    assert response.status_code == 502
    assert "401" in response.json()["detail"]


def test_empty_question_is_rejected(client: TestClient) -> None:
    assert client.post("/api/query", json={"question": ""}).status_code == 422


JOB_DESCRIPTION = """
Requirements:
- Build RAG pipelines with FAISS and embeddings
- Lead a team of twelve site reliability engineers
"""


class _StubTailor:
    """Stands in for ResumeTailor with one supported and one missing requirement."""

    def tailor(self, job_description: str, max_requirements: int = 25):
        from rag_studio.tailoring.matching import (
            MATCHED,
            MISSING,
            Requirement,
            RequirementMatch,
        )
        from rag_studio.tailoring.service import TailoredBullet, TailoringResult

        evidence = RetrievedChunk(
            chunk=Chunk(
                id="c1",
                text="Built a local RAG pipeline with FAISS embeddings.",
                metadata={"title": "ai_ml.pdf", "page": 1},
            ),
            score=0.8,
        )
        matches = [
            RequirementMatch(
                Requirement(1, "Build RAG pipelines with FAISS and embeddings"),
                MATCHED,
                0.75,
                [evidence],
            ),
            RequirementMatch(
                Requirement(2, "Lead a team of twelve site reliability engineers"),
                MISSING,
                0.0,
                [],
            ),
        ]
        return TailoringResult(
            job_description=job_description,
            matches=matches,
            bullets=[TailoredBullet(1, "Built a RAG pipeline with FAISS embeddings.", [1])],
            recommended_resume="ai_ml.pdf",
            coverage=0.5,
            citations=[Citation(source_id=1, title="ai_ml.pdf", location="page 1", score=0.8)],
            contexts=[evidence],
            trace=[],
            is_generated=True,
        )


@pytest.fixture
def tailor_client(monkeypatch: pytest.MonkeyPatch, stub_agent: _StubAgent):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_load(self, docs_dir):  # noqa: ANN001
        self.agent = stub_agent
        self.tailor = _StubTailor()
        self.chunks = 15
        self.documents = [app_module.DocumentOut(title="ai_ml.pdf")]

    monkeypatch.setattr(app_module.AgentService, "load", fake_load)
    with TestClient(create_app("docs")) as test_client:
        yield test_client


def test_tailor_returns_gaps_bullets_and_coverage(tailor_client: TestClient) -> None:
    response = tailor_client.post("/api/tailor", json={"job_description": JOB_DESCRIPTION})

    assert response.status_code == 200
    body = response.json()
    assert body["coverage"] == 0.5
    assert body["matched_count"] == 1
    assert body["missing_count"] == 1
    assert body["recommended_resume"] == "ai_ml.pdf"
    assert body["bullets"][0]["requirement_id"] == 1
    assert body["bullets"][0]["source_ids"] == [1]

    statuses = {r["id"]: r["status"] for r in body["requirements"]}
    assert statuses == {1: "matched", 2: "missing"}


def test_tailor_shows_no_evidence_for_a_gap(tailor_client: TestClient) -> None:
    """A gap listing "supporting" evidence would mislead the user into claiming it."""
    body = tailor_client.post(
        "/api/tailor", json={"job_description": JOB_DESCRIPTION}
    ).json()

    gap = next(r for r in body["requirements"] if r["status"] == "missing")

    assert gap["evidence"] == []
    assert gap["id"] not in {b["requirement_id"] for b in body["bullets"]}


def test_tailor_rejects_a_job_description_that_is_too_short(
    tailor_client: TestClient,
) -> None:
    assert tailor_client.post("/api/tailor", json={"job_description": "AI job"}).status_code == 422


def test_tailor_without_documents_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module.AgentService, "load", lambda self, docs_dir: None)
    with TestClient(create_app("docs")) as client:
        response = client.post("/api/tailor", json={"job_description": JOB_DESCRIPTION})

    assert response.status_code == 503


def test_query_without_documents_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_module.AgentService,
        "load",
        lambda self, docs_dir: None,
    )
    with TestClient(create_app("docs")) as client:
        response = client.post("/api/query", json={"question": "Anything?"})

    assert response.status_code == 503
    assert "No documents are loaded" in response.json()["detail"]


def test_frontend_is_served_when_built(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Regression: the API worked but / returned 404 inside the container.

    _frontend_dist walked up from the package file, which lands in site-packages once the
    package is pip-installed rather than installed editable. FRONTEND_DIST and a
    cwd-relative fallback cover that.
    """
    dist = tmp_path / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>ok</title>", encoding="utf-8")
    monkeypatch.setenv("FRONTEND_DIST", str(dist))
    monkeypatch.setattr(app_module.AgentService, "load", lambda self, docs_dir=None: None)

    with TestClient(create_app("docs")) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "ok" in response.text


def test_missing_frontend_leaves_the_api_working(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("FRONTEND_DIST", str(tmp_path / "nothing-here"))
    monkeypatch.setattr(app_module.AgentService, "load", lambda self, docs_dir=None: None)

    with TestClient(create_app("docs")) as client:
        assert client.get("/healthz").status_code == 200
