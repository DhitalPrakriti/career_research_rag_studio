import { useCallback, useEffect, useRef, useState } from "react";
import {
  askQuestion,
  fetchHealth,
  fetchSession,
  logout,
  tailorResume,
  UnauthorizedError,
  type HealthResponse,
  type QueryResponse,
  type TailorResponse,
} from "./api";
import { Badge } from "./components/Badge";
import { DocumentsPanel } from "./components/DocumentsPanel";
import { LoginScreen } from "./components/LoginScreen";
import { RouteCard } from "./components/RouteCard";
import { SourceList } from "./components/SourceList";
import { StatRow } from "./components/StatRow";
import { TailorResult } from "./components/TailorResult";
import { TraceTimeline } from "./components/TraceTimeline";

type Mode = "tailor" | "ask" | "documents";

const EXAMPLE_QUESTIONS = [
  "What binary F1 score was achieved in the capstone project?",
  "Which resume best fits an AI engineering role?",
  "What is my GPA?",
];

const SAMPLE_JD = `AI Engineer

Responsibilities:
- Build and evaluate RAG pipelines over internal documents
- Work with embeddings and vector search at scale
- Deploy containerised services to Google Cloud Run

Requirements:
- Strong Python, PyTorch and deep learning experience
- Experience with FAISS or a comparable vector database
- Familiarity with LLM evaluation and prompt engineering
- Kubernetes and Terraform for infrastructure automation
- 8 years leading a platform engineering team`;

export default function App() {
  const [mode, setMode] = useState<Mode>("tailor");
  const [jobDescription, setJobDescription] = useState("");
  const [question, setQuestion] = useState("");
  const [tailored, setTailored] = useState<TailorResponse | null>(null);
  const [answer, setAnswer] = useState<QueryResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [signedIn, setSignedIn] = useState<boolean | null>(null);
  const pending = useRef<AbortController | null>(null);

  const refreshHealth = useCallback(() => {
    fetchHealth()
      .then(setHealth)
      .catch((caught) => {
        setHealth(null);
        if (caught instanceof UnauthorizedError) setSignedIn(false);
      });
  }, []);

  // Ask the server whether a session is needed before rendering anything, so an
  // unauthenticated visitor never sees the app shell or a flash of empty panels.
  const checkSession = useCallback(() => {
    fetchSession()
      .then((session) => setSignedIn(session.authenticated))
      .catch(() => setSignedIn(false));
  }, []);

  useEffect(checkSession, [checkSession]);

  useEffect(() => {
    if (signedIn) refreshHealth();
  }, [signedIn, refreshHealth]);

  useEffect(() => () => pending.current?.abort(), []);

  const run = useCallback(async <T,>(task: (signal: AbortSignal) => Promise<T>, apply: (value: T) => void) => {
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;

    setLoading(true);
    setError(null);
    try {
      apply(await task(controller.signal));
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      if (caught instanceof UnauthorizedError) {
        setSignedIn(false);
        return;
      }
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally {
      if (pending.current === controller) {
        pending.current = null;
        setLoading(false);
      }
    }
  }, []);

  const submitJd = () => {
    if (jobDescription.trim().length < 20) return;
    setTailored(null);
    void run((signal) => tailorResume(jobDescription, signal), setTailored);
  };

  const submitQuestion = (asked: string) => {
    if (!asked.trim()) return;
    setAnswer(null);
    void run((signal) => askQuestion(asked.trim(), signal), setAnswer);
  };

  const switchMode = (next: Mode) => {
    pending.current?.abort();
    setMode(next);
    setError(null);
    setLoading(false);
  };

  const noDocuments = health?.status === "no_documents";

  if (signedIn === null) {
    return (
      <div className="shell">
        <p className="notice">
          <span className="spinner" aria-hidden="true" />
          Loading…
        </p>
      </div>
    );
  }

  if (!signedIn) return <LoginScreen onSignedIn={checkSession} />;

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1>Career Research RAG Studio</h1>
          <p>
            Paste a job description and get your resume rewritten against it — every bullet
            grounded in your own resume text, every unmet requirement listed as a gap rather
            than invented.
          </p>
        </div>
        {health && (
          <div className="corpus">
            <Badge tone={health.is_generated ? "good" : "warning"}>{health.provider}</Badge>
            <span>
              {health.documents.length} document{health.documents.length === 1 ? "" : "s"} ·{" "}
              {health.chunks} chunks
            </span>
            <button
              type="button"
              className="chip"
              onClick={() => {
                void logout().then(checkSession);
              }}
            >
              Sign out
            </button>
          </div>
        )}
      </header>

      <nav className="tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "tailor"}
          className={mode === "tailor" ? "tab is-active" : "tab"}
          onClick={() => switchMode("tailor")}
        >
          Tailor to a job
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "ask"}
          className={mode === "ask" ? "tab is-active" : "tab"}
          onClick={() => switchMode("ask")}
        >
          Ask a question
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "documents"}
          className={mode === "documents" ? "tab is-active" : "tab"}
          onClick={() => switchMode("documents")}
        >
          My documents
        </button>
      </nav>

      {mode === "documents" && <DocumentsPanel onChanged={refreshHealth} />}

      {noDocuments && mode !== "documents" && (
        <p className="notice">
          No documents are indexed. Open <strong>My documents</strong> to upload a resume.
        </p>
      )}

      {mode === "tailor" && (
        <section className="ask">
          <form
            onSubmit={(event) => {
              event.preventDefault();
              submitJd();
            }}
          >
            <textarea
              value={jobDescription}
              placeholder="Paste the full job description here…"
              aria-label="Job description"
              rows={12}
              onChange={(event) => setJobDescription(event.target.value)}
              disabled={loading}
            />
            <div className="form-actions">
              <button
                type="submit"
                disabled={loading || jobDescription.trim().length < 20}
              >
                {loading ? "Tailoring…" : "Tailor my resume"}
              </button>
              <button
                type="button"
                className="chip"
                disabled={loading}
                onClick={() => setJobDescription(SAMPLE_JD)}
              >
                Use a sample posting
              </button>
              {jobDescription && (
                <button
                  type="button"
                  className="chip"
                  disabled={loading}
                  onClick={() => {
                    setJobDescription("");
                    setTailored(null);
                  }}
                >
                  Clear
                </button>
              )}
            </div>
          </form>
        </section>
      )}

      {mode === "ask" && (
        <section className="ask">
          <form
            onSubmit={(event) => {
              event.preventDefault();
              submitQuestion(question);
            }}
          >
            <input
              type="text"
              value={question}
              placeholder="Ask about your resumes, projects, or coursework…"
              aria-label="Your question"
              onChange={(event) => setQuestion(event.target.value)}
              disabled={loading}
            />
            <button type="submit" disabled={loading || !question.trim()}>
              {loading ? "Thinking…" : "Ask"}
            </button>
          </form>
          <div className="examples">
            <span>Try:</span>
            {EXAMPLE_QUESTIONS.map((example) => (
              <button
                type="button"
                className="chip"
                key={example}
                disabled={loading}
                onClick={() => {
                  setQuestion(example);
                  submitQuestion(example);
                }}
              >
                {example}
              </button>
            ))}
          </div>
        </section>
      )}

      {error && (
        <p className="notice is-error" role="alert">
          {error}
        </p>
      )}

      {loading && (
        <p className="notice">
          <span className="spinner" aria-hidden="true" />
          {mode === "tailor"
            ? "Extracting requirements, retrieving evidence, rewriting bullets…"
            : "Routing, retrieving, grading, generating…"}
        </p>
      )}

      {mode === "tailor" && tailored && !loading && (
        <>
          <TailorResult result={tailored} />
          <TraceTimeline trace={tailored.trace} />
          <SourceList citations={tailored.citations} contexts={tailored.contexts} />
        </>
      )}

      {mode === "ask" && answer && !loading && (
        <>
          <StatRow result={answer} />
          <section className="card" style={{ marginTop: 16 }}>
            <div className="answer-head">
              <h2 style={{ margin: 0, flex: 1 }}>Answer</h2>
              <Badge tone={answer.is_generated ? "good" : "warning"}>
                {answer.is_generated ? "Generated" : "Extractive fallback"}
              </Badge>
              {answer.refused && <Badge tone="neutral">Declined to answer</Badge>}
              {answer.retry_count > 0 && (
                <Badge tone="warning">
                  {answer.retry_count} rewrite{answer.retry_count === 1 ? "" : "s"}
                </Badge>
              )}
            </div>
            <p className="answer-body">{answer.answer}</p>
            {!answer.is_generated && (
              <p className="notice">
                No LLM is configured, so this is extracted source text rather than a
                generated answer. Set a provider in <code>.env</code> for real generation.
              </p>
            )}
          </section>
          <RouteCard route={answer.route} grade={answer.grade} />
          <TraceTimeline trace={answer.trace} />
          <SourceList citations={answer.citations} contexts={answer.contexts} />
        </>
      )}
    </div>
  );
}
