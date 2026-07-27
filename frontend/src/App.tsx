import { useCallback, useEffect, useRef, useState } from "react";
import { askQuestion, fetchHealth, type HealthResponse, type QueryResponse } from "./api";
import { Badge } from "./components/Badge";
import { RouteCard } from "./components/RouteCard";
import { SourceList } from "./components/SourceList";
import { StatRow } from "./components/StatRow";
import { TraceTimeline } from "./components/TraceTimeline";

const EXAMPLE_QUESTIONS = [
  "What binary F1 score was achieved in the capstone project?",
  "Which resume best fits an AI engineering role?",
  "What database was used for conversation memory?",
  "What is my GPA?",
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const pending = useRef<AbortController | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  useEffect(() => () => pending.current?.abort(), []);

  const submit = useCallback(async (asked: string) => {
    const trimmed = asked.trim();
    if (!trimmed) return;

    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;

    setLoading(true);
    setError(null);
    try {
      setResult(await askQuestion(trimmed, controller.signal));
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setResult(null);
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally {
      if (pending.current === controller) {
        pending.current = null;
        setLoading(false);
      }
    }
  }, []);

  const noDocuments = health?.status === "no_documents";

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1>Career Research RAG Studio</h1>
          <p>
            Agentic retrieval over personal career documents. Every answer shows the route
            the agent chose, how it graded its own retrieval, and the exact chunks it read.
          </p>
        </div>
        {health && (
          <div className="corpus">
            <Badge tone={health.is_generated ? "good" : "warning"}>{health.provider}</Badge>
            <span>
              {health.documents.length} document{health.documents.length === 1 ? "" : "s"} ·{" "}
              {health.chunks} chunks
            </span>
          </div>
        )}
      </header>

      <section className="ask">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void submit(question);
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
                void submit(example);
              }}
            >
              {example}
            </button>
          ))}
        </div>
      </section>

      {noDocuments && (
        <p className="notice">
          No documents are loaded. Put PDFs in <code>docs/</code> (or set{" "}
          <code>RAG_DOCS_DIR</code>) and restart the server.
        </p>
      )}

      {error && (
        <p className="notice is-error" role="alert">
          {error}
        </p>
      )}

      {loading && !result && (
        <p className="notice">
          <span className="spinner" aria-hidden="true" />
          Routing, retrieving, grading, generating…
        </p>
      )}

      {result && (
        <>
          <StatRow result={result} />

          <section className="card" style={{ marginTop: 16 }}>
            <div className="answer-head">
              <h2 style={{ margin: 0, flex: 1 }}>Answer</h2>
              <Badge tone={result.is_generated ? "good" : "warning"}>
                {result.is_generated ? "Generated" : "Extractive fallback"}
              </Badge>
              {result.refused && <Badge tone="neutral">Declined to answer</Badge>}
              {result.retry_count > 0 && (
                <Badge tone="warning">
                  {result.retry_count} rewrite{result.retry_count === 1 ? "" : "s"}
                </Badge>
              )}
            </div>

            <p className="answer-body">{result.answer}</p>

            {!result.is_generated && (
              <p className="notice">
                No LLM is configured, so this is extracted source text rather than a
                generated answer. Set a provider in <code>.env</code> to get real
                generation — and never score a run like this for faithfulness.
              </p>
            )}

            {result.refused && result.is_generated && (
              <p className="notice">
                The agent reported that the documents do not contain this. For the
                negative-control questions in the golden set, that is the correct outcome.
              </p>
            )}
          </section>

          <RouteCard route={result.route} grade={result.grade} />
          <TraceTimeline trace={result.trace} />
          <SourceList citations={result.citations} contexts={result.contexts} />
        </>
      )}
    </div>
  );
}
