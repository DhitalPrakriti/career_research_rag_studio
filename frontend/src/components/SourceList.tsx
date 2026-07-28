import type { Citation, Context } from "../api";

interface SourceListProps {
  citations: Citation[];
  contexts: Context[];
}

/** Citations with the retrieved chunk text behind a disclosure, so a claim can be
 *  checked against the exact text the model was given. */
export function SourceList({ citations, contexts }: SourceListProps) {
  if (citations.length === 0) {
    return (
      <section className="card">
        <h2>Sources</h2>
        <p className="empty">No sources were cited.</p>
      </section>
    );
  }

  const textById = new Map(contexts.map((context) => [context.source_id, context.text]));

  return (
    <section className="card">
      <h2>Sources</h2>
      {citations.map((citation) => (
        <details className="source" key={citation.source_id}>
          <summary>
            <span className="source-id">[{citation.source_id}]</span>
            <span className="source-title">{citation.title}</span>
            <span className="source-meta">
              {citation.location} · score {citation.score.toFixed(3)}
            </span>
          </summary>
          <p>{textById.get(citation.source_id) ?? "Chunk text unavailable."}</p>
        </details>
      ))}
    </section>
  );
}
