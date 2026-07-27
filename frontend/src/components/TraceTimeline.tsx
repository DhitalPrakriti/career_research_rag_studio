import type { TraceEvent } from "../api";

interface TraceTimelineProps {
  trace: TraceEvent[];
}

/** The agent's steps in order. This is the part that makes the graph legible:
 *  a rewrite-and-retry shows up as a repeated retrieve_answer node. */
export function TraceTimeline({ trace }: TraceTimelineProps) {
  return (
    <section className="card">
      <h2>Agent trace</h2>
      {trace.length === 0 ? (
        <p className="empty">No trace events were recorded.</p>
      ) : (
        <ol className="trace">
          {trace.map((event) => (
            <li key={event.step}>
              <div className="trace-node">
                <code>{event.node}</code>
                <small>{event.message}</small>
              </div>
              {Object.keys(event.details).length > 0 && (
                <dl className="trace-details">
                  {Object.entries(event.details).map(([key, value]) => (
                    <div key={key}>
                      <dt>{key}</dt>
                      <dd>{value}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
