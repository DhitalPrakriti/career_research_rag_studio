import type { Grade, Route } from "../api";
import { Badge } from "./Badge";

interface RouteCardProps {
  route: Route | null;
  grade: Grade | null;
}

function Flag({ on }: { on: boolean }) {
  return <>{on ? "on" : "off"}</>;
}

/** What the router decided and how the retrieval graded — the "why" behind the answer. */
export function RouteCard({ route, grade }: RouteCardProps) {
  if (!route) return null;

  return (
    <section className="card">
      <h2>Routing decision</h2>
      <dl className="facts">
        <div>
          <dt>Route</dt>
          <dd>{route.route}</dd>
        </div>
        <div>
          <dt>Retriever</dt>
          <dd>{route.retriever}</dd>
        </div>
        <div>
          <dt>Parent context</dt>
          <dd>
            <Flag on={route.parent_context} />
          </dd>
        </div>
        <div>
          <dt>Multi-query</dt>
          <dd>
            <Flag on={route.multi_query} />
          </dd>
        </div>
        <div>
          <dt>HyDE</dt>
          <dd>
            <Flag on={route.hyde} />
          </dd>
        </div>
        <div>
          <dt>Rewrite first</dt>
          <dd>
            <Flag on={route.rewrite_before_retrieval} />
          </dd>
        </div>
      </dl>

      <p className="reason">{route.reason}</p>

      {grade && (
        <>
          <div className="answer-head" style={{ marginTop: 14, marginBottom: 0 }}>
            <Badge tone={grade.is_relevant ? "good" : "critical"}>
              {grade.is_relevant ? "Context judged relevant" : "Context judged weak"}
            </Badge>
          </div>
          <p className="reason" style={{ borderTop: 0, paddingTop: 8, marginTop: 8 }}>
            {grade.reason}
          </p>
        </>
      )}
    </section>
  );
}
