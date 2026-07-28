type Tone = "good" | "warning" | "critical" | "neutral";

interface BadgeProps {
  tone: Tone;
  children: React.ReactNode;
}

/** A coloured dot plus a text label. The label is mandatory: status colour never
 *  carries meaning on its own, which also keeps this readable under CVD. */
export function Badge({ tone, children }: BadgeProps) {
  return (
    <span className="badge" data-tone={tone}>
      {children}
    </span>
  );
}
