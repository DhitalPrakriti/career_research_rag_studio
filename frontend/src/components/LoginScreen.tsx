import { useState } from "react";
import { login } from "../api";

export function LoginScreen({ onSignedIn }: { onSignedIn: () => void }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!password) return;

    setBusy(true);
    setError(null);
    try {
      await login(password);
      setPassword("");
      onSignedIn();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not sign in.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1>Career Research RAG Studio</h1>
          <p>Sign in to tailor your resume and search your documents.</p>
        </div>
      </header>

      <section className="card login">
        <form onSubmit={submit}>
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            value={password}
            autoComplete="current-password"
            autoFocus
            onChange={(event) => setPassword(event.target.value)}
            disabled={busy}
          />
          <button type="submit" disabled={busy || !password}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>

        {error && (
          <p className="notice is-error" role="alert">
            {error}
          </p>
        )}
      </section>
    </div>
  );
}
