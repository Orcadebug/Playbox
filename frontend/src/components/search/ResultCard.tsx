import type { SearchResult } from "@/lib/api-client";

function highlightSnippet(snippet: string, query: string) {
  const tokens = query
    .toLowerCase()
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 3)
    .slice(0, 6);

  if (tokens.length === 0) {
    return snippet;
  }

  const pattern = new RegExp(`(${tokens.map(escapeRegExp).join("|")})`, "ig");
  const parts = snippet.split(pattern);

  return parts.map((part, index) =>
    tokens.some((token) => token.toLowerCase() === part.toLowerCase()) ? (
      <mark key={`${part}-${index}`} className="rounded bg-cyan-400/15 px-1 text-inherit">
        {part}
      </mark>
    ) : (
      <span key={`${part}-${index}`}>{part}</span>
    ),
  );
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function ResultCard({
  result,
  query,
  id,
}: {
  result: SearchResult;
  query: string;
  id?: string;
}) {
  return (
    <article
      id={id}
      className="glass-card rounded-[26px] p-5 transition hover:border-cyan-300/20"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-display text-xl text-white">
              {result.title}
            </h3>
            <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] uppercase tracking-[0.22em] text-[rgb(var(--muted))]">
              {result.kind}
            </span>
          </div>
          <p className="mt-2 text-sm text-[rgb(var(--muted))]">
            {result.sourceLabel} · {result.location}
          </p>
        </div>
        <div className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-3 py-1 text-sm font-semibold text-cyan-200">
          {Math.round(result.score * 100)}%
        </div>
      </div>

      <p className="mt-4 text-sm leading-7 text-[rgb(var(--text))]">
        {highlightSnippet(result.snippet, query)}
      </p>

      {result.tags.length > 0 ? (
        <div className="mt-4 flex flex-wrap gap-2">
          {result.tags.map((tag) => (
            <span
              key={tag}
              className="rounded-full border border-white/10 bg-black/15 px-3 py-1 text-xs text-[rgb(var(--muted))]"
            >
              {tag}
            </span>
          ))}
        </div>
      ) : null}
    </article>
  );
}
