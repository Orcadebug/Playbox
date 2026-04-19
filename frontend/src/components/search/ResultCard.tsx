import type { ReactNode } from "react";
import type { SearchResult, SearchSpan } from "@/lib/api-client";

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

function highlightSpan(span: SearchSpan) {
  if (span.highlights.length === 0) {
    return span.snippet;
  }

  const parts: ReactNode[] = [];
  let cursor = 0;
  const highlights = [...span.highlights]
    .filter((highlight) => highlight.start >= 0 && highlight.end > highlight.start)
    .sort((left, right) => left.start - right.start);

  for (const highlight of highlights) {
    const start = Math.min(highlight.start, span.snippet.length);
    const end = Math.min(highlight.end, span.snippet.length);
    if (cursor < start) {
      parts.push(
        <span key={`text-${cursor}-${start}`}>{span.snippet.slice(cursor, start)}</span>,
      );
    }
    parts.push(
      <mark key={`mark-${start}-${end}`} className="rounded bg-cyan-400/15 px-1 text-inherit">
        {span.snippet.slice(start, end)}
      </mark>,
    );
    cursor = Math.max(cursor, end);
  }

  if (cursor < span.snippet.length) {
    parts.push(<span key={`text-${cursor}`}>{span.snippet.slice(cursor)}</span>);
  }

  return parts;
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
  const additionalSpans = result.primarySpan
    ? result.matchedSpans
        .filter(
          (span, index) =>
            index > 0 ||
            span.sourceStart !== result.primarySpan?.sourceStart ||
            span.sourceEnd !== result.primarySpan?.sourceEnd,
        )
        .slice(0, 3)
    : result.matchedSpans.slice(0, 3);

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
            <span className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-2.5 py-1 text-[11px] uppercase tracking-[0.22em] text-cyan-200">
              {result.sourceOrigin}
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

      <p className="mt-4 whitespace-pre-wrap text-sm leading-7 text-[rgb(var(--text))]">
        {result.primarySpan ? highlightSpan(result.primarySpan) : highlightSnippet(result.snippet, query)}
      </p>

      {result.primarySpan ? (
        <div className="mt-3 flex flex-wrap gap-2 text-xs text-[rgb(var(--muted))]">
          <span className="rounded-full border border-white/10 bg-black/15 px-3 py-1">
            chars {result.primarySpan.sourceStart}-{result.primarySpan.sourceEnd}
          </span>
          <span className="rounded-full border border-white/10 bg-black/15 px-3 py-1">
            {result.primarySpan.highlights.length > 0 ? "exact span" : "context span"}
          </span>
        </div>
      ) : null}

      {additionalSpans.length > 0 ? (
        <div className="mt-4 space-y-2">
          <div className="text-[11px] uppercase tracking-[0.22em] text-[rgb(var(--muted))]">
            More spans
          </div>
          {additionalSpans.map((span) => (
            <p
              key={`${span.sourceStart}-${span.sourceEnd}`}
              className="rounded-2xl border border-white/10 bg-black/15 px-4 py-3 text-xs leading-6 text-[rgb(var(--muted))]"
            >
              {highlightSpan(span)}
            </p>
          ))}
        </div>
      ) : null}

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
