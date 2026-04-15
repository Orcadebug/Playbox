"use client";

import type { SearchCitation } from "@/lib/api-client";

export function AnswerCard({
  answer,
  fallbackNotice,
  citations,
  loading,
  onCitationClick,
}: {
  answer: string | null;
  fallbackNotice?: string;
  citations: SearchCitation[];
  loading?: boolean;
  onCitationClick?: (sourceId: string) => void;
}) {
  return (
    <section className="glass-card rounded-[28px] p-5 md:p-6">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
            Answer
          </p>
          <h2 className="font-display mt-2 text-2xl text-white">
            Synthesized response
          </h2>
        </div>
        <span className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-200">
          {loading ? "Generating" : "Ready"}
        </span>
      </div>

      {loading ? (
        <div className="mt-5 space-y-3">
          <div className="h-4 w-5/6 animate-pulse rounded-full bg-white/10" />
          <div className="h-4 w-4/6 animate-pulse rounded-full bg-white/10" />
          <div className="h-4 w-2/3 animate-pulse rounded-full bg-white/10" />
        </div>
      ) : (
        <div className="mt-5 space-y-4">
          <p className="whitespace-pre-wrap text-sm leading-7 text-[rgb(var(--text))] md:text-base">
            {answer ?? "No synthesized answer was returned for this query."}
          </p>
          {fallbackNotice ? (
            <div className="rounded-2xl border border-amber-300/20 bg-amber-400/10 p-4 text-sm leading-6 text-amber-100">
              {fallbackNotice}
            </div>
          ) : null}
          {citations.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {citations.map((citation) => (
                <button
                  key={`${citation.sourceId}-${citation.label}`}
                  type="button"
                  onClick={() => onCitationClick?.(citation.sourceId)}
                  className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs font-medium text-[rgb(var(--muted))] transition hover:border-cyan-300/30 hover:text-white"
                >
                  {citation.label}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}
