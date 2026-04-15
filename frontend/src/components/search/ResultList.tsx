import type { SearchResult } from "@/lib/api-client";
import { ResultCard } from "./ResultCard";

export function ResultList({
  results,
  query,
}: {
  results: SearchResult[];
  query: string;
}) {
  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
            Ranked sources
          </p>
          <h2 className="font-display mt-2 text-2xl text-white">
            Top passages
          </h2>
        </div>
        <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-[rgb(var(--muted))]">
          {results.length} result{results.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="space-y-3">
        {results.length > 0 ? (
          results.map((result) => (
            <ResultCard
              key={result.id}
              id={`source-${result.id}`}
              result={result}
              query={query}
            />
          ))
        ) : (
          <div className="glass-card rounded-[26px] p-6 text-sm leading-7 text-[rgb(var(--muted))]">
            No passages matched this query yet.
          </div>
        )}
      </div>
    </section>
  );
}
