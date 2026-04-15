"use client";

import { useEffect, useState } from "react";
import {
  demoQueries,
  checkHealthz,
  streamSearch,
  type SearchResponse,
} from "@/lib/api-client";
import { AnswerCard } from "./AnswerCard";
import { ResultList } from "./ResultList";
import { SearchBar } from "./SearchBar";

const exampleQueries = [
  "Which customers complained about billing last week?",
  "Find mentions of a launch blocker in Slack.",
  "Show the row where churn risk was marked high.",
];

export function SearchWorkspace({
  initialQuery = "",
  variant = "standard",
}: {
  initialQuery?: string;
  variant?: "standard" | "demo";
}) {
  const [query, setQuery] = useState(initialQuery);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<"checking" | "up" | "down">("checking");

  useEffect(() => {
    void checkHealth();
    if (initialQuery) {
      void runSearch(initialQuery);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialQuery]);

  async function checkHealth() {
    const response = await checkHealthz();
    setHealth(response ? "up" : "down");
  }

  async function runSearch(nextQuery?: string) {
    const finalQuery = (nextQuery ?? query).trim();
    if (!finalQuery) {
      return;
    }

    setLoading(true);
    setSelectedCitation(null);

    try {
      const result = await streamSearch({
        query: finalQuery,
        skipAnswer: false,
        mode: variant,
      });

      setResponse(result);
      setQuery(finalQuery);
    } finally {
      setLoading(false);
    }
  }

  function jumpToSource(sourceId: string) {
    setSelectedCitation(sourceId);
    const node = document.getElementById(`source-${sourceId}`);
    node?.scrollIntoView({ behavior: "smooth", block: "start" });
    node?.classList.add("ring-2", "ring-cyan-300/30");
    globalThis.setTimeout(() => {
      node?.classList.remove("ring-2", "ring-cyan-300/30");
    }, 1200);
  }

  const effectiveResults = response?.results ?? [];
  const citations = response?.answer?.citations ?? [];
  const answerNotice = response?.answerError ?? (response?.fallback ? response.notice : undefined);

  return (
    <div className="space-y-6">
      <SearchBar
        value={query}
        onChange={setQuery}
        onSubmit={() => void runSearch()}
        loading={loading}
        examples={variant === "demo" ? demoQueries : exampleQueries}
      />

      <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
        <AnswerCard
          answer={response?.answer?.summary ?? null}
          fallbackNotice={answerNotice}
          citations={citations}
          loading={loading && !response}
          onCitationClick={jumpToSource}
        />

        <div className="glass-card rounded-[28px] p-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
                Query status
              </p>
              <h2 className="font-display mt-2 text-2xl text-white">
                {response ? "Results loaded" : "Ready to search"}
              </h2>
            </div>
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-[rgb(var(--muted))]">
              {selectedCitation ? `Focused ${selectedCitation}` : "No focus"}
            </span>
          </div>

          <div className="mt-5 rounded-2xl border border-white/10 bg-black/15 p-4 text-sm leading-7 text-[rgb(var(--muted))]">
            {health === "checking"
              ? "Checking backend health..."
              : health === "up"
                ? "Backend health check passed."
                : "Backend unavailable. Using mock data and local fallback responses."}
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            {[
              {
                label: "Query",
                value: response?.query || query || "Awaiting input",
              },
              {
                label: "Passages",
                value: String(effectiveResults.length),
              },
              {
                label: "Sources",
                value: String(response?.sources.length ?? 0),
              },
              {
                label: "Mode",
                value: response?.fallback ? "Mock fallback" : "Backend",
              },
            ].map((item) => (
              <div
                key={item.label}
                className="rounded-2xl border border-white/10 bg-black/15 p-4"
              >
                <div className="text-[11px] uppercase tracking-[0.24em] text-[rgb(var(--muted))]">
                  {item.label}
                </div>
                <div className="mt-2 text-sm font-medium text-white">
                  {item.value}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <ResultList results={effectiveResults} query={response?.query ?? query} />
    </div>
  );
}
