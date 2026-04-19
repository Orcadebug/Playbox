"use client";

import { useEffect, useState } from "react";
import {
  demoQueries,
  checkHealthz,
  streamSearch,
  type ConnectorSearchConfig,
  type RawSearchSource,
  type SearchResponse,
} from "@/lib/api-client";
import { AnswerCard } from "./AnswerCard";
import { ResultList } from "./ResultList";
import { SearchBar } from "./SearchBar";
import { SourceControls } from "./SourceControls";

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
  const [includeStoredSources, setIncludeStoredSources] = useState(true);
  const [rawSourceName, setRawSourceName] = useState("Raw scratchpad");
  const [rawSourceContent, setRawSourceContent] = useState("");
  const [webhookEnabled, setWebhookEnabled] = useState(false);
  const [webhookContent, setWebhookContent] = useState("");
  const [answerLoading, setAnswerLoading] = useState(false);

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

  function buildSourcePayload(): {
    rawSources: RawSearchSource[];
    connectorConfigs: ConnectorSearchConfig[];
  } {
    const rawSources: RawSearchSource[] = rawSourceContent.trim()
      ? [
          {
            id: "inline-raw-source",
            name: rawSourceName.trim() || "Raw scratchpad",
            content: rawSourceContent,
            media_type: "text/plain",
            source_type: "raw",
          },
        ]
      : [];

    const connectorConfigs: ConnectorSearchConfig[] =
      webhookEnabled && webhookContent.trim()
        ? [
            {
              connector_id: "webhook",
              documents: [
                {
                  name: "webhook-inline.txt",
                  content: webhookContent,
                  media_type: "text/plain",
                },
              ],
            },
          ]
        : [];

    return { rawSources, connectorConfigs };
  }

  async function runSearch(nextQuery?: string, answerMode: "off" | "llm" = "off") {
    const finalQuery = (nextQuery ?? query).trim();
    if (!finalQuery) {
      return;
    }

    setLoading(true);
    setAnswerLoading(answerMode === "llm");
    setSelectedCitation(null);
    const { rawSources, connectorConfigs } = buildSourcePayload();

    try {
      const result = await streamSearch({
        query: finalQuery,
        answerMode,
        includeStoredSources,
        rawSources,
        connectorConfigs,
        mode: variant,
      });

      setResponse(result);
      setQuery(finalQuery);
    } finally {
      setLoading(false);
      setAnswerLoading(false);
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
  const answerNotice = response?.fallback ? response?.notice : undefined;
  const sourceErrors = response?.sourceErrors ?? [];
  const uniqueSourceCount = new Set(effectiveResults.map((result) => result.sourceLabel)).size;

  return (
    <div className="space-y-6">
      <SearchBar
        value={query}
        onChange={setQuery}
        onSubmit={() => void runSearch()}
        loading={loading}
        examples={variant === "demo" ? demoQueries : exampleQueries}
      />

      <SourceControls
        includeStoredSources={includeStoredSources}
        onIncludeStoredSourcesChange={setIncludeStoredSources}
        rawSourceName={rawSourceName}
        onRawSourceNameChange={setRawSourceName}
        rawSourceContent={rawSourceContent}
        onRawSourceContentChange={setRawSourceContent}
        webhookEnabled={webhookEnabled}
        onWebhookEnabledChange={setWebhookEnabled}
        webhookContent={webhookContent}
        onWebhookContentChange={setWebhookContent}
        disabled={loading}
      />

      <div className="glass-card rounded-[28px] p-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
              Retrieval status
            </p>
            <h2 className="font-display mt-2 text-2xl text-white">
              {response ? "Spans loaded" : "Ready to retrieve"}
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

        {sourceErrors.length > 0 ? (
          <div className="mt-4 rounded-2xl border border-amber-300/20 bg-amber-400/10 p-4 text-sm leading-6 text-amber-100">
            {sourceErrors.map((error) => error.message).join(" ")}
          </div>
        ) : null}

        <div className="mt-4 grid gap-3 sm:grid-cols-4">
          {[
            {
              label: "Query",
              value: response?.query || query || "Awaiting input",
            },
            {
              label: "Spans",
              value: String(effectiveResults.length),
            },
            {
              label: "Sources",
              value: String(uniqueSourceCount),
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
              <div className="mt-2 text-sm font-medium text-white">{item.value}</div>
            </div>
          ))}
        </div>
      </div>

      <ResultList results={effectiveResults} query={response?.query ?? query} />

      {response && effectiveResults.length > 0 ? (
        <section className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
                Optional answer
              </p>
              <h2 className="font-display mt-2 text-2xl text-white">
                Generate synthesis
              </h2>
            </div>
            <button
              type="button"
              disabled={loading || answerLoading}
              onClick={() => void runSearch(response.query, "llm")}
              className="rounded-2xl bg-[rgb(var(--accent-strong))] px-5 py-3 text-sm font-semibold text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {answerLoading ? "Generating..." : "Generate answer"}
            </button>
          </div>
          {response.answer || answerNotice ? (
            <AnswerCard
              answer={response.answer?.summary ?? null}
              fallbackNotice={answerNotice}
              citations={citations}
              loading={answerLoading}
              onCitationClick={jumpToSource}
            />
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
