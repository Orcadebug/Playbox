const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const STORAGE_KEY = "waver.mock.sources";

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

type BackendSearchPayload = {
  query: string;
  results: Array<{
    document_id?: string | null;
    source_id?: string | null;
    source_name: string;
    title?: string | null;
    content: string;
    snippet?: string | null;
    score: number;
    citation_label?: string;
    metadata?: Record<string, unknown>;
    chunk_id?: string | null;
    source_origin?: "stored" | "raw" | "connector";
    primary_span?: BackendSearchSpan | null;
    matched_spans?: BackendSearchSpan[];
    channels?: string[];
    channel_scores?: Record<string, number>;
  }>;
  answer?: {
    markdown?: string;
    confidence?: string;
  } | null;
  answer_error?: string | null;
  source_errors?: BackendSourceError[];
  execution?: BackendExecution;
};

type BackendExecution = {
  tier?: string;
  channels?: string[];
  window_count?: number;
  scan_limit?: number;
  candidate_limit?: number;
  rerank_limit?: number;
  partial?: boolean;
  metadata_prefilter?: boolean;
  use_cache?: boolean;
  scanned_windows?: number;
  candidate_count?: number;
  shortlisted_candidates?: number;
  [key: string]: unknown;
};

type BackendSearchSpan = {
  text: string;
  snippet: string;
  source_start: number;
  source_end: number;
  snippet_start: number;
  snippet_end: number;
  offset_basis?: "source" | "parsed";
  highlights: Array<{
    start: number;
    end: number;
    source_start?: number;
    source_end?: number;
    offset_basis?: "source" | "parsed";
    text?: string;
  }>;
  location?: Record<string, unknown>;
};

type BackendSourceError = {
  source_type: string;
  connector_id?: string | null;
  source_name?: string | null;
  message: string;
};

type BackendSource = {
  id: string;
  name: string;
  source_type?: string;
  parser_name?: string | null;
  document_count?: number;
  metadata?: Record<string, unknown>;
};

type UploadPayload = {
  files?: File[];
  text?: string;
};

type SearchPayload = {
  query: string;
  skipAnswer?: boolean;
  answerMode?: "off" | "llm";
  includeStoredSources?: boolean;
  rawSources?: RawSearchSource[];
  connectorConfigs?: ConnectorSearchConfig[];
  mode?: "standard" | "demo";
};

export type RawSearchSource = {
  id?: string;
  name: string;
  content: JsonValue;
  media_type?: string;
  source_type?: string;
  metadata?: Record<string, unknown>;
};

export type ConnectorSearchConfig = {
  connector_id: "webhook" | "slack" | string;
  documents?: Array<Record<string, unknown>>;
  channels?: string[];
  limit?: number;
};

type ConnectorPayload = {
  connectorId: string;
};

type DemoPassage = {
  id: string;
  title: string;
  sourceLabel: string;
  kind: string;
  location: string;
  text: string;
  tags: string[];
};

export type SearchCitation = {
  label: string;
  sourceId: string;
};

export type SearchSpan = {
  text: string;
  snippet: string;
  sourceStart: number;
  sourceEnd: number;
  snippetStart: number;
  snippetEnd: number;
  offsetBasis: "source" | "parsed";
  highlights: Array<{
    start: number;
    end: number;
    sourceStart?: number;
    sourceEnd?: number;
    offsetBasis?: "source" | "parsed";
    text?: string;
  }>;
  location?: Record<string, unknown>;
};

export type SearchResult = {
  id: string;
  title: string;
  kind: string;
  sourceLabel: string;
  location: string;
  snippet: string;
  score: number;
  tags: string[];
  sourceOrigin: "stored" | "raw" | "connector";
  primarySpan: SearchSpan | null;
  matchedSpans: SearchSpan[];
  channels: string[];
  channelScores: Record<string, number>;
};

export type SearchAnswer = {
  summary: string;
  citations: SearchCitation[];
  confidence?: string;
};

export type SearchResponse = {
  query: string;
  answer: SearchAnswer | null;
  results: SearchResult[];
  fallback: boolean;
  notice?: string;
  sourceErrors: BackendSourceError[];
  execution?: BackendExecution;
  metadata?: {
    backend: boolean;
  };
};

export type SearchStreamEvent =
  | {
      type: "sources_loaded";
      query: string;
      sourceErrors: BackendSourceError[];
      execution?: BackendExecution;
    }
  | {
      type: "exact_results" | "proxy_results" | "reranked_results";
      response: SearchResponse;
    }
  | {
      type: "answer_delta";
      query: string;
      delta: string;
    }
  | {
      type: "done";
      response: SearchResponse;
    }
  | {
      type: "error";
      query: string;
      message: string;
      response?: SearchResponse;
    };

export type UploadedSource = {
  id: string;
  name: string;
  kind: string;
  summary: string;
  status: string;
};

export type WorkspaceSource = UploadedSource;

export const demoQueries = [
  "Which customers complained about billing last week?",
  "What launch blocker showed up in Slack?",
  "Where was churn risk marked high?",
];

const demoPassages: DemoPassage[] = [
  {
    id: "demo-1",
    title: "Billing escalation row",
    sourceLabel: "complaints.csv",
    kind: "csv",
    location: "Row 14",
    text: "customer: Acme Health | issue: billing complaint | week: last week | owner: Priya",
    tags: ["billing", "complaint", "last week"],
  },
  {
    id: "demo-2",
    title: "Slack blocker thread",
    sourceLabel: "slack-export.json",
    kind: "json",
    location: "Message 48",
    text: "channel: launch-war-room | message: The launch blocker is a billing webhook timeout for enterprise customers.",
    tags: ["slack", "launch", "blocker"],
  },
  {
    id: "demo-3",
    title: "Risk review note",
    sourceLabel: "customer-risk.md",
    kind: "markdown",
    location: "Section churn-review",
    text: "Beta Retail was marked high churn risk after repeated billing confusion and delayed support follow-up.",
    tags: ["churn", "risk", "billing"],
  },
  {
    id: "demo-4",
    title: "Shipping complaint row",
    sourceLabel: "complaints.csv",
    kind: "csv",
    location: "Row 15",
    text: "customer: Beta Retail | issue: shipping delay | week: this week | owner: Mina",
    tags: ["shipping"],
  },
];

function getStoredSources(): WorkspaceSource[] {
  if (typeof window === "undefined") {
    return [];
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as WorkspaceSource[]) : [];
  } catch {
    return [];
  }
}

function setStoredSources(sources: WorkspaceSource[]) {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sources));
}

function upsertStoredSources(sources: WorkspaceSource[]) {
  const current = getStoredSources();
  const merged = new Map<string, WorkspaceSource>();
  [...sources, ...current].forEach((source) => merged.set(source.id, source));
  setStoredSources(Array.from(merged.values()));
}

function tokenize(value: string): string[] {
  return value
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 1);
}

function scorePassage(query: string, text: string): number {
  const queryTerms = tokenize(query);
  if (queryTerms.length === 0) {
    return 0;
  }
  const content = text.toLowerCase();
  let score = 0;
  for (const term of queryTerms) {
    if (content.includes(term)) {
      score += 1;
    }
  }
  return score / queryTerms.length;
}

function buildMockSearch(query: string, notice?: string): SearchResponse {
  const ranked = demoPassages
    .map((passage) => ({
      ...passage,
    score: scorePassage(query, `${passage.title} ${passage.text} ${passage.tags.join(" ")}`),
    }))
    .filter((passage) => passage.score > 0)
    .sort((left, right) => right.score - left.score)
    .slice(0, 6);

  const results: SearchResult[] = ranked.map((passage) => ({
    id: passage.id,
    title: passage.title,
    kind: passage.kind,
    sourceLabel: passage.sourceLabel,
    location: passage.location,
    snippet: passage.text,
    score: Math.max(0.18, passage.score),
    tags: passage.tags,
    sourceOrigin: "stored",
    primarySpan: {
      text: passage.text,
      snippet: passage.text,
      sourceStart: 0,
      sourceEnd: passage.text.length,
      snippetStart: 0,
      snippetEnd: passage.text.length,
      offsetBasis: "source",
      highlights: [],
      location: { label: passage.location },
    },
    matchedSpans: [],
    channels: ["exact"],
    channelScores: {
      exact: Math.max(0.18, passage.score),
      semantic: 0,
      structure: 0,
    },
  }));

  const citations = results.slice(0, 3).map((result, index) => ({
    label: `[${index + 1}]`,
    sourceId: result.id,
  }));

  const summary =
    results.length > 0
      ? `${results[0].sourceLabel} contains the strongest match for "${query}". ${results
          .slice(0, 2)
          .map((result, index) => `${result.snippet} [${index + 1}]`)
          .join(" ")}`
      : `No seeded demo passages matched "${query}".`;

  return {
    query,
    answer: {
      summary,
      citations,
      confidence: "low",
    },
    results,
    fallback: true,
    notice: notice ?? "Backend unavailable. Showing mock demo data.",
    sourceErrors: [],
    metadata: {
      backend: false,
    },
  };
}

function locationFromMetadata(metadata: Record<string, unknown>): string {
  if (typeof metadata.page_number === "number") {
    return `Page ${metadata.page_number}`;
  }
  if (typeof metadata.row_index === "number") {
    return `Row ${metadata.row_index}`;
  }
  if (typeof metadata.line_number === "number") {
    return `Line ${metadata.line_number}`;
  }
  if (typeof metadata.section === "string") {
    return metadata.section;
  }
  return "Parsed document";
}

function backendSourceToWorkspace(source: BackendSource): WorkspaceSource {
  const parser = source.parser_name || source.source_type || "document";
  const count = source.document_count ?? Number(source.metadata?.document_count ?? 0);
  return {
    id: source.id,
    name: source.name,
    kind: parser,
    summary: `${count || 1} parsed segment${count === 1 ? "" : "s"} via ${parser}.`,
    status: "Ready",
  };
}

function normalizeScores(values: number[]): number[] {
  const max = values.reduce((current, value) => Math.max(current, value), 0);
  if (max <= 0) {
    return values.map(() => 0.5);
  }
  return values.map((value) => Math.max(0.12, Math.min(1, value / max)));
}

function backendSpanToSearchSpan(span?: BackendSearchSpan | null): SearchSpan | null {
  if (!span) {
    return null;
  }
  return {
    text: span.text,
    snippet: span.snippet,
    sourceStart: span.source_start,
    sourceEnd: span.source_end,
    snippetStart: span.snippet_start,
    snippetEnd: span.snippet_end,
    offsetBasis: span.offset_basis ?? "source",
    highlights: span.highlights.map((highlight) => ({
      start: highlight.start,
      end: highlight.end,
      sourceStart: highlight.source_start,
      sourceEnd: highlight.source_end,
      offsetBasis: highlight.offset_basis,
      text: highlight.text,
    })),
    location: span.location,
  };
}

function backendToSearchResponse(payload: BackendSearchPayload): SearchResponse {
  const normalizedScores = normalizeScores(payload.results.map((result) => result.score));
  const results: SearchResult[] = payload.results.map((result, index) => {
    const metadata = result.metadata ?? {};
    const parserName =
      typeof metadata.parser_name === "string" ? metadata.parser_name : undefined;
    const mediaType = typeof metadata.media_type === "string" ? metadata.media_type : undefined;
    const tags = [parserName, mediaType].filter(Boolean) as string[];
    return {
      id:
        result.chunk_id ??
        result.document_id ??
        result.source_id ??
        `${result.source_name.replace(/\s+/g, "-").toLowerCase()}-${index}`,
      title:
        result.title ??
        (typeof metadata.title === "string" ? metadata.title : result.source_name),
      kind: parserName ?? mediaType ?? "document",
      sourceLabel: result.source_name,
      location: locationFromMetadata(metadata),
      snippet: result.primary_span?.snippet ?? result.snippet ?? result.content,
      score: normalizedScores[index] ?? 0.5,
      tags,
      sourceOrigin: result.source_origin ?? "stored",
      primarySpan: backendSpanToSearchSpan(result.primary_span),
      matchedSpans: (result.matched_spans ?? [])
        .map(backendSpanToSearchSpan)
        .filter((span): span is SearchSpan => span !== null),
      channels:
        result.channels ??
        (Array.isArray(metadata.channels) ? metadata.channels.map(String) : []),
      channelScores:
        result.channel_scores ??
        (typeof metadata.channel_scores === "object" && metadata.channel_scores !== null
          ? (metadata.channel_scores as Record<string, number>)
          : {}),
    };
  });

  const citations = results.slice(0, 5).map((result, index) => ({
    label: payload.results[index]?.citation_label ?? `[${index + 1}]`,
    sourceId: result.id,
  }));

  return {
    query: payload.query,
    answer: payload.answer?.markdown
      ? {
          summary: payload.answer.markdown,
          citations,
          confidence: payload.answer.confidence,
        }
      : null,
    results,
    fallback: Boolean(payload.answer_error),
    notice: payload.answer_error ?? undefined,
    sourceErrors: payload.source_errors ?? [],
    execution: payload.execution,
    metadata: {
      backend: true,
    },
  };
}

async function fetchBackend<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function searchDocuments(payload: SearchPayload): Promise<SearchResponse> {
  if (payload.mode === "demo") {
    return buildMockSearch(payload.query, "Demo mode is using a seeded local corpus.");
  }

  try {
    const backend = await fetchBackend<BackendSearchPayload>("/api/v1/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: payload.query,
        top_k: 8,
        answer_mode: payload.answerMode ?? (payload.skipAnswer ? "off" : "off"),
        include_stored_sources: payload.includeStoredSources ?? true,
        raw_sources: payload.rawSources ?? [],
        connector_configs: payload.connectorConfigs ?? [],
      }),
    });
    return backendToSearchResponse(backend);
  } catch {
    return buildMockSearch(payload.query);
  }
}

export async function uploadDocuments(payload: UploadPayload): Promise<{
  notice: string;
  sources: UploadedSource[];
}> {
  const formData = new FormData();
  payload.files?.forEach((file) => formData.append("files", file));
  if (payload.text) {
    formData.append("raw_text", payload.text);
    formData.append("raw_text_name", "Pasted text");
  }

  try {
    const backend = await fetchBackend<{ sources: BackendSource[] }>("/api/v1/upload", {
      method: "POST",
      body: formData,
    });
    const sources = backend.sources.map(backendSourceToWorkspace);
    upsertStoredSources(sources);
    return {
      notice: `Uploaded ${sources.length} source${sources.length === 1 ? "" : "s"}.`,
      sources,
    };
  } catch {
    const mockSources: WorkspaceSource[] = [];
    payload.files?.forEach((file, index) => {
      mockSources.push({
        id: `mock-file-${Date.now()}-${index}`,
        name: file.name,
        kind: "upload",
        summary: "Stored in local mock mode while the backend is offline.",
        status: "Mocked",
      });
    });
    if (payload.text) {
      mockSources.push({
        id: `mock-text-${Date.now()}`,
        name: "Pasted text",
        kind: "paste",
        summary: "Stored in local mock mode while the backend is offline.",
        status: "Mocked",
      });
    }
    upsertStoredSources(mockSources);
    return {
      notice: "Backend unavailable. Saved the source list locally for the UI.",
      sources: mockSources,
    };
  }
}

export async function listSources(): Promise<{ sources: WorkspaceSource[] }> {
  try {
    const backend = await fetchBackend<{ sources: BackendSource[] }>("/api/v1/sources");
    const sources = backend.sources.map(backendSourceToWorkspace);
    upsertStoredSources(sources);
    return { sources };
  } catch {
    return { sources: getStoredSources() };
  }
}

export async function connectConnector(payload: ConnectorPayload): Promise<{
  notice: string;
  sources: UploadedSource[];
}> {
  const source: WorkspaceSource = {
    id: `connector-${payload.connectorId}-${Date.now()}`,
    name: `${payload.connectorId} connector`,
    kind: payload.connectorId,
    summary: "Connector scaffolding only. Real sync is not implemented yet.",
    status: "Scaffolded",
  };
  upsertStoredSources([source]);
  return {
    notice: `${payload.connectorId} was scaffolded locally.`,
    sources: [source],
  };
}

export async function checkHealthz(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/healthz`);
    return response.ok;
  } catch {
    return false;
  }
}

export async function streamSearch(
  payload: SearchPayload,
  onEvent?: (event: SearchStreamEvent) => void,
): Promise<SearchResponse> {
  if (payload.mode === "demo") {
    return buildMockSearch(payload.query, "Demo mode is using a seeded local corpus.");
  }

  try {
    const response = await fetch(`${API_BASE}/api/v1/search/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: payload.query,
        top_k: 8,
        answer_mode: payload.answerMode ?? (payload.skipAnswer ? "off" : "off"),
        include_stored_sources: payload.includeStoredSources ?? true,
        raw_sources: payload.rawSources ?? [],
        connector_configs: payload.connectorConfigs ?? [],
      }),
    });
    if (!response.ok) {
      return buildMockSearch(payload.query);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      return buildMockSearch(payload.query);
    }

    let buffer = "";
    let lastResponse: SearchResponse | null = null;
    let latestSourceErrors: BackendSourceError[] = [];
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const eventPayload = JSON.parse(line.slice(6)) as
              | BackendSearchPayload
              | {
                  event?: string;
                  query?: string;
                  message?: string;
                  delta?: string;
                  results?: BackendSearchPayload["results"];
                  source_errors?: BackendSourceError[];
                  execution?: BackendExecution;
                  answer?: BackendSearchPayload["answer"];
                  answer_error?: string | null;
                };

            if (!("event" in eventPayload) || !eventPayload.event) {
              lastResponse = backendToSearchResponse(eventPayload as BackendSearchPayload);
              continue;
            }

            const event = eventPayload.event;
            if (event === "sources_loaded") {
              latestSourceErrors = eventPayload.source_errors ?? [];
              onEvent?.({
                type: "sources_loaded",
                query: eventPayload.query ?? payload.query,
                sourceErrors: latestSourceErrors,
                execution: eventPayload.execution,
              });
              continue;
            }

            if (event === "answer_delta") {
              onEvent?.({
                type: "answer_delta",
                query: eventPayload.query ?? payload.query,
                delta: eventPayload.delta ?? "",
              });
              continue;
            }

            if (event === "error") {
              onEvent?.({
                type: "error",
                query: eventPayload.query ?? payload.query,
                message: eventPayload.message ?? "Search stream failed",
                response: lastResponse ?? undefined,
              });
              continue;
            }

            const phasePayload: BackendSearchPayload = {
              query: eventPayload.query ?? payload.query,
              results: eventPayload.results ?? [],
              answer: eventPayload.answer ?? null,
              answer_error: eventPayload.answer_error ?? null,
              source_errors: eventPayload.source_errors ?? latestSourceErrors,
              execution: eventPayload.execution,
            };
            const responseValue = backendToSearchResponse(phasePayload);
            lastResponse = responseValue;

            if (event === "done") {
              onEvent?.({ type: "done", response: responseValue });
            } else if (
              event === "exact_results" ||
              event === "proxy_results" ||
              event === "reranked_results"
            ) {
              onEvent?.({ type: event, response: responseValue });
            }
          } catch {
            // Skip invalid JSON lines
          }
        }
      }
    }

    return lastResponse ?? buildMockSearch(payload.query);
  } catch {
    return buildMockSearch(payload.query);
  }
}
