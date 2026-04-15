const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const STORAGE_KEY = "waver.mock.sources";

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
  }>;
  answer?: {
    markdown?: string;
    confidence?: string;
  } | null;
  answer_error?: string | null;
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
  mode?: "standard" | "demo";
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

export type SearchResult = {
  id: string;
  title: string;
  kind: string;
  sourceLabel: string;
  location: string;
  snippet: string;
  score: number;
  tags: string[];
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
  metadata?: {
    backend: boolean;
  };
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
        result.document_id ??
        result.source_id ??
        `${result.source_name.replace(/\s+/g, "-").toLowerCase()}-${index}`,
      title:
        result.title ??
        (typeof metadata.title === "string" ? metadata.title : result.source_name),
      kind: parserName ?? mediaType ?? "document",
      sourceLabel: result.source_name,
      location: locationFromMetadata(metadata),
      snippet: result.snippet ?? result.content,
      score: normalizedScores[index] ?? 0.5,
      tags,
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
        skip_answer: payload.skipAnswer ?? false,
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

