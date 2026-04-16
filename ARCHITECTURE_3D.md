# Waver 3D Architecture Map

## Layer 1: User Interaction (Frontend)

```
┌─────────────────────────────────────────────────────────────────┐
│                    BROWSER / USER INTERFACE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐    ┌──────────────────┐  ┌─────────────┐   │
│  │   Upload Page   │    │   Search Page    │  │  Demo Page  │   │
│  │  /upload        │    │   /search        │  │  /demo      │   │
│  │                 │    │                  │  │             │   │
│  │ • FileDropzone  │    │ • SearchBar      │  │ • Examples  │   │
│  │ • PasteBox      │    │ • ResultList     │  │ • Tutorials │   │
│  │ • Connector     │    │ • ResultCard     │  │             │   │
│  │   Picker        │    │ • AnswerCard     │  │             │   │
│  └────────┬────────┘    └────────┬─────────┘  └─────────────┘   │
│           │                      │                                │
│           │ FormData             │ Query + Headers                │
│           └──────────────┬───────┘                                │
│                          │                                        │
│                    ┌─────▼──────┐                                │
│                    │ API Client  │                                │
│                    │ Typed Fetch │                                │
│                    │ Wrapper     │                                │
│                    └─────┬───────┘                                │
└─────────────────────────┼──────────────────────────────────────┘
                          │
                    HTTP │ HTTPS
                          │
```

---

## Layer 2: API Gateway (Backend Entry)

```
┌─────────────────────────────────────────────────────────────────┐
│              FASTAPI APPLICATION LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              MIDDLEWARE (CORS, Logging)                  │   │
│  └────────────────────┬─────────────────────────────────────┘   │
│                       │                                           │
│        ┌──────────────┼──────────────┬──────────────┐            │
│        │              │              │              │            │
│  ┌─────▼─────┐  ┌────▼──────┐  ┌───▼─────┐  ┌─────▼────┐       │
│  │  Upload   │  │  Search   │  │ Sources │  │ Health   │       │
│  │ Endpoint  │  │ Endpoints │  │ Endpoint│  │ Check    │       │
│  │ POST/     │  │ POST /    │  │ GET /   │  │ GET /    │       │
│  │ upload    │  │ search    │  │ sources │  │ healthz  │       │
│  │           │  │ POST /    │  │ DELETE/ │  │          │       │
│  │ (FormData)│  │ search/   │  │ sources │  │          │       │
│  │           │  │ stream    │  │ /{id}   │  │          │       │
│  │ workspace │  │           │  │         │  │          │       │
│  │ _id       │  │ workspace │  │ work-   │  │          │       │
│  │ file      │  │ _id       │  │ space   │  │          │       │
│  │           │  │ query     │  │ _id     │  │          │       │
│  └─────┬─────┘  └────┬──────┘  └───┬─────┘  └─────┬────┘       │
│        │             │             │              │             │
│        └─────────────┼─────────────┼──────────────┘             │
│                      │             │                            │
│        ┌─────────────▼─────────────▼──┐                        │
│        │   SERVICE LAYER               │                        │
│        │ (Business Logic)              │                        │
│        │                               │                        │
│        │  • SearchService              │                        │
│        │  • SourceService              │                        │
│        └───────────┬───────────────────┘                        │
└──────────────────┼─────────────────────────────────────────────┘
                   │
```

---

## Layer 3: Document Processing Pipeline (The Brain)

```
┌─────────────────────────────────────────────────────────────────┐
│           DOCUMENT UPLOAD & PARSING FLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  FILE INPUT                                                      │
│      │                                                            │
│      ├─ PDF, CSV, JSON, HTML, Markdown, Plaintext               │
│      │                                                            │
│      ▼                                                            │
│  ┌─────────────────────────────────────────┐                    │
│  │  MIME TYPE DETECTOR                     │                    │
│  │  (detector.py)                          │                    │
│  │  Analyzes file extension + magic bytes  │                    │
│  └──────────────────┬──────────────────────┘                    │
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────┐                    │
│  │  PARSER REGISTRY                        │                    │
│  │  (registry.py)                          │                    │
│  │  Maps MIME type → Parser class          │                    │
│  │                                          │                    │
│  │  • pdf.py (pypdf)                       │                    │
│  │  • html.py (BeautifulSoup)              │                    │
│  │  • csv_parser.py (pandas)               │                    │
│  │  • json_parser.py                       │                    │
│  │  • markdown.py                          │                    │
│  │  • plaintext.py                         │                    │
│  └──────────────────┬──────────────────────┘                    │
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────┐                    │
│  │  PARSER (base.py interface)             │                    │
│  │  Extracts text from specific format     │                    │
│  │                                          │                    │
│  │  Returns: List[Document]                │                    │
│  │  Each doc = {text, metadata}            │                    │
│  └──────────────────┬──────────────────────┘                    │
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────┐                    │
│  │  CHUNKER (chunker.py)                   │                    │
│  │  Splits long documents into chunks      │                    │
│  │  Overlapping windows for context        │                    │
│  │                                          │                    │
│  │  Input:  Long document (e.g. 100KB)     │                    │
│  │  Output: 100 chunks (512 tokens each)   │                    │
│  └──────────────────┬──────────────────────┘                    │
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────┐                    │
│  │  TOKENIZER (tokenizer.py)               │                    │
│  │  Breaks chunks into tokens              │                    │
│  │  Tracks token counts + boundaries       │                    │
│  │                                          │                    │
│  │  For BM25 indexing                      │                    │
│  └──────────────────┬──────────────────────┘                    │
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────┐                    │
│  │  DATABASE STORE                         │                    │
│  │  (SQLAlchemy async ORM)                 │                    │
│  │                                          │                    │
│  │  SOURCE table:                          │                    │
│  │  ├─ id, workspace_id                    │                    │
│  │  ├─ filename, file_type                 │                    │
│  │  ├─ upload_time                         │                    │
│  │  └─ metadata                            │                    │
│  │                                          │                    │
│  │  DOCUMENT table:                        │                    │
│  │  ├─ id, workspace_id, source_id         │                    │
│  │  ├─ text (chunk content)                │                    │
│  │  ├─ tokens (tokenized text)             │                    │
│  │  └─ metadata (page_no, position)        │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 4: Search & Retrieval Pipeline (The Muscle)

```
┌─────────────────────────────────────────────────────────────────┐
│              SEARCH REQUEST PROCESSING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  USER QUERY + WORKSPACE_ID                                       │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────────────────────────────────┐                   │
│  │  RETRIEVAL PIPELINE                      │                   │
│  │  (pipeline.py - orchestrator)            │                   │
│  │                                           │                   │
│  │  Coordinates: BM25 → Rerank → Filter    │                   │
│  └──────────────────┬───────────────────────┘                   │
│                     │                                            │
│        ┌────────────┴────────────┐                              │
│        │                         │                              │
│        ▼                         ▼                              │
│  ┌──────────────────┐   ┌──────────────────────────┐           │
│  │  BM25 RETRIEVAL  │   │  BM25_CACHE              │           │
│  │  (bm25.py)       │   │  (bm25_cache.py)         │           │
│  │                  │   │                          │           │
│  │ Ranking scores   │   │ Caches BM25 index        │           │
│  │ all chunks by    │   │ Avoids recomputing       │           │
│  │ text relevance   │   │                          │           │
│  │                  │   │ Persists between         │           │
│  │ Input: Query     │   │ requests                 │           │
│  │ Output: Ranked   │   │                          │           │
│  │ chunks with      │   │ Key: workspace_id +      │           │
│  │ scores (top 50)  │   │ source_id                │           │
│  └────────┬─────────┘   └──────────────────────────┘           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────────────────────────────┐                  │
│  │  CROSS-ENCODER RERANKER                  │                  │
│  │  (reranker.py - ONNX inference)          │                  │
│  │                                           │                  │
│  │ Takes top 50 chunks                      │                  │
│  │ Runs each through cross-encoder model    │                  │
│  │ Gets more accurate relevance scores      │                  │
│  │                                           │                  │
│  │ Model: e.g. ms-marco-MiniLM-L-6-v2       │                  │
│  │ Input: (query, chunk) pairs              │                  │
│  │ Output: Reranked top-K chunks            │                  │
│  │                                           │                  │
│  │ MUCH SLOWER but more accurate            │                  │
│  │ (LLM-quality relevance)                  │                  │
│  └───────────┬────────────────────────────┬─┘                  │
│              │                            │                    │
│              ▼                            ▼                    │
│  ┌──────────────────────────┐  ┌────────────────────────┐    │
│  │  FILTER BY WORKSPACE     │  │ TRIM TO TOP-K          │    │
│  │  Ensure workspace_id     │  │ Keep best N results    │    │
│  │  matches                 │  │ (e.g. top 5-10)        │    │
│  └──────────────────────────┘  └────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 5: Answer Generation (Optional LLM Layer)

```
┌─────────────────────────────────────────────────────────────────┐
│           LLM ANSWER GENERATION (Optional)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  RETRIEVED CHUNKS + QUERY                                        │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────────────────────────────┐                       │
│  │  ANSWER GENERATOR                    │                       │
│  │  (generator.py)                      │                       │
│  │                                       │                       │
│  │  Constructs prompt:                  │                       │
│  │  ├─ System prompt (role)             │                       │
│  │  ├─ Retrieved chunks (context)       │                       │
│  │  ├─ User query (question)            │                       │
│  │  └─ Response instructions            │                       │
│  │                                       │                       │
│  │  Prompts in: prompts.py              │                       │
│  └──────────────────┬────────────────────┘                       │
│                     │                                            │
│                     ▼                                            │
│  ┌──────────────────────────────────────┐                       │
│  │  LLM PROVIDER CALL                    │                       │
│  │  (OpenRouter or Anthropic)            │                       │
│  │                                       │                       │
│  │  • OpenRouter API (model router)      │                       │
│  │  • Anthropic API (direct)             │                       │
│  │  • Streaming: Collect tokens as they │                       │
│  │    arrive                             │                       │
│  │                                       │                       │
│  │  Returns: Streamed answer text        │                       │
│  └──────────────────┬────────────────────┘                       │
│                     │                                            │
│                     ▼                                            │
│  ┌──────────────────────────────────────┐                       │
│  │  CITATION MAPPING                    │                       │
│  │  (citations.py)                      │                       │
│  │                                       │                       │
│  │  Analyzes answer spans (sentences)   │                       │
│  │  Maps each span to source documents  │                       │
│  │  using fuzzy matching + BM25         │                       │
│  │                                       │                       │
│  │  Output: {span: [source_ids]}        │                       │
│  │                                       │                       │
│  │  Sent to frontend for attribution    │                       │
│  └──────────────────────────────────────┘                       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 6: Streaming Response (SSE)

```
┌─────────────────────────────────────────────────────────────────┐
│         STREAMING RESPONSE (Server-Sent Events)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Backend (FastAPI)                     Frontend (React)          │
│                                                                   │
│  ┌──────────────────────┐              ┌────────────────────┐   │
│  │ Search Request      │              │ EventSource Listen │   │
│  │ /search/stream      │──HTTP GET───>│ Real-time updates  │   │
│  │                     │              │                    │   │
│  │ Returns: SSE Stream │              └────────┬───────────┘   │
│  │ Content-Type:       │                       │               │
│  │ text/event-stream   │                       │               │
│  └────────┬────────────┘                       │               │
│           │                                    │               │
│           ▼                                    │               │
│  ┌──────────────────────┐                     │               │
│  │ Event 1: "result"    │                     │               │
│  │ {chunk, score}       │──SSE message───────>│ Update UI      │
│  │                      │                     │ Add to list    │
│  └────────┬─────────────┘                     │               │
│           │                                    │               │
│           ▼                                    │               │
│  ┌──────────────────────┐                     │               │
│  │ Event 2: "result"    │                     │               │
│  │ {chunk, score}       │──SSE message───────>│ Update UI      │
│  │                      │                     │               │
│  └────────┬─────────────┘                     │               │
│           │                                    │               │
│           ▼                                    │               │
│  ┌──────────────────────┐                     │               │
│  │ Event 3: "answer"    │                     │               │
│  │ {text_chunk}         │──SSE message───────>│ Stream LLM     │
│  │ (if LLM enabled)     │                     │ answer        │
│  │                      │                     │               │
│  └────────┬─────────────┘                     │               │
│           │                                    │               │
│           ▼                                    │               │
│  ┌──────────────────────┐                     │               │
│  │ Event N: "done"      │                     │               │
│  │ (stream ends)        │──SSE message───────>│ Mark complete │
│  │                      │                     │               │
│  └──────────────────────┘                     │               │
│                                                 │               │
│                                            ┌────▼────────────┐  │
│                                            │ Final UI State: │  │
│                                            │ • All results   │  │
│                                            │ • Full answer   │  │
│                                            │ • Citations     │  │
│                                            └─────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 7: Data Models & Storage

```
┌─────────────────────────────────────────────────────────────────┐
│                   DATABASE SCHEMA                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  POSTGRES (Neon) / SQLite (Local Dev)                            │
│                                                                   │
│  ┌─────────────────────────────┐                                │
│  │       SOURCE TABLE          │                                │
│  ├─────────────────────────────┤                                │
│  │ id (PK)                     │  ◄──┐                          │
│  │ workspace_id (FK)           │     │                          │
│  │ filename                    │     │  One-to-Many             │
│  │ file_type (MIME)            │     │                          │
│  │ uploaded_at                 │     │                          │
│  │ metadata (JSON)             │     │                          │
│  └─────────────────────────────┘     │                          │
│                                       │                          │
│  ┌─────────────────────────────┐     │                          │
│  │     DOCUMENT TABLE          │─────┘                          │
│  ├─────────────────────────────┤                                │
│  │ id (PK)                     │                                │
│  │ workspace_id (FK)           │                                │
│  │ source_id (FK) ─────────────┘                                │
│  │ text (chunk content)        │  ◄── Indexed for BM25          │
│  │ tokens (tokenized)          │      Reranked on search        │
│  │ metadata (page, pos)        │                                │
│  │ created_at                  │                                │
│  └─────────────────────────────┘                                │
│                                                                   │
│  INDEXES:                                                        │
│  ├─ workspace_id (for isolation)                                │
│  ├─ source_id (for deletion cascade)                            │
│  └─ text (FTS if available)                                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 8: Integrations (Connectors)

```
┌─────────────────────────────────────────────────────────────────┐
│              CONNECTORS (Slack, Webhook)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         BaseConnector (Abstract)                        │   │
│  │  (base.py)                                              │   │
│  │                                                          │   │
│  │  Methods:                                               │   │
│  │  • authenticate()  → OAuth/Token exchange               │   │
│  │  • fetch_data()    → Get content from service           │   │
│  │  • listen()        → Webhook receiver (for events)      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                  ▲                          ▲                   │
│                  │ implements               │ implements        │
│        ┌─────────┴────────┐        ┌───────┴──────────┐        │
│        │                  │        │                  │        │
│  ┌─────▼──────────┐  ┌────▼───────▼──────────┐       │        │
│  │  SlackConnector│  │  WebhookConnector     │       │        │
│  │  (slack.py)    │  │  (webhook.py)         │       │        │
│  │                │  │                       │       │        │
│  │ • OAuth flow   │  │ • Receives POST       │       │        │
│  │ • List files   │  │   events              │       │        │
│  │ • Download     │  │ • Parses payload      │       │        │
│  │ • Parse        │  │ • Routes to parser    │       │        │
│  └─────────────────  └───────────────────────┘       │        │
│                                                                   │
│  REGISTRY (registry.py):                                         │
│  Maps service name → Connector class                            │
│  "slack" → SlackConnector                                       │
│  "webhook" → WebhookConnector                                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete Request-Response Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│              UPLOAD → PROCESS → SEARCH CYCLE                     │
└─────────────────────────────────────────────────────────────────┘

STEP 1: UPLOAD
┌────────┐
│ Browser│
└───┬────┘
    │ POST /upload (FormData: file, workspace_id)
    ▼
┌─────────────────────────────────────────┐
│ FastAPI /upload endpoint                │
│ • Receive file                          │
│ • Validate workspace_id                 │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ DETECTOR → REGISTRY → PARSER             │
│ File type detected                       │
│ Parser selected (pdf/csv/html/etc)       │
│ Content extracted to text                │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ CHUNKER → TOKENIZER                     │
│ Long text → 512-token chunks            │
│ Overlapping windows (256 overlap)       │
│ Each chunk tokenized for BM25           │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ DATABASE INSERT                         │
│ CREATE Source (filename, metadata)      │
│ CREATE Documents (chunks, tokens)       │
│ Commit transaction                      │
└────────┬────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│ Browser: Uploaded. Files now searchable.   │
└────────────────────────────────────────────┘


STEP 2: SEARCH
┌────────┐
│ Browser│
└───┬────┘
    │ POST /search/stream (query, workspace_id)
    │ Browser opens EventSource connection
    ▼
┌─────────────────────────────────────────┐
│ FastAPI /search/stream endpoint         │
│ • Receive query, workspace_id           │
│ • Start async generator (yields events) │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ BM25 RETRIEVAL (retrieval/pipeline.py)  │
│ • Load BM25 cache (or rebuild)          │
│ • Score ALL documents against query     │
│ • Get top 50 by BM25 score              │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ RERANKER (ONNX inference)               │
│ • Take top 50 chunks                    │
│ • Run cross-encoder on each             │
│ • Re-rank by semantic similarity        │
│ • Keep top 5-10                         │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ STREAM RESULTS (SSE)                    │
│ For each result:                        │
│ • Emit "result" event → Browser         │
│ • Include: text, score, source_id      │
│ (Browser updates UI in real-time)       │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ OPTIONAL: GENERATE ANSWER (generator.py)│
│ IF answer_enabled:                      │
│ • Build prompt (context + query)        │
│ • Call LLM (OpenRouter/Anthropic)       │
│ • Stream response token-by-token        │
│ • Emit "answer" events → Browser        │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ OPTIONAL: CITATION MAPPING (citations)  │
│ • Parse answer spans (sentences)        │
│ • Match to retrieved documents          │
│ • Emit "citations" event → Browser      │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ EMIT "done" EVENT (stream closes)       │
└────────┬────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│ Browser:                                      │
│ • All results displayed (progressive)        │
│ • Answer fully streamed                      │
│ • Citations shown                            │
│ • User can interact with results             │
└───────────────────────────────────────────────┘
```

---

## Architecture Patterns

### 1. **Request → Service → Domain** (Separation of Concerns)

```
API Endpoint  →  Service Layer  →  Domain Modules
  (routes)      (business logic)   (core logic)
    ↓               ↓                  ↓
upload.py    →  SourceService  →  Parsers
search.py    →  SearchService  →  Retrieval
sources.py   →  SourceService  →  ORM Models
```

### 2. **Registry Pattern** (Extensibility)

```
MIME Type or Service Name  →  Registry Lookup  →  Plugin Class
        "application/pdf"         →                 PDFParser
        "text/csv"                →                 CSVParser
        "slack"                   →                 SlackConnector
```

### 3. **Pipeline Pattern** (Retrieval is a Pipeline)

```
Input (Query)  →  Stage 1 (BM25)  →  Stage 2 (Reranker)  →  Output (Top-K)
   Query            Rank 50              Rerank 50            Top 5-10
   ↓                ↓                    ↓                     ↓
   └─────────────── Pipeline.run() ──────────────────────────┘
                    (orchestrator)
```

### 4. **Workspace Isolation** (Multi-tenancy)

```
Every Request  →  workspace_id  →  Query Filter
                      ↓              ↓
               Used in:         Documents
               • Upload         Sources
               • Search         Results
               • Delete
```

### 5. **Streaming over Polling** (Real-time UX)

```
Browser  ←─────────────── FastAPI (SSE)
  │                           │
  │ POST /search/stream       │ Start Event Stream
  │──────────────────────────→│ (infinite response)
  │                           │
  │ ← Event: "result" ────────│
  │ ← Event: "result" ────────│
  │ ← Event: "answer" ────────│
  │ ← Event: "done" ──────────│ (close connection)
```

---

## Key Data Flows

### Upload Data Flow
```
File → Detector → Parser → Chunker → Tokenizer → DB Insert
(Raw)  (MIME)   (Text)    (Chunks)  (Tokens)    (Persistent)
```

### Search Data Flow
```
Query → BM25 Index → Rank:50 → Reranker → Top-K → LLM (opt) → Citations → SSE Stream
(Text)  (Vectorized) (Scored)  (Refined)  (Final) (Answer)   (Sources)  (Browser)
```

### Database Flow
```
Source (file metadata) ←─┐
                         ├─ workspace_id
Document (chunks)      ←─┘
                         └─ source_id (foreign key)
                         └─ tokens (for BM25)
```

---

## Performance Optimization Points

| Component | Optimization | Benefit |
|-----------|--------------|---------|
| **BM25 Cache** | Persist index between requests | Avoid recomputation |
| **Reranker** | ONNX inference (vs HuggingFace) | 10x faster |
| **Chunking** | Pre-split on upload | No per-search overhead |
| **Streaming (SSE)** | Progressive results | Perceived speed (UX) |
| **Async DB** | SQLAlchemy async | Non-blocking I/O |
| **Workspace Filter** | Index on workspace_id | Fast multi-tenancy |

---

## Deployment Topology

```
┌──────────────────────────┐
│      Vercel (CDN)        │  ← Frontend (Next.js)
│      Fast, global        │    JavaScript bundles
└────────────┬─────────────┘
             │
      HTTPS  │
             │
┌────────────▼──────────────┐
│      Fly.io (FastAPI)     │  ← Backend API
│      Distributed compute  │    Python processes
│      512MB shared-cpu     │
└────────────┬──────────────┘
             │
      TCP   │
             │
┌────────────▼──────────────┐
│    Neon (Postgres)        │  ← Persistent Data
│    Managed database       │    Async pooling
└───────────────────────────┘
```

---

## Summary: How It Works

1. **Upload**: File → (detect MIME) → (select parser) → extract text → chunk → tokenize → store chunks + metadata in DB
2. **Search**: Query → (BM25 rank all docs) → (reranker: top-50 → top-K) → stream results to browser
3. **Answer** (optional): Use top-K chunks as context → prompt LLM → stream answer → extract citations
4. **Streaming**: Server-Sent Events keep browser updated progressively (no page reloads)
5. **Isolation**: workspace_id filters all queries (multi-tenant safe)
6. **Indexing**: BM25 cache persists; recomputed only on new uploads

This is a **retrieval-first** architecture: prioritize finding relevant chunks, let LLM refine if needed.
