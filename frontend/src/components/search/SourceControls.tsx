"use client";

export function SourceControls({
  includeStoredSources,
  onIncludeStoredSourcesChange,
  rawSourceName,
  onRawSourceNameChange,
  rawSourceContent,
  onRawSourceContentChange,
  webhookEnabled,
  onWebhookEnabledChange,
  webhookContent,
  onWebhookContentChange,
  disabled,
}: {
  includeStoredSources: boolean;
  onIncludeStoredSourcesChange: (value: boolean) => void;
  rawSourceName: string;
  onRawSourceNameChange: (value: string) => void;
  rawSourceContent: string;
  onRawSourceContentChange: (value: string) => void;
  webhookEnabled: boolean;
  onWebhookEnabledChange: (value: boolean) => void;
  webhookContent: string;
  onWebhookContentChange: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <section className="glass-card rounded-[28px] p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
            Sources
          </p>
          <h2 className="font-display mt-2 text-2xl text-white">
            Search live or saved content
          </h2>
        </div>
        <label className="inline-flex items-center gap-3 rounded-2xl border border-white/10 bg-black/15 px-4 py-3 text-sm text-[rgb(var(--text))]">
          <input
            type="checkbox"
            checked={includeStoredSources}
            disabled={disabled}
            onChange={(event) => onIncludeStoredSourcesChange(event.target.checked)}
            className="h-4 w-4 accent-[rgb(var(--accent))]"
          />
          Saved workspace
        </label>
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        <div className="rounded-2xl border border-white/10 bg-black/15 p-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <label className="flex-1">
              <span className="text-xs uppercase tracking-[0.22em] text-[rgb(var(--muted))]">
                Raw source
              </span>
              <input
                value={rawSourceName}
                disabled={disabled}
                onChange={(event) => onRawSourceNameChange(event.target.value)}
                className="mt-2 w-full rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-sm text-white outline-none transition placeholder:text-[rgb(var(--muted))] focus:border-cyan-300/40"
              />
            </label>
          </div>
          <textarea
            value={rawSourceContent}
            disabled={disabled}
            onChange={(event) => onRawSourceContentChange(event.target.value)}
            placeholder="Paste raw notes, logs, rows, or extracted text"
            rows={6}
            className="mt-3 min-h-36 w-full resize-y rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-sm leading-6 text-white outline-none transition placeholder:text-[rgb(var(--muted))] focus:border-cyan-300/40"
          />
        </div>

        <div className="rounded-2xl border border-white/10 bg-black/15 p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.22em] text-[rgb(var(--muted))]">
                Webhook connector
              </p>
              <h3 className="mt-2 font-semibold text-white">
                Transient connector payload
              </h3>
            </div>
            <label className="inline-flex items-center gap-2 text-sm text-[rgb(var(--text))]">
              <input
                type="checkbox"
                checked={webhookEnabled}
                disabled={disabled}
                onChange={(event) => onWebhookEnabledChange(event.target.checked)}
                className="h-4 w-4 accent-[rgb(var(--accent))]"
              />
              Use
            </label>
          </div>
          <textarea
            value={webhookContent}
            disabled={disabled || !webhookEnabled}
            onChange={(event) => onWebhookContentChange(event.target.value)}
            placeholder="Paste connector document text"
            rows={6}
            className="mt-3 min-h-36 w-full resize-y rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-sm leading-6 text-white outline-none transition placeholder:text-[rgb(var(--muted))] focus:border-cyan-300/40 disabled:opacity-50"
          />
        </div>
      </div>
    </section>
  );
}
