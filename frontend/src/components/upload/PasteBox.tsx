"use client";

import { FormEvent, useState } from "react";

export function PasteBox({
  onSubmit,
  disabled,
}: {
  onSubmit: (value: string) => void;
  disabled?: boolean;
}) {
  const [value, setValue] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!value.trim()) {
      return;
    }
    onSubmit(value.trim());
    setValue("");
  }

  return (
    <form onSubmit={handleSubmit} className="glass-card rounded-[28px] p-5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
            Paste box
          </p>
          <h3 className="font-display mt-2 text-2xl text-white">
            Paste text, JSON, or logs
          </h3>
        </div>
      </div>
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder="Paste raw text here, then send it to the ingestion pipeline."
        className="mt-4 min-h-40 w-full rounded-2xl border border-white/10 bg-black/20 px-4 py-4 text-sm leading-7 text-white outline-none transition placeholder:text-[rgb(var(--muted))] focus:border-cyan-300/40 focus:bg-black/25"
        disabled={disabled}
      />
      <div className="mt-4 flex items-center justify-between gap-3">
        <p className="text-sm text-[rgb(var(--muted))]">
          The upload endpoint is mocked automatically if the backend is down.
        </p>
        <button
          type="submit"
          disabled={disabled || !value.trim()}
          className="rounded-2xl bg-[rgb(var(--accent))] px-5 py-3 text-sm font-semibold text-slate-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
        >
          Ingest text
        </button>
      </div>
    </form>
  );
}
