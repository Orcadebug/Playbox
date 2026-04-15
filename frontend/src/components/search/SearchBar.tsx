"use client";

import { FormEvent } from "react";

export function SearchBar({
  value,
  onChange,
  onSubmit,
  loading,
  examples,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading?: boolean;
  examples?: string[];
}) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <form onSubmit={handleSubmit} className="glass-card rounded-[28px] p-4 md:p-5">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
        <label className="flex-1">
          <span className="sr-only">Search query</span>
          <input
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder="Ask about files, logs, Slack threads, or pasted text"
            className="w-full rounded-2xl border border-white/10 bg-black/20 px-4 py-4 text-base text-white outline-none transition placeholder:text-[rgb(var(--muted))] focus:border-cyan-300/40 focus:bg-black/25"
          />
        </label>
        <button
          type="submit"
          disabled={loading}
          className="rounded-2xl bg-[rgb(var(--accent))] px-5 py-4 text-sm font-semibold text-slate-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {examples && examples.length > 0 ? (
        <div className="mt-4 flex flex-wrap gap-2">
          {examples.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => onChange(example)}
              className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs text-[rgb(var(--muted))] transition hover:border-white/20 hover:bg-white/10 hover:text-white"
            >
              {example}
            </button>
          ))}
        </div>
      ) : null}
    </form>
  );
}
