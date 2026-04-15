"use client";

import { useState } from "react";

const connectors = [
  {
    id: "webhook",
    title: "Webhook",
    description: "Push documents directly into Waver from your app.",
  },
  {
    id: "slack",
    title: "Slack",
    description: "Pull channel messages for quick team search.",
  },
  {
    id: "drive",
    title: "Google Drive",
    description: "Seed a workspace with docs and exports.",
  },
];

export function ConnectorPicker({
  onConnect,
  disabled,
}: {
  onConnect: (connectorId: string) => Promise<void> | void;
  disabled?: boolean;
}) {
  const [selected, setSelected] = useState("webhook");

  return (
    <section className="glass-card rounded-[28px] p-5">
      <div>
        <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
          Connectors
        </p>
        <h3 className="font-display mt-2 text-2xl text-white">
          Select a SaaS source
        </h3>
      </div>

      <div className="mt-4 grid gap-3">
        {connectors.map((connector) => {
          const active = selected === connector.id;
          return (
            <button
              key={connector.id}
              type="button"
              onClick={() => setSelected(connector.id)}
              className={[
                "rounded-2xl border p-4 text-left transition",
                active
                  ? "border-cyan-300/30 bg-cyan-400/10"
                  : "border-white/10 bg-black/15 hover:bg-white/10",
              ].join(" ")}
            >
              <div className="flex items-center justify-between gap-3">
                <span className="font-semibold text-white">{connector.title}</span>
                <span className="text-xs uppercase tracking-[0.22em] text-[rgb(var(--muted))]">
                  {active ? "Selected" : "Pick"}
                </span>
              </div>
              <p className="mt-2 text-sm leading-6 text-[rgb(var(--muted))]">
                {connector.description}
              </p>
            </button>
          );
        })}
      </div>

      <button
        type="button"
        disabled={disabled}
        onClick={() => void onConnect(selected)}
        className="mt-4 rounded-2xl bg-[rgb(var(--accent-strong))] px-5 py-3 text-sm font-semibold text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
      >
        Connect source
      </button>
    </section>
  );
}
