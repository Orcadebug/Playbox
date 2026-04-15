"use client";

import { useEffect, useState } from "react";
import {
  connectConnector,
  listSources,
  uploadDocuments,
  type UploadedSource,
  type WorkspaceSource,
} from "@/lib/api-client";
import { ConnectorPicker } from "./ConnectorPicker";
import { FileDropzone } from "./FileDropzone";
import { PasteBox } from "./PasteBox";

export function UploadWorkspace() {
  const [sources, setSources] = useState<WorkspaceSource[]>([]);
  const [status, setStatus] = useState<string>("Idle");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void refreshSources();
  }, []);

  async function refreshSources() {
    const data = await listSources();
    setSources(data.sources);
  }

  async function handleFiles(files: File[]) {
    setBusy(true);
    try {
      setStatus(`Uploading ${files.length} file${files.length === 1 ? "" : "s"}...`);
      const response = await uploadDocuments({ files });
      setStatus(response.notice);
      mergeUploaded(response.sources);
    } finally {
      setBusy(false);
    }
  }

  async function handlePaste(value: string) {
    setBusy(true);
    try {
      setStatus("Sending pasted content...");
      const response = await uploadDocuments({ text: value });
      setStatus(response.notice);
      mergeUploaded(response.sources);
    } finally {
      setBusy(false);
    }
  }

  async function handleConnect(connectorId: string) {
    setBusy(true);
    try {
      setStatus(`Connecting ${connectorId}...`);
      const response = await connectConnector({ connectorId });
      setStatus(response.notice);
      mergeUploaded(response.sources);
    } finally {
      setBusy(false);
    }
  }

  function mergeUploaded(nextSources: UploadedSource[]) {
    setSources((current) => {
      const merged = [...nextSources, ...current];
      const unique = new Map<string, WorkspaceSource>();
      for (const source of merged) {
        unique.set(source.id, source);
      }
      return Array.from(unique.values());
    });
    void refreshSources();
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <FileDropzone onFiles={(files) => void handleFiles(files)} disabled={busy} />
        <ConnectorPicker onConnect={handleConnect} disabled={busy} />
      </div>

      <PasteBox onSubmit={handlePaste} disabled={busy} />

      <section className="glass-card rounded-[28px] p-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
              Sources
            </p>
            <h3 className="font-display mt-2 text-2xl text-white">
              Current workspace content
            </h3>
          </div>
          <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-[rgb(var(--muted))]">
            {status}
          </span>
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {sources.map((source) => (
            <article
              key={source.id}
              className="rounded-2xl border border-white/10 bg-black/15 p-4"
            >
              <div className="flex items-center justify-between gap-3">
                <h4 className="font-semibold text-white">{source.name}</h4>
                <span className="text-xs uppercase tracking-[0.22em] text-[rgb(var(--muted))]">
                  {source.kind}
                </span>
              </div>
              <p className="mt-2 text-sm leading-6 text-[rgb(var(--muted))]">
                {source.summary}
              </p>
              <div className="mt-3 text-xs uppercase tracking-[0.22em] text-[rgb(var(--muted))]">
                {source.status}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
