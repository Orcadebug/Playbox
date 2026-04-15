"use client";

import { ChangeEvent, DragEvent, useState } from "react";

export function FileDropzone({
  onFiles,
  disabled,
}: {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
}) {
  const [isDragging, setIsDragging] = useState(false);

  function handleFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) {
      return;
    }

    onFiles(Array.from(fileList));
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    handleFiles(event.dataTransfer.files);
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    handleFiles(event.target.files);
    event.target.value = "";
  }

  return (
    <label
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={[
        "glass-card block cursor-pointer rounded-[28px] border-2 border-dashed p-6 transition",
        isDragging ? "border-cyan-300 bg-cyan-400/10" : "border-white/10",
        disabled ? "pointer-events-none opacity-70" : "",
      ].join(" ")}
    >
      <input
        type="file"
        className="sr-only"
        multiple
        onChange={handleChange}
        accept=".csv,.json,.txt,.md,.pdf,.html,.htm,.log"
        disabled={disabled}
      />
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
            File dropzone
          </p>
          <h3 className="font-display mt-2 text-2xl text-white">
            Drop documents to ingest them instantly
          </h3>
          <p className="mt-2 max-w-2xl text-sm leading-7 text-[rgb(var(--muted))]">
            CSV, JSON, TXT, MD, HTML, log files, and PDFs are all supported.
            Files are parsed locally in the frontend scaffold if the backend is
            unavailable.
          </p>
        </div>
        <div className="rounded-2xl border border-white/10 bg-black/15 px-4 py-3 text-sm text-[rgb(var(--muted))]">
          Click or drag and drop
        </div>
      </div>
    </label>
  );
}
