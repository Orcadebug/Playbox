import { AppShell } from "@/components/layout/AppShell";
import { UploadWorkspace } from "@/components/upload/UploadWorkspace";

export default function UploadPage() {
  return (
    <AppShell
      eyebrow="Upload"
      title="Ingest files, text, and connectors"
      description="Drop documents, paste raw content, or connect a SaaS source. The UI stays useful even when the backend is offline."
    >
      <UploadWorkspace />
    </AppShell>
  );
}

