import { AppShell } from "@/components/layout/AppShell";
import { UploadWorkspace } from "@/components/upload/UploadWorkspace";

export default function UploadPage() {
  return (
    <AppShell
      eyebrow="Saved sources"
      title="Manage stored workspace content"
      description="Save files or pasted text when you want them available across searches. Raw and connector search can run directly from the retrieval page."
    >
      <UploadWorkspace />
    </AppShell>
  );
}
