import { AppShell } from "@/components/layout/AppShell";
import { SearchWorkspace } from "@/components/search/SearchWorkspace";

export default function SearchPage() {
  return (
    <AppShell
      eyebrow="Retrieval"
      title="Find the exact span that matters"
      description="Search saved workspace content, raw text, and transient connector payloads without making upload the required first step."
    >
      <SearchWorkspace
        initialQuery="Which customers complained about billing last week?"
        variant="standard"
      />
    </AppShell>
  );
}
