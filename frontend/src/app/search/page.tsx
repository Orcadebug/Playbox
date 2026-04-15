import { AppShell } from "@/components/layout/AppShell";
import { SearchWorkspace } from "@/components/search/SearchWorkspace";

export default function SearchPage() {
  return (
    <AppShell
      eyebrow="Search"
      title="Query any messy corpus"
      description="Type a question and get ranked passages plus a synthesized answer with citations."
    >
      <SearchWorkspace
        initialQuery="Which customers complained about billing last week?"
        variant="standard"
      />
    </AppShell>
  );
}

