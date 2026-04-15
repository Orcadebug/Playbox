import { AppShell } from "@/components/layout/AppShell";
import { SearchWorkspace } from "@/components/search/SearchWorkspace";

export default function DemoPage() {
  return (
    <AppShell
      eyebrow="Demo"
      title="Try the product with preloaded sample data"
      description="This demo stays functional without a backend so you can validate the flow before wiring real data."
    >
      <SearchWorkspace
        initialQuery="Which customers complained about billing last week?"
        variant="demo"
      />
    </AppShell>
  );
}

