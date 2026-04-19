import Link from "next/link";
import { Header } from "@/components/layout/Header";

const highlights = [
  {
    title: "No indexing ceremony",
    body: "Search raw text, connectors, or saved sources immediately.",
  },
  {
    title: "Span-first retrieval",
    body: "Ranked results point to the exact passage and character range that matched.",
  },
  {
    title: "Answers stay optional",
    body: "Generate synthesis only after the source spans are already visible.",
  },
];

const examples = [
  "Which customers complained about billing last week?",
  "Show the Slack thread about the launch blocker.",
  "Find the row where churn risk was marked high.",
];

export default function LandingPage() {
  return (
    <main className="min-h-screen px-4 py-5 md:px-6">
      <Header mode="landing" />
      <section className="mx-auto mt-6 grid max-w-7xl gap-6 lg:grid-cols-[1.25fr_0.75fr]">
        <div className="glass-card relative overflow-hidden rounded-[28px] p-8 md:p-12">
          <div className="grid-noise absolute inset-0 opacity-25" />
          <div className="relative max-w-3xl">
            <span className="inline-flex rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-semibold uppercase tracking-[0.28em] text-[rgb(var(--accent))]">
              Search & understand messy data
            </span>
            <h1 className="font-display mt-6 max-w-2xl text-5xl font-bold tracking-tight text-white md:text-7xl">
              Waver finds the exact span in scattered data.
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-8 text-[rgb(var(--muted))]">
              Search CSVs, PDFs, logs, JSON, pasted text, or connector payloads.
              Waver parses on demand and ranks the source spans without a
              persistent indexing pipeline.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link
                href="/demo"
                className="rounded-full bg-[rgb(var(--accent))] px-5 py-3 text-sm font-semibold text-slate-950 transition hover:brightness-110"
              >
                Try the demo
              </Link>
              <Link
                href="/search"
                className="rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-white transition hover:bg-white/10"
              >
                Open search
              </Link>
            </div>

            <div className="mt-10 grid gap-3 sm:grid-cols-3">
              {examples.map((example) => (
                <div
                  key={example}
                  className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-[rgb(var(--muted))]"
                >
                  {example}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="glass-card rounded-[28px] p-6">
            <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
              MVP flow
            </p>
            <div className="mt-5 space-y-4">
              {[
                "Choose saved, raw, or connector sources",
                "Search instantly with no required index step",
                "Inspect ranked spans before optional synthesis",
              ].map((step, index) => (
                <div
                  key={step}
                  className="flex gap-3 rounded-2xl border border-white/10 bg-black/15 p-4"
                >
                  <span className="flex h-8 w-8 items-center justify-center rounded-full bg-sky-400/15 text-sm font-semibold text-cyan-300">
                    {index + 1}
                  </span>
                  <p className="text-sm leading-6 text-[rgb(var(--text))]">
                    {step}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-card rounded-[28px] p-6">
            <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
              Why it works
            </p>
            <div className="mt-5 space-y-4">
              {highlights.map((item) => (
                <div key={item.title} className="rounded-2xl border border-white/10 bg-black/15 p-4">
                  <h2 className="font-display text-lg text-white">
                    {item.title}
                  </h2>
                  <p className="mt-2 text-sm leading-6 text-[rgb(var(--muted))]">
                    {item.body}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
