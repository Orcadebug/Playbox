"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const items = [
  {
    href: "/search",
    title: "Search",
    body: "Ask questions across any uploaded corpus.",
  },
  {
    href: "/upload",
    title: "Upload",
    body: "Drop files, paste text, or connect a source.",
  },
  {
    href: "/demo",
    title: "Demo",
    body: "Use a seeded corpus with no setup.",
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="glass-card hidden h-fit min-w-[280px] rounded-[28px] p-4 lg:block">
      <div className="rounded-[24px] border border-white/10 bg-black/15 p-4">
        <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
          Workspace
        </p>
        <div className="font-display mt-2 text-2xl text-white">
          Query anything.
        </div>
        <p className="mt-2 text-sm leading-6 text-[rgb(var(--muted))]">
          Waver keeps the flow simple: ingest, search, understand, cite.
        </p>
      </div>

      <nav className="mt-4 space-y-2">
        {items.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
                className={[
                  "block rounded-2xl border p-4 transition",
                active
                  ? "border-cyan-300/30 bg-cyan-400/10"
                  : "border-white/10 bg-white/5 hover:bg-white/10",
              ].join(" ")}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="font-semibold text-white">{item.title}</div>
                <span className="text-xs uppercase tracking-[0.22em] text-[rgb(var(--muted))]">
                  {active ? "Active" : "Open"}
                </span>
              </div>
              <p className="mt-2 text-sm leading-6 text-[rgb(var(--muted))]">
                {item.body}
              </p>
            </Link>
          );
        })}
      </nav>

      <div className="mt-4 rounded-[24px] border border-white/10 bg-black/15 p-4">
        <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
          Runtime status
        </p>
        <div className="mt-3 flex items-center gap-3">
          <span className="h-2.5 w-2.5 rounded-full bg-cyan-300" />
          <p className="text-sm text-[rgb(var(--text))]">
            Backend-aware. Falls back to mock data if the API is unreachable.
          </p>
        </div>
      </div>
    </aside>
  );
}
