import Link from "next/link";

const links = [
  { href: "/search", label: "Search" },
  { href: "/upload", label: "Upload" },
  { href: "/demo", label: "Demo" },
];

export function Header({
  mode = "app",
}: {
  mode?: "app" | "landing";
}) {
  return (
    <header className="mx-auto flex max-w-7xl items-center justify-between gap-4 rounded-[24px] border border-white/10 bg-black/20 px-5 py-4 backdrop-blur md:px-6">
      <Link href="/" className="flex items-center gap-3">
        <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-cyan-400/10 text-lg font-bold text-cyan-300">
          W
        </span>
        <div>
          <div className="font-display text-lg font-semibold tracking-tight text-white">
            Waver
          </div>
          <div className="text-xs text-[rgb(var(--muted))]">
            Search & understand messy data
          </div>
        </div>
      </Link>

      <nav className="hidden items-center gap-2 md:flex">
        {links.map((link) => (
          <Link
            key={link.href}
            href={link.href as any}
            className="rounded-full border border-white/10 px-4 py-2 text-sm text-[rgb(var(--muted))] transition hover:border-white/20 hover:bg-white/5 hover:text-white"
          >
            {link.label}
          </Link>
        ))}
      </nav>

      <div className="flex items-center gap-2">
        <span className="rounded-full border border-cyan-300/25 bg-cyan-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-cyan-200">
          {mode === "landing" ? "MVP" : "Live scaffold"}
        </span>
      </div>
    </header>
  );
}
