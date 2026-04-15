import type { ReactNode } from "react";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";

export function AppShell({
  eyebrow,
  title,
  description,
  children,
}: Readonly<{
  eyebrow: string;
  title: string;
  description: string;
  children: ReactNode;
}>) {
  return (
    <main className="min-h-screen px-4 py-5 md:px-6">
      <Header mode="app" />
      <div className="mx-auto mt-6 flex max-w-7xl gap-6">
        <Sidebar />
        <section className="min-w-0 flex-1">
          <div className="glass-card rounded-[28px] p-6 md:p-8">
            <p className="text-xs uppercase tracking-[0.28em] text-[rgb(var(--muted))]">
              {eyebrow}
            </p>
            <h1 className="font-display mt-3 text-3xl font-semibold tracking-tight text-white md:text-5xl">
              {title}
            </h1>
            <p className="mt-4 max-w-3xl text-base leading-7 text-[rgb(var(--muted))] md:text-lg">
              {description}
            </p>
          </div>
          <div className="mt-6">{children}</div>
        </section>
      </div>
    </main>
  );
}
