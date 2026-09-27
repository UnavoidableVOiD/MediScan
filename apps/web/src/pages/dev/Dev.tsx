import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowUpRight, Copy, Palette } from "lucide-react";
import {
  Avatar,
  Badge,
  Button,
  GlassCard,
  Input,
  Progress,
  SectionHeading,
  Stat,
} from "@/components/ui";
import { PageTransition, Reveal } from "@/components/motion";
import { ROUTES, resolve, type RouteGroup } from "@/app/routes";

const groups: RouteGroup[] = ["Public", "Auth", "Patient", "Doctor", "Admin", "Developer"];

export function PagesIndex() {
  const base = typeof window !== "undefined" ? window.location.origin : "";
  return (
    <PageTransition>
      <section className="relative pt-36 pb-28">
        <div className="pointer-events-none absolute inset-0 grid-dots opacity-60" />
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6">
          <Reveal>
            <SectionHeading
              eyebrow="Developer"
              title={
                <>
                  Every page, <span className="text-gradient">one click away.</span>
                </>
              }
              lead={`${ROUTES.length} routes. No authentication is enforced during the building phase — everything is reachable. Base: ${base}`}
            />
          </Reveal>
          <div className="mt-12 space-y-10">
            {groups.map((g, gi) => (
              <Reveal key={g} delay={gi * 0.05}>
                <h3 className="mb-3 text-xs font-bold uppercase tracking-[0.2em] text-brand-700">
                  {g}
                </h3>
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {ROUTES.filter((r) => r.group === g).map((r) => (
                    <motion.div key={r.path} whileHover={{ y: -3 }}>
                      <div className="glass flex h-full flex-col p-5">
                        <div className="flex items-start justify-between gap-2">
                          <Link to={resolve(r)} className="font-bold hover:text-brand-700">
                            {r.name}
                          </Link>
                          <Link
                            to={resolve(r)}
                            className="text-ink-muted hover:text-brand-700"
                            aria-label="Open"
                          >
                            <ArrowUpRight className="size-4" />
                          </Link>
                        </div>
                        <p className="mt-1 flex-1 text-sm text-ink-soft">{r.description}</p>
                        <div className="mt-3 flex items-center gap-2">
                          <code className="flex-1 truncate rounded-lg bg-white/80 px-2 py-1 font-mono text-[11px] text-brand-900">
                            {resolve(r)}
                          </code>
                          <button
                            onClick={() => navigator.clipboard?.writeText(base + resolve(r))}
                            className="grid size-7 place-items-center rounded-lg text-ink-muted hover:bg-brand-50 hover:text-brand-800"
                            aria-label="Copy URL"
                          >
                            <Copy className="size-3.5" />
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>
    </PageTransition>
  );
}

export function StyleGuide() {
  const swatches = ["50", "100", "200", "300", "400", "500", "600", "700", "800", "900", "950"];
  return (
    <PageTransition>
      <section className="relative pt-36 pb-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <Reveal>
            <SectionHeading
              eyebrow="Design system"
              title={
                <>
                  Clinical Glass — <span className="text-gradient">tokens & components.</span>
                </>
              }
              lead="Green and white only. Frosted surfaces on a paper background. Sora for display, Manrope for body."
            />
          </Reveal>
          <div className="mt-12 space-y-10">
            <GlassCard>
              <h3 className="flex items-center gap-2 text-lg font-bold">
                <Palette className="size-5 text-brand-600" /> Brand scale
              </h3>
              <div className="mt-4 grid grid-cols-6 gap-2 sm:grid-cols-11">
                {swatches.map((s) => (
                  <div key={s} className="text-center">
                    <div
                      className={`h-14 rounded-xl bg-brand-${s}`}
                      style={{ background: `var(--color-brand-${s})` }}
                    />
                    <p className="mt-1 text-[10px] font-semibold text-ink-muted">{s}</p>
                  </div>
                ))}
              </div>
            </GlassCard>
            <GlassCard>
              <h3 className="text-lg font-bold">Type</h3>
              <h1 className="mt-4 text-6xl font-bold">Display / Sora 800</h1>
              <h2 className="mt-2 text-3xl font-bold">Heading / Sora 700</h2>
              <p className="mt-2 text-lg text-ink-soft">
                Lead / Manrope 500 — MediScan reads your report and explains it in plain language.
              </p>
              <p className="mt-2 text-sm text-ink-soft">
                Body / Manrope 400 — every number validated against your report.
              </p>
              <p className="mt-2 text-[11px] font-bold uppercase tracking-[0.2em] text-brand-700">
                Eyebrow / 11px 700 tracking 0.2em
              </p>
            </GlassCard>
            <GlassCard>
              <h3 className="text-lg font-bold">Buttons</h3>
              <div className="mt-4 flex flex-wrap gap-2">
                <Button>Primary</Button>
                <Button variant="secondary">Secondary</Button>
                <Button variant="glass">Glass</Button>
                <Button variant="ghost">Ghost</Button>
                <Button variant="danger">Danger</Button>
                <Button size="sm">Small</Button>
                <Button size="lg">Large</Button>
              </div>
              <h3 className="mt-8 text-lg font-bold">Badges</h3>
              <div className="mt-4 flex flex-wrap gap-2">
                <Badge>brand</Badge>
                <Badge tone="neutral">neutral</Badge>
                <Badge tone="info">info</Badge>
                <Badge tone="warn">warn</Badge>
                <Badge tone="critical" dot>
                  critical
                </Badge>
              </div>
              <h3 className="mt-8 text-lg font-bold">Inputs & progress</h3>
              <div className="mt-4 grid gap-4 sm:grid-cols-2">
                <Input label="Label" name="x" placeholder="Placeholder" hint="Hint text" />
                <div className="space-y-3 pt-6">
                  <Progress value={72} />
                  <Progress value={45} tone="warn" />
                  <Progress value={90} tone="critical" />
                </div>
              </div>
            </GlassCard>
            <div className="grid gap-4 sm:grid-cols-3">
              <Stat label="Stat card" value="1,284" delta="+84 this month" />
              <GlassCard strong className="flex items-center gap-3">
                <Avatar name="Sunita Karki" size={44} />{" "}
                <span className="font-bold">Avatar + glass-strong</span>
              </GlassCard>
              <GlassCard dark>
                <p className="text-xs font-bold uppercase tracking-wider text-brand-300">
                  glass-dark
                </p>
                <p className="mt-1 font-bold">For dark sections</p>
              </GlassCard>
            </div>
          </div>
        </div>
      </section>
    </PageTransition>
  );
}

export function NotFound() {
  return (
    <PageTransition>
      <section className="flex min-h-screen flex-col items-center justify-center px-4 text-center">
        <p className="font-display text-8xl font-bold text-brand-200">404</p>
        <h1 className="mt-2 text-3xl font-bold">This page isn't in the chart.</h1>
        <p className="mt-2 text-ink-soft">Try the pages index — every route is listed there.</p>
        <div className="mt-6 flex gap-2">
          <Button to="/pages">Pages index</Button>
          <Button to="/" variant="glass">
            Home
          </Button>
        </div>
      </section>
    </PageTransition>
  );
}
