import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, HelpCircle, ShieldAlert } from "lucide-react";
import { Badge, GlassCard } from "@/components/ui";
import type { CriticalFlag, JobStatus, Observation, RiskAssessment } from "@/mocks/data";
import { pct } from "@/lib/format";
import { cn } from "@/lib/cn";

export function StatusBadge({ status }: { status: JobStatus }) {
  const map: Record<
    JobStatus,
    { tone: "brand" | "neutral" | "warn" | "info" | "critical"; label: string }
  > = {
    QUEUED: { tone: "neutral", label: "Queued" },
    EXTRACTING: { tone: "info", label: "Extracting" },
    AWAITING_REVIEW: { tone: "warn", label: "Needs your review" },
    EVALUATING: { tone: "info", label: "Checking safety" },
    ASSESSING: { tone: "info", label: "Assessing" },
    EXPLAINING: { tone: "info", label: "Writing summary" },
    DONE: { tone: "brand", label: "Done" },
    FAILED: { tone: "critical", label: "Failed" },
  };
  const m = map[status];
  return (
    <Badge tone={m.tone} dot>
      {m.label}
    </Badge>
  );
}

export const JOB_STEPS: { id: JobStatus; label: string }[] = [
  { id: "EXTRACTING", label: "Extract" },
  { id: "AWAITING_REVIEW", label: "Review" },
  { id: "EVALUATING", label: "Safety" },
  { id: "ASSESSING", label: "Assess" },
  { id: "EXPLAINING", label: "Explain" },
  { id: "DONE", label: "Done" },
];

export function JobProgress({ status }: { status: JobStatus }) {
  const idx = JOB_STEPS.findIndex((s) => s.id === status);
  return (
    <ol className="flex items-center gap-2">
      {JOB_STEPS.map((s, i) => {
        const done = i < idx || status === "DONE";
        const active = i === idx && status !== "DONE";
        return (
          <li key={s.id} className="flex flex-1 items-center gap-2">
            <div className="flex flex-col items-center">
              <span
                className={cn(
                  "grid size-7 place-items-center rounded-full text-[11px] font-bold transition",
                  done
                    ? "bg-brand-600 text-white"
                    : active
                      ? "bg-brand-100 text-brand-800 ring-2 ring-brand-400"
                      : "bg-white/70 text-ink-muted",
                )}
              >
                {done ? "✓" : i + 1}
              </span>
              <span
                className={cn(
                  "mt-1 text-[10px] font-semibold",
                  active ? "text-brand-800" : "text-ink-muted",
                )}
              >
                {s.label}
              </span>
            </div>
            {i < JOB_STEPS.length - 1 && (
              <span className={cn("mb-4 h-px flex-1", done ? "bg-brand-500" : "bg-line")} />
            )}
          </li>
        );
      })}
    </ol>
  );
}

export function inRange(o: Observation) {
  if (o.refLow !== undefined && o.value < o.refLow) return "low";
  if (o.refHigh !== undefined && o.value > o.refHigh) return "high";
  return "normal";
}

export function CriticalBanner({ flags }: { flags: CriticalFlag[] }) {
  if (!flags.length) return null;
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className="relative overflow-hidden rounded-xl3 border border-red-300 bg-gradient-to-br from-red-600 to-red-700 p-6 text-white shadow-[0_20px_60px_-20px_rgb(220_38_38/0.6)]"
    >
      <span className="absolute -right-10 -top-10 size-40 rounded-full bg-white/10 blur-2xl" />
      <div className="flex items-start gap-4">
        <span className="grid size-12 shrink-0 place-items-center rounded-2xl bg-white/15">
          <ShieldAlert className="size-6" />
        </span>
        <div className="flex-1">
          <p className="text-xs font-bold uppercase tracking-[0.2em] text-white/80">
            Critical value detected
          </p>
          {flags.map((f) => (
            <div key={f.analyte} className="mt-3">
              <p className="font-display text-2xl font-bold">
                {f.analyte}: {f.value.toLocaleString()} {f.unit}{" "}
                <span className="text-base font-semibold text-white/80">({f.limit})</span>
              </p>
              <p className="mt-1 text-sm text-white/90">{f.message}</p>
              <p className="mt-1 text-[11px] text-white/60">Rule source: {f.guideline}</p>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

export function RiskCard({ a, index = 0 }: { a: RiskAssessment; index?: number }) {
  const assessed = a.status === "ASSESSED";
  const p = a.probability ?? 0;
  const over = assessed && a.threshold !== undefined && p >= a.threshold;
  const tone = !assessed ? "neutral" : over ? "warn" : "brand";
  return (
    <GlassCard
      hover
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className={cn("relative overflow-hidden", !assessed && "border-dashed")}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-ink-muted">
            {a.condition.replace("_", " ")}
          </p>
          <h3 className="mt-1 text-lg font-bold">{a.title}</h3>
        </div>
        {assessed ? (
          over ? (
            <AlertTriangle className="size-5 text-amber-600" />
          ) : (
            <CheckCircle2 className="size-5 text-brand-600" />
          )
        ) : (
          <HelpCircle className="size-5 text-ink-muted" />
        )}
      </div>
      {assessed ? (
        <>
          <div className="mt-5 flex items-end justify-between">
            <p className="font-display text-4xl font-bold text-ink">{pct(p)}</p>
            <Badge tone={tone}>{a.label}</Badge>
          </div>
          <div className="relative mt-3 h-2 w-full overflow-hidden rounded-full bg-brand-900/10">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${p * 100}%` }}
              transition={{ duration: 1, delay: 0.2 + index * 0.06, ease: [0.16, 1, 0.3, 1] }}
              className={cn(
                "h-full rounded-full bg-gradient-to-r",
                over ? "from-amber-300 to-amber-500" : "from-brand-400 to-brand-600",
              )}
            />
            {a.threshold !== undefined && (
              <span
                className="absolute top-0 h-full w-0.5 bg-ink/40"
                style={{ left: `${a.threshold * 100}%` }}
                title={`Decision threshold ${pct(a.threshold)}`}
              />
            )}
          </div>
          <p className="mt-2 text-[11px] text-ink-muted">
            Threshold {a.threshold !== undefined && pct(a.threshold)} · {a.modelVersion}
          </p>
        </>
      ) : (
        <>
          <p className="mt-5 font-display text-2xl font-bold text-ink-muted">Not assessable</p>
          <p className="mt-2 text-sm text-ink-soft">This report is missing:</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {a.missing?.map((m) => (
              <span
                key={m}
                className="rounded-full bg-white/80 px-2.5 py-1 text-xs font-semibold text-ink-soft"
              >
                {m}
              </span>
            ))}
          </div>
        </>
      )}
    </GlassCard>
  );
}
