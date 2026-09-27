import { useState } from "react";
import { Link } from "react-router-dom";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  ArrowRight,
  CalendarClock,
  FileText,
  ShieldAlert,
  Stethoscope,
  TrendingUp,
  UploadCloud,
} from "lucide-react";
import { Avatar, Badge, Button, GlassCard, LinkCard, PageHead, Stat } from "@/components/ui";
import { Reveal } from "@/components/motion";
import { appointments, doctors, me, reports, trends } from "@/mocks/data";
import { fmtDate, fmtTime } from "@/lib/format";
import { StatusBadge } from "./shared";
import { cn } from "@/lib/cn";

const metrics = ["Hemoglobin", "Creatinine", "Fasting glucose", "LDL"] as const;

export default function PatientDashboard() {
  const [metric, setMetric] = useState<(typeof metrics)[number]>("Hemoglobin");
  const latest = reports[0];
  const flagged = reports.find((r) => r.flags.length);
  const next = appointments.find((a) => a.status === "PAID");
  const nextDoc = doctors.find((d) => d.id === next?.doctorId);

  return (
    <>
      <PageHead
        title={
          <>
            Good morning, <span className="text-gradient">{me.name.split(" ")[0]}</span>
          </>
        }
        lead="Here's what your recent reports say and what to do next."
        actions={
          <Button to="/reports/upload" icon={UploadCloud}>
            Upload report
          </Button>
        }
      />

      {flagged && (
        <Reveal>
          <div className="mb-6 flex flex-col gap-4 rounded-xl3 border border-red-200 bg-red-50/80 p-5 backdrop-blur sm:flex-row sm:items-center">
            <span className="grid size-11 shrink-0 place-items-center rounded-2xl bg-red-600 text-white">
              <ShieldAlert className="size-5" />
            </span>
            <div className="flex-1">
              <p className="font-bold text-red-800">
                Critical value in your {fmtDate(flagged.uploadedAt)} report
              </p>
              <p className="text-sm text-red-700/90">{flagged.flags[0].message}</p>
            </div>
            <Button
              to={`/reports/${flagged.id}/result`}
              variant="danger"
              size="sm"
              icon={ArrowRight}
              trailing
            >
              View
            </Button>
          </div>
        </Reveal>
      )}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat
          label="Reports analysed"
          value={reports.filter((r) => r.status === "DONE").length}
          icon={FileText}
          delta="+1 this week"
        />
        <Stat
          label="Values tracked"
          value={latest.observations.length}
          icon={Activity}
          delta="across 6 panels"
        />
        <Stat
          label="Assessed conditions"
          value={`${latest.assessments.filter((a) => a.status === "ASSESSED").length}/6`}
          icon={TrendingUp}
          delta="1 not assessable"
        />
        <Stat
          label="Next consultation"
          value={next ? fmtTime(next.start) : "—"}
          icon={CalendarClock}
          delta={next ? fmtDate(next.date) : "Nothing booked"}
        />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1.5fr_1fr]">
        <Reveal>
          <GlassCard className="h-full">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs font-bold uppercase tracking-wider text-ink-muted">Trend</p>
                <h3 className="text-xl font-bold">{metric} over time</h3>
              </div>
              <div className="glass-pill flex p-1">
                {metrics.map((m) => (
                  <button
                    key={m}
                    onClick={() => setMetric(m)}
                    className={cn(
                      "rounded-full px-3 py-1.5 text-xs font-semibold transition",
                      metric === m ? "bg-brand-600 text-white" : "text-ink-soft hover:text-ink",
                    )}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>
            <div className="mt-6 h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trends} margin={{ left: -20, right: 8, top: 8 }}>
                  <defs>
                    <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10b981" stopOpacity={0.45} />
                      <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid vertical={false} stroke="rgba(6,78,59,0.08)" />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 12, fill: "#7b918a" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 12, fill: "#7b918a" }}
                    axisLine={false}
                    tickLine={false}
                    domain={["auto", "auto"]}
                  />
                  <Tooltip
                    contentStyle={{
                      borderRadius: 16,
                      border: "1px solid rgba(255,255,255,.8)",
                      background: "rgba(255,255,255,.9)",
                      fontSize: 12,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey={metric}
                    stroke="#047857"
                    strokeWidth={2.5}
                    fill="url(#g1)"
                    dot={{ r: 4, fill: "#047857", strokeWidth: 2, stroke: "#fff" }}
                    activeDot={{ r: 6 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </GlassCard>
        </Reveal>

        <Reveal delay={0.1}>
          <GlassCard className="h-full">
            <p className="text-xs font-bold uppercase tracking-wider text-ink-muted">Upcoming</p>
            {next && nextDoc ? (
              <>
                <div className="mt-4 flex items-center gap-4">
                  <Avatar name={nextDoc.name} size={52} />
                  <div>
                    <p className="font-bold">{nextDoc.name}</p>
                    <p className="text-xs text-ink-muted">{nextDoc.hospital}</p>
                  </div>
                </div>
                <div className="mt-4 rounded-2xl bg-brand-900 p-4 text-white">
                  <p className="text-[11px] font-bold uppercase tracking-wider text-brand-300">
                    {fmtDate(next.date)}
                  </p>
                  <p className="mt-1 font-display text-2xl font-bold">
                    {fmtTime(next.start)} – {fmtTime(next.end)}
                  </p>
                  <p className="mt-2 text-xs text-white/70">Linked report: {next.reportId}</p>
                </div>
                <Button
                  to="/appointments"
                  variant="glass"
                  className="mt-4 w-full"
                  icon={ArrowRight}
                  trailing
                >
                  Manage appointments
                </Button>
              </>
            ) : (
              <p className="mt-4 text-sm text-ink-soft">No consultations booked.</p>
            )}
            <div className="mt-6 border-t border-line pt-5">
              <p className="text-xs font-bold uppercase tracking-wider text-ink-muted">
                Suggested for you
              </p>
              <p className="mt-2 text-sm text-ink-soft">
                Your last report suggests a follow-up with a hepatologist.
              </p>
              <Button to="/doctors" size="sm" className="mt-3" icon={Stethoscope}>
                Find hepatologists
              </Button>
            </div>
          </GlassCard>
        </Reveal>
      </div>

      <Reveal className="mt-8">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-xl font-bold">Recent reports</h3>
          <Link to="/reports" className="text-sm font-semibold text-brand-700 hover:underline">
            View all
          </Link>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {reports.map((r) => (
            <Link
              key={r.id}
              to={
                r.status === "AWAITING_REVIEW"
                  ? `/reports/${r.id}/review`
                  : `/reports/${r.id}/result`
              }
              className="group glass block p-5 transition-all duration-500 hover:-translate-y-1 hover:shadow-glow"
            >
              <div className="flex items-start justify-between">
                <span className="grid size-10 place-items-center rounded-xl bg-brand-100 text-brand-700">
                  <FileText className="size-4" />
                </span>
                <StatusBadge status={r.status} />
              </div>
              <p className="mt-4 font-bold leading-tight">{r.lab}</p>
              <p className="mt-1 text-xs text-ink-muted">
                {fmtDate(r.uploadedAt)} · {r.observations.length} values · {r.pages} page
                {r.pages > 1 && "s"}
              </p>
              {r.flags.length > 0 && (
                <Badge tone="critical" className="mt-3" dot>
                  {r.flags.length} critical
                </Badge>
              )}
            </Link>
          ))}
        </div>
      </Reveal>

      <div className="mt-8 grid gap-4 md:grid-cols-3">
        <LinkCard
          to="/reports/upload"
          icon={UploadCloud}
          title="Upload a new report"
          body="PDF or photo. Takes about a minute."
        />
        <LinkCard
          to="/doctors"
          icon={Stethoscope}
          title="Find a doctor"
          body="Verified specialists matched to your results."
        />
        <LinkCard
          to="/profile"
          icon={Activity}
          title="Your profile"
          body="Personal details, data export and deletion."
        />
      </div>
    </>
  );
}
