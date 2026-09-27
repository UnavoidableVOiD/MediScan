import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis } from "recharts";
import {
  ArrowRight,
  BadgeCheck,
  CalendarClock,
  CheckCircle2,
  Clock,
  FileText,
  FileUp,
  MessageSquareText,
  Plus,
  Search,
  ShieldCheck,
  Stethoscope,
  Trash2,
  TrendingUp,
  Users,
  Wallet,
  X,
} from "lucide-react";
import {
  Avatar,
  Badge,
  Button,
  GlassCard,
  Input,
  PageHead,
  Stat,
  Table,
  Tabs,
  Td,
  Textarea,
} from "@/components/ui";
import { Reveal } from "@/components/motion";
import { appointments, doctors, patients, reports, specializationLabel } from "@/mocks/data";
import { fmtDate, fmtMoney, fmtTime } from "@/lib/format";
import { cn } from "@/lib/cn";
import { CriticalBanner, RiskCard, inRange } from "../patient/shared";

const meDoc = doctors[0];
const myAppts = appointments.filter((a) => a.doctorId === meDoc.id);

/* ---------------------------------------------------------- dashboard */
export function DoctorDashboard() {
  const today = myAppts.filter((a) => a.status === "PAID");
  const revenue = myAppts
    .filter((a) => ["PAID", "COMPLETED"].includes(a.status))
    .reduce((s, a) => s + a.amount * 0.75, 0);
  const week = [3, 5, 4, 6, 7, 2, 1].map((n, i) => ({
    d: ["M", "T", "W", "T", "F", "S", "S"][i],
    n,
  }));
  return (
    <>
      <PageHead
        title={
          <>
            Welcome, <span className="text-gradient">{meDoc.name}</span>
          </>
        }
        lead={`${specializationLabel[meDoc.specialization]} · ${meDoc.hospital}`}
        actions={
          <Button to="/doctor/availability" icon={CalendarClock}>
            Manage availability
          </Button>
        }
      />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Patients" value={patients.length} icon={Users} delta="+2 this week" />
        <Stat
          label="Today's consultations"
          value={today.length}
          icon={Clock}
          delta="next at 4:30 PM"
        />
        <Stat
          label="Net revenue (month)"
          value={fmtMoney(revenue)}
          icon={Wallet}
          delta="after 25% platform share"
        />
        <Stat
          label="Reports awaiting comment"
          value={2}
          icon={MessageSquareText}
          delta="1 has a critical flag"
        />
      </div>
      <div className="mt-6 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <Reveal>
          <GlassCard className="h-full">
            <h3 className="text-xl font-bold">Today</h3>
            <div className="mt-4 space-y-3">
              {today.map((a) => {
                const p = patients.find((x) => x.name === a.patient);
                return (
                  <div
                    key={a.id}
                    className="flex flex-wrap items-center gap-4 rounded-2xl bg-white/70 p-4"
                  >
                    <div className="rounded-xl bg-brand-900 px-3 py-2 text-center text-white">
                      <p className="font-display text-lg font-bold leading-none">
                        {fmtTime(a.start).split(" ")[0]}
                      </p>
                      <p className="text-[10px] font-bold">{fmtTime(a.start).split(" ")[1]}</p>
                    </div>
                    <Avatar name={a.patient} size={44} />
                    <div className="min-w-0 flex-1">
                      <p className="font-bold">{a.patient}</p>
                      <p className="text-xs text-ink-muted">
                        {p?.age} y · {p?.sex} · {p?.city}
                      </p>
                    </div>
                    {p?.risk === "High" && (
                      <Badge tone="critical" dot>
                        High risk
                      </Badge>
                    )}
                    <Button
                      to={`/doctor/patients/${p?.id ?? "p1"}`}
                      size="sm"
                      variant="glass"
                      icon={ArrowRight}
                      trailing
                    >
                      Open
                    </Button>
                  </div>
                );
              })}
            </div>
          </GlassCard>
        </Reveal>
        <Reveal delay={0.1}>
          <GlassCard className="h-full">
            <h3 className="text-xl font-bold">Consultations this week</h3>
            <div className="mt-4 h-44">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={week}>
                  <XAxis
                    dataKey="d"
                    tick={{ fontSize: 12, fill: "#7b918a" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(16,185,129,0.08)" }}
                    contentStyle={{ borderRadius: 12, fontSize: 12 }}
                  />
                  <Bar dataKey="n" radius={[8, 8, 8, 8]} fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 flex items-center gap-2 rounded-2xl bg-brand-50 p-3 text-sm text-brand-900">
              <TrendingUp className="size-4" /> 28 consultations · up 12% on last week
            </div>
          </GlassCard>
        </Reveal>
      </div>
    </>
  );
}

/* ------------------------------------------------------- appointments */
export function DoctorAppointments() {
  const [tab, setTab] = useState<"upcoming" | "completed" | "cancelled">("upcoming");
  const list = myAppts.filter((a) =>
    tab === "upcoming" ? ["PAID", "PENDING"].includes(a.status) : a.status === tab.toUpperCase(),
  );
  return (
    <>
      <PageHead title="Appointments" lead="Confirmed bookings appear once payment is verified." />
      <Tabs
        tabs={[
          {
            id: "upcoming",
            label: "Upcoming",
            count: myAppts.filter((a) => ["PAID", "PENDING"].includes(a.status)).length,
          },
          { id: "completed", label: "Completed" },
          { id: "cancelled", label: "Cancelled" },
        ]}
        value={tab}
        onChange={setTab}
      />
      <div className="mt-6">
        <Table head={["Patient", "Date", "Time", "Status", "Fee", "Report", ""]}>
          {list.map((a) => {
            const p = patients.find((x) => x.name === a.patient);
            return (
              <tr key={a.id} className="transition hover:bg-white/50">
                <Td>
                  <span className="flex items-center gap-3">
                    <Avatar name={a.patient} size={32} />
                    <span>
                      <span className="block font-semibold">{a.patient}</span>
                      <span className="block text-xs text-ink-muted">{p?.city}</span>
                    </span>
                  </span>
                </Td>
                <Td>{fmtDate(a.date)}</Td>
                <Td>{fmtTime(a.start)}</Td>
                <Td>
                  <Badge
                    tone={
                      a.status === "PAID"
                        ? "brand"
                        : a.status === "PENDING"
                          ? "warn"
                          : a.status === "COMPLETED"
                            ? "info"
                            : "critical"
                    }
                  >
                    {a.status}
                  </Badge>
                </Td>
                <Td>{fmtMoney(a.amount)}</Td>
                <Td>
                  {a.reportId ? (
                    <Link
                      to={`/doctor/patients/${p?.id ?? "p1"}`}
                      className="font-mono text-xs font-semibold text-brand-700 hover:underline"
                    >
                      {a.reportId}
                    </Link>
                  ) : (
                    <span className="text-ink-muted">—</span>
                  )}
                </Td>
                <Td className="text-right">
                  {a.status === "PAID" && (
                    <Button size="sm" variant="glass" icon={CheckCircle2}>
                      Complete
                    </Button>
                  )}
                </Td>
              </tr>
            );
          })}
        </Table>
      </div>
    </>
  );
}

/* ----------------------------------------------------------- patients */
export function DoctorPatients() {
  const [q, setQ] = useState("");
  const list = patients.filter((p) => p.name.toLowerCase().includes(q.toLowerCase()));
  return (
    <>
      <PageHead
        title="Patients"
        lead="People who have booked or been linked to you."
        actions={
          <div className="w-64">
            <Input
              name="q"
              icon={Search}
              placeholder="Search patients"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
        }
      />
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {list.map((p, i) => (
          <motion.div
            key={p.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.04 }}
          >
            <Link
              to={`/doctor/patients/${p.id}`}
              className="glass block p-5 transition-all duration-500 hover:-translate-y-1 hover:shadow-glow"
            >
              <div className="flex items-center gap-4">
                <Avatar name={p.name} size={52} />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-bold">{p.name}</p>
                  <p className="text-xs text-ink-muted">
                    {p.age} y · {p.sex} · {p.city}
                  </p>
                </div>
                <Badge
                  tone={p.risk === "High" ? "critical" : p.risk === "Medium" ? "warn" : "brand"}
                >
                  {p.risk}
                </Badge>
              </div>
              <div className="mt-4 flex items-center justify-between text-xs text-ink-muted">
                <span>Last report {fmtDate(p.lastReport)}</span>
                <ArrowRight className="size-4" />
              </div>
            </Link>
          </motion.div>
        ))}
      </div>
    </>
  );
}

/* ------------------------------------------------------ patient detail */
export function PatientDetail() {
  const { id } = useParams();
  const p = patients.find((x) => x.id === id) ?? patients[0];
  const r = p.risk === "High" ? reports[1] : reports[0];
  const [comment, setComment] = useState(r.doctorComment?.text ?? "");
  const [tab, setTab] = useState<"summary" | "values" | "history">("summary");
  return (
    <>
      <PageHead
        crumbs={["Patients", p.name]}
        title={p.name}
        lead={`${p.age} years · ${p.sex === "M" ? "Male" : "Female"} · ${p.city} · ${p.email}`}
        actions={
          <>
            <Button variant="glass" icon={FileUp}>
              Request new report
            </Button>
            <Button icon={CheckCircle2}>Mark consultation complete</Button>
          </>
        }
      />
      <div className="space-y-6">
        <CriticalBanner flags={r.flags} />
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Tabs
            tabs={[
              { id: "summary", label: "Clinical summary" },
              { id: "values", label: "Values", count: r.observations.length },
              { id: "history", label: "History" },
            ]}
            value={tab}
            onChange={setTab}
          />
          <p className="text-xs text-ink-muted">
            Report {r.id} · {r.lab} · {fmtDate(r.uploadedAt)}
          </p>
        </div>

        {tab === "summary" && (
          <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
            <div className="space-y-6">
              <Reveal>
                <GlassCard strong>
                  <p className="text-xs font-bold uppercase tracking-wider text-ink-muted">
                    Clinician summary
                  </p>
                  <p className="mt-3 leading-relaxed">{r.explanation.clinician}</p>
                  <p className="mt-3 text-xs text-ink-muted">
                    Generated from the confirmed values and model outputs below · citations:{" "}
                    {r.explanation.citations.map((c) => c.doc).join(", ")}
                  </p>
                </GlassCard>
              </Reveal>
              <div className="grid gap-4 sm:grid-cols-2">
                {r.assessments.map((a, i) => (
                  <RiskCard key={a.condition} a={a} index={i} />
                ))}
              </div>
            </div>
            <Reveal delay={0.1}>
              <GlassCard className="sticky top-24">
                <h3 className="flex items-center gap-2 text-lg font-bold">
                  <MessageSquareText className="size-5 text-brand-600" /> Your comment on this
                  report
                </h3>
                <p className="mt-1 text-xs text-ink-muted">
                  The patient sees this beneath their explanation.
                </p>
                <Textarea
                  className="mt-4"
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  placeholder="Clinical observations, next steps, reassurance…"
                />
                <div className="mt-3 flex gap-2">
                  <Button size="sm">Save comment</Button>
                  <Button size="sm" variant="ghost" onClick={() => setComment("")}>
                    Clear
                  </Button>
                </div>
                <div className="mt-6 border-t border-line pt-4">
                  <p className="text-xs font-bold uppercase tracking-wider text-ink-muted">
                    Private notes
                  </p>
                  <Textarea className="mt-2 min-h-20" placeholder="Only you can see these." />
                </div>
              </GlassCard>
            </Reveal>
          </div>
        )}

        {tab === "values" && (
          <Table head={["Analyte", "LOINC", "Value", "Reference", "Status", "Source"]}>
            {r.observations.map((o) => {
              const s = inRange(o);
              return (
                <tr key={o.loinc} className="hover:bg-white/50">
                  <Td className="font-semibold">{o.analyte}</Td>
                  <Td className="font-mono text-xs text-ink-muted">{o.loinc}</Td>
                  <Td>
                    <span className="font-display font-bold">{o.value.toLocaleString()}</span>{" "}
                    <span className="text-xs text-ink-muted">{o.unit}</span>
                  </Td>
                  <Td className="text-ink-soft">
                    {o.refLow ?? "—"} – {o.refHigh ?? "—"}
                  </Td>
                  <Td>
                    {s === "normal" ? (
                      <Badge tone="brand">Normal</Badge>
                    ) : (
                      <Badge tone="warn">{s}</Badge>
                    )}
                  </Td>
                  <Td className="text-xs text-ink-muted">
                    {o.source === "MANUAL"
                      ? "Patient-verified"
                      : `OCR ${Math.round((o.confidence ?? 1) * 100)}%`}
                  </Td>
                </tr>
              );
            })}
          </Table>
        )}

        {tab === "history" && (
          <div className="space-y-3">
            {reports.map((rep) => (
              <GlassCard key={rep.id} className="flex flex-wrap items-center gap-4">
                <span className="grid size-10 place-items-center rounded-xl bg-brand-100 text-brand-700">
                  <FileText className="size-4" />
                </span>
                <div className="min-w-0 flex-1">
                  <p className="font-bold">{rep.lab}</p>
                  <p className="text-xs text-ink-muted">
                    {fmtDate(rep.uploadedAt)} · {rep.observations.length} values
                  </p>
                </div>
                {rep.flags.length > 0 && <Badge tone="critical">critical</Badge>}
                <Badge tone="neutral">{rep.status}</Badge>
              </GlassCard>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

/* --------------------------------------------------------- availability */
export function DoctorAvailability() {
  type Slot = { id: number; date: string; start: string; end: string; label: string };
  const [slots, setSlots] = useState<Slot[]>([
    { id: 1, date: "2026-09-19", start: "16:00", end: "18:00", label: "Evening clinic" },
    { id: 2, date: "2026-09-20", start: "09:00", end: "12:00", label: "Morning" },
    { id: 3, date: "2026-09-22", start: "16:00", end: "18:00", label: "Evening clinic" },
  ]);
  const [draft, setDraft] = useState<Omit<Slot, "id">>({
    date: "",
    start: "09:00",
    end: "12:00",
    label: "",
  });
  return (
    <>
      <PageHead
        title="Availability"
        lead="Publish the windows patients can book. 20-minute slots are generated automatically."
      />
      <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
        <Reveal>
          <GlassCard strong>
            <h3 className="text-lg font-bold">Add a window</h3>
            <form
              className="mt-4 space-y-3"
              onSubmit={(e) => {
                e.preventDefault();
                if (!draft.date) return;
                setSlots((s) => [...s, { ...draft, id: Date.now() }]);
                setDraft({ date: "", start: "09:00", end: "12:00", label: "" });
              }}
            >
              <Input
                label="Date"
                type="date"
                name="date"
                value={draft.date}
                onChange={(e) => setDraft({ ...draft, date: e.target.value })}
              />
              <div className="grid grid-cols-2 gap-3">
                <Input
                  label="From"
                  type="time"
                  name="start"
                  value={draft.start}
                  onChange={(e) => setDraft({ ...draft, start: e.target.value })}
                />
                <Input
                  label="To"
                  type="time"
                  name="end"
                  value={draft.end}
                  onChange={(e) => setDraft({ ...draft, end: e.target.value })}
                />
              </div>
              <Input
                label="Label (optional)"
                name="label"
                placeholder="Morning clinic"
                value={draft.label}
                onChange={(e) => setDraft({ ...draft, label: e.target.value })}
              />
              <Button type="submit" icon={Plus} className="w-full">
                Add window
              </Button>
            </form>
            <div className="mt-6 rounded-2xl bg-brand-50 p-4 text-xs text-brand-900">
              Overlapping windows on the same day are rejected. Windows in the past are hidden from
              patients.
            </div>
          </GlassCard>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="space-y-3">
            {slots
              .sort((a, b) => a.date.localeCompare(b.date))
              .map((s) => (
                <motion.div
                  key={s.id}
                  layout
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <GlassCard className="flex flex-wrap items-center gap-4">
                    <div className="rounded-xl bg-brand-900 px-3 py-2 text-center text-white">
                      <p className="text-[10px] font-bold uppercase text-brand-300">
                        {new Date(s.date).toLocaleDateString("en", { weekday: "short" })}
                      </p>
                      <p className="font-display text-xl font-bold leading-none">
                        {new Date(s.date).getDate()}
                      </p>
                    </div>
                    <div className="flex-1">
                      <p className="font-bold">
                        {fmtTime(s.start)} – {fmtTime(s.end)}
                      </p>
                      <p className="text-xs text-ink-muted">
                        {s.label || "Unlabelled"} ·{" "}
                        {Math.floor(
                          ((Number(s.end.slice(0, 2)) - Number(s.start.slice(0, 2))) * 60) / 20,
                        )}{" "}
                        slots
                      </p>
                    </div>
                    <Badge tone="brand">Active</Badge>
                    <button
                      onClick={() => setSlots((x) => x.filter((y) => y.id !== s.id))}
                      className="grid size-9 place-items-center rounded-full text-ink-muted hover:bg-red-50 hover:text-red-600"
                      aria-label="Remove"
                    >
                      <Trash2 className="size-4" />
                    </button>
                  </GlassCard>
                </motion.div>
              ))}
          </div>
        </Reveal>
      </div>
    </>
  );
}

/* ---------------------------------------------------------- profile */
export function DoctorProfile() {
  return (
    <>
      <PageHead title="Profile" lead="What patients see when they find you." />
      <div className="grid gap-6 lg:grid-cols-[1fr_1.6fr]">
        <Reveal>
          <GlassCard strong className="text-center">
            <Avatar name={meDoc.name} size={96} className="mx-auto text-3xl" />
            <h2 className="mt-4 flex items-center justify-center gap-2 text-2xl font-bold">
              {meDoc.name} <BadgeCheck className="size-5 text-brand-600" />
            </h2>
            <p className="font-semibold text-brand-700">
              {specializationLabel[meDoc.specialization]}
            </p>
            <p className="text-xs text-ink-muted">{meDoc.hospital}</p>
            <div className="mt-6 grid grid-cols-3 gap-2 text-center">
              {[
                ["Rating", meDoc.rating],
                ["Reviews", meDoc.reviews],
                ["Years", meDoc.experience],
              ].map(([k, v]) => (
                <div key={k as string} className="rounded-2xl bg-white/70 p-3">
                  <p className="font-display text-xl font-bold">{v}</p>
                  <p className="text-[10px] font-bold uppercase text-ink-muted">{k}</p>
                </div>
              ))}
            </div>
          </GlassCard>
        </Reveal>
        <Reveal delay={0.05}>
          <GlassCard>
            <form className="grid gap-4 sm:grid-cols-2" onSubmit={(e) => e.preventDefault()}>
              <Input label="Full name" name="name" defaultValue={meDoc.name} />
              <Input
                label="Specialization"
                name="spec"
                defaultValue={specializationLabel[meDoc.specialization]}
              />
              <Input label="Hospital / clinic" name="hospital" defaultValue={meDoc.hospital} />
              <Input
                label="Consultation fee (Rs.)"
                name="fee"
                type="number"
                defaultValue={meDoc.fee}
              />
              <Input
                label="Years of experience"
                name="exp"
                type="number"
                defaultValue={meDoc.experience}
              />
              <Input
                label="NMC number"
                name="nmc"
                defaultValue="NMC-12345"
                disabled
                hint="Locked after verification."
              />
              <div className="sm:col-span-2">
                <Textarea label="Bio" name="bio" defaultValue={meDoc.bio} />
              </div>
              <div className="sm:col-span-2">
                <Button type="submit">Save profile</Button>
              </div>
            </form>
          </GlassCard>
        </Reveal>
      </div>
    </>
  );
}

/* ------------------------------------------------------- verification */
export function DoctorVerify() {
  const [status] = useState<"UNVERIFIED" | "PENDING" | "VERIFIED" | "REJECTED">("PENDING");
  const steps = [
    { t: "Account created", done: true },
    { t: "Licence uploaded", done: true },
    { t: "Admin review", done: status === "VERIFIED", active: status === "PENDING" },
    { t: "Clinical features unlocked", done: status === "VERIFIED" },
  ];
  return (
    <>
      <PageHead
        title="Verification"
        lead="Clinical features unlock once an administrator confirms your licence."
      />
      <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
        <Reveal>
          <GlassCard strong>
            <div className="flex items-center gap-3">
              <span
                className={cn(
                  "grid size-12 place-items-center rounded-2xl",
                  status === "VERIFIED" ? "bg-brand-600 text-white" : "bg-amber-100 text-amber-700",
                )}
              >
                {status === "VERIFIED" ? (
                  <BadgeCheck className="size-6" />
                ) : (
                  <Clock className="size-6" />
                )}
              </span>
              <div>
                <p className="text-xs font-bold uppercase tracking-wider text-ink-muted">Status</p>
                <p className="text-xl font-bold">
                  {status === "PENDING" ? "Under review" : status}
                </p>
              </div>
            </div>
            <ol className="mt-6 space-y-3">
              {steps.map((s, i) => (
                <li key={s.t} className="flex items-center gap-3">
                  <span
                    className={cn(
                      "grid size-7 place-items-center rounded-full text-xs font-bold",
                      s.done
                        ? "bg-brand-600 text-white"
                        : s.active
                          ? "bg-amber-100 text-amber-800 ring-2 ring-amber-400"
                          : "bg-white/70 text-ink-muted",
                    )}
                  >
                    {s.done ? "✓" : i + 1}
                  </span>
                  <span
                    className={cn(
                      "text-sm",
                      s.done || s.active ? "font-semibold" : "text-ink-muted",
                    )}
                  >
                    {s.t}
                  </span>
                </li>
              ))}
            </ol>
            <p className="mt-6 text-xs text-ink-muted">
              Typical review time: 1–2 working days. You will be emailed either way.
            </p>
          </GlassCard>
        </Reveal>
        <Reveal delay={0.1}>
          <GlassCard>
            <h3 className="flex items-center gap-2 text-lg font-bold">
              <ShieldCheck className="size-5 text-brand-600" /> Documents
            </h3>
            <div className="mt-4 space-y-3">
              {[
                ["NMC registration certificate", "nmc_certificate.pdf", "Uploaded 16 Sep"],
                ["Government ID", "citizenship.jpg", "Uploaded 16 Sep"],
              ].map(([t, f, s]) => (
                <div key={t} className="flex items-center gap-3 rounded-2xl bg-white/70 p-4">
                  <FileText className="size-5 text-brand-600" />
                  <div className="flex-1">
                    <p className="text-sm font-bold">{t}</p>
                    <p className="text-xs text-ink-muted">
                      {f} · {s}
                    </p>
                  </div>
                  <button
                    className="grid size-8 place-items-center rounded-full text-ink-muted hover:bg-red-50 hover:text-red-600"
                    aria-label="Remove"
                  >
                    <X className="size-4" />
                  </button>
                </div>
              ))}
              <label className="flex cursor-pointer items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-brand-200 p-5 text-sm font-semibold text-brand-800 hover:bg-brand-50">
                <FileUp className="size-4" /> Add supporting document
                <input type="file" className="sr-only" />
              </label>
            </div>
            <Input className="mt-4" label="Licence number" name="lic" defaultValue="NMC-12345" />
            <Button className="mt-4" icon={Stethoscope}>
              Resubmit for review
            </Button>
          </GlassCard>
        </Reveal>
      </div>
    </>
  );
}
