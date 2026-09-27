import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowRight,
  BookOpen,
  Check,
  CheckCircle2,
  Download,
  FileText,
  FileUp,
  Image as ImageIcon,
  MessageSquareText,
  Pencil,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  X,
} from "lucide-react";
import {
  Avatar,
  Badge,
  Button,
  EmptyState,
  GlassCard,
  PageHead,
  Table,
  Tabs,
  Td,
} from "@/components/ui";
import { Reveal } from "@/components/motion";
import { doctors, reports, specializationLabel, type Observation } from "@/mocks/data";
import { fmtDate } from "@/lib/format";
import { cn } from "@/lib/cn";
import { CriticalBanner, JobProgress, RiskCard, StatusBadge, inRange } from "./shared";

/* ------------------------------------------------------------ list */
export function ReportsList() {
  return (
    <>
      <PageHead
        title="My reports"
        lead="Every report you have uploaded, with its pipeline status."
        actions={
          <Button to="/reports/upload" icon={FileUp}>
            Upload
          </Button>
        }
      />
      <Table head={["Report", "Lab", "Uploaded", "Values", "Flags", "Status", ""]}>
        {reports.map((r) => (
          <tr key={r.id} className="transition hover:bg-white/50">
            <Td>
              <span className="font-mono text-xs font-semibold text-ink-soft">{r.id}</span>
            </Td>
            <Td>
              <span className="font-semibold">{r.lab}</span>
              <span className="block text-xs text-ink-muted">{r.fileName}</span>
            </Td>
            <Td>{fmtDate(r.uploadedAt)}</Td>
            <Td>{r.observations.length}</Td>
            <Td>
              {r.flags.length ? (
                <Badge tone="critical">{r.flags.length}</Badge>
              ) : (
                <span className="text-ink-muted">—</span>
              )}
            </Td>
            <Td>
              <StatusBadge status={r.status} />
            </Td>
            <Td className="text-right">
              <Button
                size="sm"
                variant="glass"
                to={
                  r.status === "AWAITING_REVIEW"
                    ? `/reports/${r.id}/review`
                    : `/reports/${r.id}/result`
                }
                icon={ArrowRight}
                trailing
              >
                {r.status === "AWAITING_REVIEW" ? "Review" : "Open"}
              </Button>
            </Td>
          </tr>
        ))}
      </Table>
    </>
  );
}

/* ---------------------------------------------------------- upload */
export function UploadReport() {
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [phase, setPhase] = useState<"idle" | "uploading" | "extracting">("idle");
  const nav = useNavigate();

  useEffect(() => {
    if (phase === "uploading") {
      const t = setTimeout(() => setPhase("extracting"), 1200);
      return () => clearTimeout(t);
    }
    if (phase === "extracting") {
      const t = setTimeout(() => nav("/reports/r1051/review"), 2200);
      return () => clearTimeout(t);
    }
  }, [phase, nav]);

  return (
    <>
      <PageHead
        title="Upload a report"
        lead="PDF, JPG or PNG up to 10 MB. You will verify every extracted value before anything is analysed."
      />
      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <Reveal>
          <GlassCard strong className="p-8">
            <AnimatePresence mode="wait">
              {phase === "idle" ? (
                <motion.div
                  key="drop"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <label
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDrag(true);
                    }}
                    onDragLeave={() => setDrag(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDrag(false);
                      setFile(e.dataTransfer.files[0] ?? null);
                    }}
                    className={cn(
                      "relative flex min-h-72 cursor-pointer flex-col items-center justify-center rounded-xl3 border-2 border-dashed p-8 text-center transition",
                      drag
                        ? "border-brand-500 bg-brand-50"
                        : "border-brand-200 bg-white/50 hover:bg-white/80",
                    )}
                  >
                    <input
                      type="file"
                      className="sr-only"
                      accept=".pdf,image/*"
                      onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                    />
                    <motion.span
                      animate={{ y: [0, -6, 0] }}
                      transition={{ repeat: Infinity, duration: 3 }}
                      className="grid size-16 place-items-center rounded-2xl bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-glow"
                    >
                      <FileUp className="size-7" />
                    </motion.span>
                    <p className="mt-6 text-lg font-bold">
                      {file ? file.name : "Drop your report here"}
                    </p>
                    <p className="mt-1 text-sm text-ink-soft">
                      {file ? `${(file.size / 1024).toFixed(0)} KB · ready` : "or click to browse"}
                    </p>
                    <div className="mt-5 flex gap-2">
                      <span className="glass-pill inline-flex items-center gap-1.5 px-3 py-1 text-xs font-semibold">
                        <FileText className="size-3.5" /> PDF
                      </span>
                      <span className="glass-pill inline-flex items-center gap-1.5 px-3 py-1 text-xs font-semibold">
                        <ImageIcon className="size-3.5" /> Photo
                      </span>
                    </div>
                  </label>
                  <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
                    <label className="flex items-center gap-2 text-sm text-ink-soft">
                      <input type="checkbox" defaultChecked className="size-4 accent-brand-600" /> I
                      consent to processing this report (see privacy)
                    </label>
                    <div className="flex gap-2">
                      <Button
                        variant="glass"
                        onClick={() => setFile(new File(["x"], "Report_Sep2026.pdf"))}
                      >
                        Use sample
                      </Button>
                      <Button
                        disabled={!file}
                        onClick={() => setPhase("uploading")}
                        icon={ArrowRight}
                        trailing
                      >
                        Upload & extract
                      </Button>
                    </div>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="prog"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex min-h-72 flex-col items-center justify-center text-center"
                >
                  <div className="relative">
                    <motion.span
                      className="absolute inset-0 rounded-full bg-brand-400/40"
                      animate={{ scale: [1, 1.8], opacity: [0.7, 0] }}
                      transition={{ repeat: Infinity, duration: 1.6 }}
                    />
                    <span className="relative grid size-20 place-items-center rounded-full bg-gradient-to-br from-brand-500 to-brand-700 text-white">
                      <Sparkles className="size-8" />
                    </span>
                  </div>
                  <p className="mt-8 text-xl font-bold">
                    {phase === "uploading" ? "Uploading securely…" : "Reading your report…"}
                  </p>
                  <p className="mt-1 text-sm text-ink-soft">
                    {phase === "uploading"
                      ? "Encrypted, stored behind a signed link."
                      : "Finding every test name, value, unit and reference range."}
                  </p>
                  <div className="mt-8 w-full max-w-md">
                    <JobProgress status={phase === "uploading" ? "QUEUED" : "EXTRACTING"} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </GlassCard>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="space-y-4">
            {[
              {
                icon: ShieldCheck,
                t: "Private by default",
                b: "Your file is encrypted at rest and only ever served through a short-lived link.",
              },
              {
                icon: Pencil,
                t: "You stay in control",
                b: "Nothing is analysed until you have confirmed the extracted values.",
              },
              {
                icon: BookOpen,
                t: "Any Nepali lab",
                b: "We learn layouts from Bir, Civil Service, Grande, Medicity and more.",
              },
            ].map((c) => (
              <GlassCard key={c.t} className="flex gap-4">
                <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-brand-100 text-brand-700">
                  <c.icon className="size-4" />
                </span>
                <div>
                  <p className="font-bold">{c.t}</p>
                  <p className="mt-0.5 text-sm text-ink-soft">{c.b}</p>
                </div>
              </GlassCard>
            ))}
          </div>
        </Reveal>
      </div>
    </>
  );
}

/* ---------------------------------------------------------- review */
export function ReviewReport() {
  const { id } = useParams();
  const src = reports.find((r) => r.id === id) ?? reports[2];
  const [rows, setRows] = useState<Observation[]>(src.observations);
  const [editing, setEditing] = useState<string | null>(null);
  const [analysing, setAnalysing] = useState(false);
  const nav = useNavigate();

  const lowConf = rows.filter((r) => (r.confidence ?? 1) < 0.75).length;
  const pending = rows.filter((r) => r.suggestion).length;

  useEffect(() => {
    if (!analysing) return;
    const t = setTimeout(() => nav("/reports/r1042/result"), 2600);
    return () => clearTimeout(t);
  }, [analysing, nav]);

  const accept = (loinc: string) =>
    setRows((rs) =>
      rs.map((r) =>
        r.loinc === loinc && r.suggestion
          ? {
              ...r,
              value: r.suggestion.proposed,
              source: "MANUAL",
              confidence: 1,
              suggestion: undefined,
            }
          : r,
      ),
    );
  const reject = (loinc: string) =>
    setRows((rs) => rs.map((r) => (r.loinc === loinc ? { ...r, suggestion: undefined } : r)));
  const setValue = (loinc: string, v: number) =>
    setRows((rs) =>
      rs.map((r) =>
        r.loinc === loinc
          ? { ...r, value: v, source: "MANUAL", confidence: 1, suggestion: undefined }
          : r,
      ),
    );

  return (
    <>
      <PageHead
        crumbs={["Reports", src.id, "Review"]}
        title="Verify the extracted values"
        lead="Low-confidence fields are highlighted. Accept suggestions, edit anything that is wrong, then run the analysis."
        actions={
          <Button size="lg" onClick={() => setAnalysing(true)} icon={Sparkles} disabled={analysing}>
            {analysing ? "Analysing…" : "Confirm & analyse"}
          </Button>
        }
      />
      <GlassCard className="mb-6">
        <JobProgress status={analysing ? "EVALUATING" : "AWAITING_REVIEW"} />
      </GlassCard>
      <div className="grid gap-6 lg:grid-cols-[1fr_1.5fr]">
        <Reveal>
          <GlassCard strong className="sticky top-24 p-4">
            <p className="mb-3 flex items-center justify-between text-xs font-bold uppercase tracking-wider text-ink-muted">
              Page 1 of {src.pages}{" "}
              <span className="normal-case tracking-normal">{src.fileName}</span>
            </p>
            {/* faux document with highlights */}
            <div className="relative aspect-[3/4] overflow-hidden rounded-2xl border border-line bg-white p-6 text-[10px] leading-relaxed text-ink/70 shadow-inner">
              <p className="text-center font-display text-sm font-bold text-ink">
                {src.lab.toUpperCase()}
              </p>
              <p className="text-center text-[9px]">Haematology & Biochemistry Report</p>
              <div className="mt-4 space-y-1.5">
                {rows.map((o) => {
                  const low = (o.confidence ?? 1) < 0.75;
                  return (
                    <div
                      key={o.loinc}
                      className={cn(
                        "flex justify-between rounded px-1.5 py-0.5",
                        low && "bg-amber-100 ring-1 ring-amber-400",
                        editing === o.loinc && "bg-brand-100 ring-1 ring-brand-500",
                      )}
                    >
                      <span>{o.analyte}</span>
                      <span className="font-semibold">
                        {o.value} {o.unit}
                      </span>
                      <span className="text-ink-muted">
                        {o.refLow ?? ""}–{o.refHigh ?? ""}
                      </span>
                    </div>
                  );
                })}
              </div>
              <p className="absolute bottom-4 left-0 right-0 text-center text-[9px] text-ink-muted">
                Highlights show where each value was read (bbox provenance).
              </p>
            </div>
          </GlassCard>
        </Reveal>

        <div>
          <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
            <Badge tone="warn">{lowConf} low confidence</Badge>
            <Badge tone="info">
              {pending} suggestion{pending !== 1 && "s"}
            </Badge>
            <Badge tone="brand">{rows.filter((r) => r.source === "MANUAL").length} edited</Badge>
          </div>
          <div className="space-y-2">
            {rows.map((o, i) => {
              const conf = o.confidence ?? 1;
              const low = conf < 0.75;
              const range = inRange(o);
              return (
                <motion.div
                  key={o.loinc}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.03 }}
                  onMouseEnter={() => setEditing(o.loinc)}
                  onMouseLeave={() => setEditing(null)}
                  className={cn("glass p-4 transition", low && "border-amber-300 bg-amber-50/60")}
                >
                  <div className="flex flex-wrap items-center gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <p className="font-bold">{o.analyte}</p>
                        <span className="rounded bg-white/70 px-1.5 py-0.5 font-mono text-[10px] text-ink-muted">
                          {o.loinc}
                        </span>
                        <Badge tone="neutral">{o.panel}</Badge>
                      </div>
                      <p className="mt-0.5 text-xs text-ink-muted">
                        Ref {o.refLow ?? "—"} – {o.refHigh ?? "—"} {o.unit} · p.{o.page} ·{" "}
                        {o.source === "MANUAL" ? "edited by you" : `OCR ${Math.round(conf * 100)}%`}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        step="any"
                        value={o.value}
                        onChange={(e) => setValue(o.loinc, Number(e.target.value))}
                        className={cn(
                          "h-10 w-28 rounded-xl border bg-white px-3 text-right font-display text-lg font-bold ring-focus",
                          range !== "normal"
                            ? "border-amber-300 text-amber-800"
                            : "border-white/80",
                        )}
                      />
                      <span className="w-14 text-xs text-ink-muted">{o.unit}</span>
                    </div>
                  </div>
                  {o.suggestion && (
                    <div className="mt-3 flex flex-wrap items-center gap-3 rounded-2xl bg-white/80 p-3">
                      <Sparkles className="size-4 text-brand-600" />
                      <p className="flex-1 text-sm">
                        Suggest <span className="font-bold">{o.suggestion.proposed}</span> —{" "}
                        <span className="text-ink-soft">{o.suggestion.reason}</span>
                      </p>
                      <div className="flex gap-1.5">
                        <button
                          onClick={() => accept(o.loinc)}
                          className="grid size-8 place-items-center rounded-full bg-brand-600 text-white hover:bg-brand-700"
                          aria-label="Accept"
                        >
                          <Check className="size-4" />
                        </button>
                        <button
                          onClick={() => reject(o.loinc)}
                          className="grid size-8 place-items-center rounded-full bg-white text-ink-muted hover:bg-brand-50"
                          aria-label="Reject"
                        >
                          <X className="size-4" />
                        </button>
                      </div>
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
          <p className="mt-4 text-xs text-ink-muted">
            Suggestions are never applied automatically. Safety rules run on the values you confirm.
          </p>
        </div>
      </div>
    </>
  );
}

/* ---------------------------------------------------------- result */
export function ReportResult() {
  const { id } = useParams();
  const r = reports.find((x) => x.id === id && x.status === "DONE") ?? reports[0];
  const [audience, setAudience] = useState<"patient" | "clinician">("patient");
  const [panel, setPanel] = useState<"ALL" | Observation["panel"]>("ALL");
  const panels = useMemo(() => Array.from(new Set(r.observations.map((o) => o.panel))), [r]);
  const shown = r.observations.filter((o) => panel === "ALL" || o.panel === panel);
  const specialists = Array.from(
    new Set(
      r.assessments
        .filter((a) => a.status === "ASSESSED" && (a.probability ?? 0) >= (a.threshold ?? 1) * 0.8)
        .map((a) => a.specialist),
    ),
  );
  const suggested = doctors
    .filter((d) => specialists.includes(d.specialization) && d.status === "VERIFIED")
    .slice(0, 3);

  return (
    <>
      <PageHead
        crumbs={["Reports", r.id]}
        title={r.lab}
        lead={`Collected ${fmtDate(r.collectedAt)} · uploaded ${fmtDate(r.uploadedAt)} · ${
          r.observations.length
        } values across ${panels.length} panels`}
        actions={
          <>
            <Button variant="glass" icon={Download}>
              PDF
            </Button>
            <Button to="/doctors" icon={Stethoscope}>
              Book a doctor
            </Button>
          </>
        }
      />
      <div className="space-y-6">
        <CriticalBanner flags={r.flags} />
        {!r.flags.length && (
          <Reveal>
            <div className="flex items-center gap-3 rounded-xl3 border border-brand-200 bg-brand-50/80 p-4 text-sm text-brand-900">
              <CheckCircle2 className="size-5 text-brand-600" /> No critical values. All safety
              rules passed on your confirmed values.
            </div>
          </Reveal>
        )}

        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-xl font-bold">Risk assessment</h2>
            <p className="text-xs text-ink-muted">
              {r.assessments.filter((a) => a.status === "ASSESSED").length} assessed ·{" "}
              {r.assessments.filter((a) => a.status === "NOT_ASSESSABLE").length} not assessable
            </p>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {r.assessments.map((a, i) => (
              <RiskCard key={a.condition} a={a} index={i} />
            ))}
          </div>
        </section>

        <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <Reveal>
            <GlassCard strong className="h-full">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-xl font-bold">Explanation</h2>
                <Tabs
                  tabs={[
                    { id: "patient", label: "For me" },
                    { id: "clinician", label: "For my doctor" },
                  ]}
                  value={audience}
                  onChange={setAudience}
                />
              </div>
              <AnimatePresence mode="wait">
                <motion.p
                  key={audience}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  className="mt-5 leading-relaxed text-ink"
                >
                  {audience === "patient" ? r.explanation.patient : r.explanation.clinician}
                </motion.p>
              </AnimatePresence>
              <div className="mt-5 flex flex-wrap gap-2">
                {r.explanation.citations.map((c) => (
                  <span
                    key={c.doc}
                    className="inline-flex items-center gap-1.5 rounded-full bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-800"
                  >
                    <BookOpen className="size-3.5" /> {c.doc} · p.{c.page}
                  </span>
                ))}
                <span className="inline-flex items-center gap-1.5 rounded-full bg-white/80 px-3 py-1 text-xs font-semibold text-ink-soft">
                  <ShieldCheck className="size-3.5 text-brand-600" /> every number validated against
                  this report
                </span>
              </div>
              <p className="mt-4 text-xs text-ink-muted">
                MediScan supports decisions; it does not make diagnoses. Always consult a doctor.
              </p>
            </GlassCard>
          </Reveal>
          <Reveal delay={0.1}>
            <GlassCard className="h-full">
              <h2 className="flex items-center gap-2 text-xl font-bold">
                <MessageSquareText className="size-5 text-brand-600" /> Doctor's comment
              </h2>
              {r.doctorComment ? (
                <div className="mt-4">
                  <div className="flex items-center gap-3">
                    <Avatar name={r.doctorComment.doctor} size={40} />
                    <div>
                      <p className="text-sm font-bold">{r.doctorComment.doctor}</p>
                      <p className="text-xs text-ink-muted">{fmtDate(r.doctorComment.at)}</p>
                    </div>
                  </div>
                  <p className="mt-4 rounded-2xl bg-white/80 p-4 text-sm leading-relaxed">
                    {r.doctorComment.text}
                  </p>
                </div>
              ) : (
                <EmptyState
                  icon={Stethoscope}
                  title="No comment yet"
                  body="Book a consultation and your doctor can annotate this report."
                  action={
                    <Button to="/doctors" size="sm">
                      Find a doctor
                    </Button>
                  }
                />
              )}
            </GlassCard>
          </Reveal>
        </div>

        <section>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-xl font-bold">Your values</h2>
            <div className="glass-pill flex flex-wrap p-1">
              {(["ALL", ...panels] as const).map((p) => (
                <button
                  key={p}
                  onClick={() => setPanel(p)}
                  className={cn(
                    "rounded-full px-3 py-1.5 text-xs font-semibold transition",
                    panel === p ? "bg-brand-600 text-white" : "text-ink-soft hover:text-ink",
                  )}
                >
                  {p === "ALL" ? "All" : p}
                </button>
              ))}
            </div>
          </div>
          <Table head={["Analyte", "Value", "Reference", "Status", "Source"]}>
            {shown.map((o) => {
              const s = inRange(o);
              return (
                <tr key={o.loinc} className="transition hover:bg-white/50">
                  <Td>
                    <span className="font-semibold">{o.analyte}</span>
                    <span className="ml-2 font-mono text-[10px] text-ink-muted">{o.loinc}</span>
                  </Td>
                  <Td>
                    <span className="font-display text-base font-bold">
                      {o.value.toLocaleString()}
                    </span>{" "}
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
                      ? "Verified by you"
                      : `OCR ${Math.round((o.confidence ?? 1) * 100)}%`}
                  </Td>
                </tr>
              );
            })}
          </Table>
        </section>

        {suggested.length > 0 && (
          <section>
            <h2 className="mb-3 text-xl font-bold">Suggested specialists</h2>
            <div className="grid gap-4 md:grid-cols-3">
              {suggested.map((d) => (
                <Link
                  key={d.id}
                  to={`/doctors/${d.id}`}
                  className="glass flex items-center gap-4 p-4 transition hover:-translate-y-0.5 hover:shadow-glow"
                >
                  <Avatar name={d.name} size={48} />
                  <div className="min-w-0">
                    <p className="truncate font-bold">{d.name}</p>
                    <p className="text-xs text-brand-700 font-semibold">
                      {specializationLabel[d.specialization]}
                    </p>
                    <p className="text-xs text-ink-muted">{d.nextSlot}</p>
                  </div>
                  <ArrowRight className="ml-auto size-4 text-ink-muted" />
                </Link>
              ))}
            </div>
          </section>
        )}
      </div>
    </>
  );
}
