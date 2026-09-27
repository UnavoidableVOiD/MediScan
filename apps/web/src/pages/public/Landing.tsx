import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { motion, useScroll, useTransform } from "framer-motion";
import {
  ArrowRight,
  BadgeCheck,
  BookOpenCheck,
  Brain,
  CheckCircle2,
  FileSearch,
  HeartPulse,
  Languages,
  Lock,
  ScanLine,
  ShieldAlert,
  Sparkles,
  Stethoscope,
  UploadCloud,
  Activity,
  ChevronDown,
  Droplets,
  FlaskConical,
  HeartHandshake,
  TrendingUp,
} from "lucide-react";
import { HeroScene } from "@/components/three/HeroScene";
import { Badge, Button, Eyebrow, GlassCard, SectionHeading } from "@/components/ui";
import { Counter, Marquee, PageTransition, Reveal, item, stagger } from "@/components/motion";

const steps = [
  {
    icon: UploadCloud,
    title: "Upload",
    body: "A PDF or a photo of your lab report. Any lab, any layout.",
  },
  {
    icon: ScanLine,
    title: "We read it",
    body: "Every value is extracted with its unit and the lab's own reference range.",
  },
  {
    icon: CheckCircle2,
    title: "You confirm",
    body: "Low-confidence values are highlighted. You verify before anything is analysed.",
  },
  {
    icon: ShieldAlert,
    title: "Safety first",
    body: "Critical values trigger an alert before any model runs — deterministic, guideline-sourced.",
  },
  {
    icon: Brain,
    title: "Risk assessed",
    body: "Versioned models per condition. If a test is missing we say 'not assessable' — never guess.",
  },
  {
    icon: Stethoscope,
    title: "Talk to a doctor",
    body: "A plain-language explanation, then a verified specialist matched to your results.",
  },
];

const conditions = [
  "Anemia",
  "Chronic kidney disease",
  "Liver function",
  "Thyroid",
  "Glycaemic risk",
  "Lipids & heart",
  "Critical values",
  "Trend tracking",
];

const coverage = [
  {
    icon: Droplets,
    title: "Anemia",
    panel: "CBC",
    body: "Hemoglobin, MCV, MCH, MCHC read against WHO sex-specific thresholds.",
  },
  {
    icon: Activity,
    title: "Kidney (CKD)",
    panel: "RFT",
    body: "Creatinine, urea, sodium, potassium — plus eGFR when age and sex are known.",
  },
  {
    icon: FlaskConical,
    title: "Liver",
    panel: "LFT",
    body: "Bilirubin, ALT, AST, ALP, albumin and protein for hepatic dysfunction.",
  },
  {
    icon: TrendingUp,
    title: "Thyroid",
    panel: "TFT",
    body: "TSH with free or total T4 — we never mix the two.",
  },
  {
    icon: HeartHandshake,
    title: "Glycaemic risk",
    panel: "Glucose",
    body: "Fasting glucose and HbA1c against ADA / WHO cut-offs, rules first.",
  },
  {
    icon: HeartPulse,
    title: "Lipids & heart",
    panel: "Lipid",
    body: "Cholesterol, HDL, LDL, triglycerides — flagged honestly as not assessable when BP is missing.",
  },
];

const faqs = [
  {
    q: "Is MediScan a diagnosis?",
    a: "No. MediScan is decision support. It reads, checks and explains your report, then connects you to a verified doctor. The doctor decides.",
  },
  {
    q: "What if my report is a phone photo?",
    a: "That is the normal case in Nepal. We de-skew and enhance the image, read every line, and show you a confidence score per value so you can correct anything uncertain before analysis.",
  },
  {
    q: "Why does it say 'not assessable' sometimes?",
    a: "Because your report did not contain a test that condition needs. We would rather tell you what to test next than invent a number.",
  },
  {
    q: "Who can see my report?",
    a: "You, and a doctor you choose to book or share with. Every access is logged. When a language model is used, your name and identifiers never leave the platform.",
  },
  {
    q: "Which labs do you support?",
    a: "Any. We have learned layouts from Bir Hospital, Civil Service Hospital, Grande, Medicity and many smaller diagnostic centres, and every new format you upload teaches us more.",
  },
];

function FaqItem({ q, a }: { q: string; a: string }) {
  const [open, setOpen] = useState(false);
  return (
    <button
      onClick={() => setOpen((o) => !o)}
      className="glass w-full p-5 text-left transition hover:bg-white/80"
    >
      <span className="flex items-center justify-between gap-4">
        <span className="font-bold">{q}</span>
        <ChevronDown
          className={`size-4 shrink-0 text-brand-700 transition-transform ${
            open ? "rotate-180" : ""
          }`}
        />
      </span>
      <motion.div
        initial={false}
        animate={{ height: open ? "auto" : 0, opacity: open ? 1 : 0 }}
        className="overflow-hidden"
      >
        <p className="pt-3 text-sm leading-relaxed text-ink-soft">{a}</p>
      </motion.div>
    </button>
  );
}

export default function Landing() {
  const heroRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ["start start", "end start"] });
  const sceneY = useTransform(scrollYProgress, [0, 1], [0, 160]);
  const sceneOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0.15]);
  const textY = useTransform(scrollYProgress, [0, 1], [0, -80]);

  return (
    <PageTransition>
      {/* ------------------------------------------------------------ HERO */}
      <section ref={heroRef} className="relative min-h-[100svh] overflow-hidden pt-28">
        <motion.div
          style={{ y: sceneY, opacity: sceneOpacity }}
          className="pointer-events-none absolute bottom-0 right-0 top-20 hidden w-[58%] lg:block"
        >
          <HeroScene className="h-full w-full" />
        </motion.div>
        <div className="pointer-events-none absolute inset-0 grid-dots opacity-70" />

        <motion.div
          style={{ y: textY }}
          className="relative mx-auto flex max-w-7xl flex-col px-4 pb-24 pt-10 sm:px-6 lg:pt-20"
        >
          <motion.div variants={stagger} initial="hidden" animate="show" className="max-w-2xl">
            <motion.div variants={item}>
              <span className="glass-pill inline-flex items-center gap-2 px-4 py-1.5 text-xs font-semibold text-brand-800">
                <Sparkles className="size-3.5 text-brand-600" />
                Built in Nepal, for real lab reports
              </span>
            </motion.div>
            <motion.h1
              variants={item}
              className="mt-6 text-5xl font-bold leading-[1.02] sm:text-6xl lg:text-7xl"
            >
              Your lab report,
              <br />
              <span className="text-gradient">finally understood.</span>
            </motion.h1>
            <motion.p
              variants={item}
              className="mt-6 max-w-xl text-lg leading-relaxed text-ink-soft sm:text-xl"
            >
              MediScan reads your report, checks it for anything urgent, assesses your risk with
              transparent models, and explains it all in language you can act on — then connects you
              to the right doctor.
            </motion.p>
            <motion.div variants={item} className="mt-10 flex flex-wrap items-center gap-3">
              <Button to="/reports/upload" size="lg" icon={UploadCloud}>
                Upload a report
              </Button>
              <Button to="/pages" size="lg" variant="glass" icon={ArrowRight} trailing>
                Explore every page
              </Button>
            </motion.div>
            <motion.div variants={item} className="mt-12 grid max-w-lg grid-cols-3 gap-4">
              {[
                ["Conditions", 6, ""],
                ["Analytes read", 40, "+"],
                ["Verified doctors", 46, ""],
              ].map(([label, n, suffix]) => (
                <div key={label as string} className="glass px-4 py-3">
                  <p className="font-display text-2xl font-bold text-brand-800">
                    <Counter to={n as number} suffix={suffix as string} />
                  </p>
                  <p className="text-[11px] font-bold uppercase tracking-wider text-ink-muted">
                    {label}
                  </p>
                </div>
              ))}
            </motion.div>
          </motion.div>

          {/* floating result preview */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9, duration: 1, ease: [0.16, 1, 0.3, 1] }}
            className="glass-strong animate-float absolute bottom-6 right-4 hidden w-80 p-5 lg:block"
          >
            <div className="flex items-center justify-between">
              <p className="text-xs font-bold uppercase tracking-wider text-ink-muted">
                Bir Hospital · 15 Sep
              </p>
              <Badge tone="brand" dot>
                Done
              </Badge>
            </div>
            <div className="mt-4 space-y-2.5">
              {[
                ["Kidney", "Low risk", 6],
                ["Liver", "Borderline", 31],
                ["Thyroid", "Euthyroid", 5],
              ].map(([k, v, p]) => (
                <div key={k as string}>
                  <div className="flex justify-between text-sm">
                    <span className="font-semibold">{k}</span>
                    <span className="text-ink-soft">{v}</span>
                  </div>
                  <div className="mt-1 h-1.5 w-full rounded-full bg-brand-900/10">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-brand-400 to-brand-600"
                      style={{ width: `${p}%` }}
                    />
                  </div>
                </div>
              ))}
              <div className="flex items-center justify-between rounded-xl bg-white/70 px-3 py-2 text-xs">
                <span className="font-semibold text-ink-soft">Heart</span>
                <span className="font-bold text-ink-muted">Not assessable · BP missing</span>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </section>

      <Marquee items={conditions} className="mx-auto max-w-7xl border-y border-line py-5" />

      {/* --------------------------------------------------------- PROBLEM */}
      <section className="mx-auto max-w-7xl px-4 py-28 sm:px-6">
        <div className="grid items-center gap-14 lg:grid-cols-2">
          <Reveal>
            <SectionHeading
              eyebrow="Why MediScan"
              title={
                <>
                  Reports are written for labs,{" "}
                  <span className="text-gradient">not for people.</span>
                </>
              }
              lead="A page of abbreviations and reference ranges tells you almost nothing about what to do next. Doctors are stretched thin. Generic AI chatbots guess, hallucinate and know nothing about the range printed on your specific report."
            />
            <div className="mt-8 flex flex-wrap gap-2">
              {[
                "Grounded in your lab's ranges",
                "No guessing on missing tests",
                "Every number validated",
                "Verified doctors",
              ].map((t) => (
                <span
                  key={t}
                  className="glass-pill px-3.5 py-1.5 text-xs font-semibold text-brand-800"
                >
                  {t}
                </span>
              ))}
            </div>
          </Reveal>
          <div className="grid gap-4 sm:grid-cols-2">
            {[
              {
                icon: FileSearch,
                title: "Reads any layout",
                body: "Nepali hospital formats, photos, scans — with per-field confidence.",
              },
              {
                icon: ShieldAlert,
                title: "Deterministic safety",
                body: "Critical values are rules, not predictions. They run first and are never edited away.",
              },
              {
                icon: Brain,
                title: "Honest models",
                body: "Versioned bundles with published metrics. Missing data → 'not assessable'.",
              },
              {
                icon: BookOpenCheck,
                title: "Cited explanations",
                body: "Plain language, grounded in WHO / KDIGO / ESC guidance, every figure checked.",
              },
            ].map((c, i) => (
              <Reveal key={c.title} delay={i * 0.08}>
                <GlassCard hover className="h-full">
                  <span className="grid size-11 place-items-center rounded-2xl bg-brand-100 text-brand-700">
                    <c.icon className="size-5" />
                  </span>
                  <h3 className="mt-5 text-lg font-bold">{c.title}</h3>
                  <p className="mt-1.5 text-sm leading-relaxed text-ink-soft">{c.body}</p>
                </GlassCard>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------- COVERAGE */}
      <section className="mx-auto max-w-7xl px-4 pb-28 sm:px-6">
        <Reveal>
          <SectionHeading
            eyebrow="What we read"
            title={
              <>
                Six conditions. <span className="text-gradient">Forty analytes.</span> One report.
              </>
            }
            lead="A standard Nepali blood panel — CBC, LFT, RFT, TFT, glucose and lipids — is enough for most of these. Each condition tells you exactly which tests it used."
          />
        </Reveal>
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {coverage.map((c, i) => (
            <Reveal key={c.title} delay={i * 0.06}>
              <GlassCard hover className="group h-full">
                <div className="flex items-start justify-between">
                  <span className="grid size-11 place-items-center rounded-2xl bg-brand-100 text-brand-700 transition group-hover:bg-brand-600 group-hover:text-white">
                    <c.icon className="size-5" />
                  </span>
                  <Badge tone="neutral">{c.panel}</Badge>
                </div>
                <h3 className="mt-5 text-lg font-bold">{c.title}</h3>
                <p className="mt-1.5 text-sm leading-relaxed text-ink-soft">{c.body}</p>
              </GlassCard>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------ HOW IT WORKS */}
      <section className="relative mx-auto max-w-7xl px-4 sm:px-6">
        <div className="glass-dark relative overflow-hidden px-6 py-20 sm:px-14">
          <div className="absolute -left-20 top-0 size-96 rounded-full bg-brand-500/25 blur-3xl" />
          <div className="absolute -right-20 bottom-0 size-96 rounded-full bg-teal/20 blur-3xl" />
          <Reveal>
            <SectionHeading
              dark
              eyebrow="How it works"
              align="center"
              title="Six steps from PDF to a plan."
              lead="The same pipeline every time — and you can see which step it is on."
            />
          </Reveal>
          <div className="relative mt-16 grid gap-5 md:grid-cols-2 lg:grid-cols-3">
            {steps.map((s, i) => (
              <Reveal key={s.title} delay={i * 0.07}>
                <div className="group relative h-full rounded-xl3 border border-white/10 bg-white/5 p-6 transition hover:bg-white/10">
                  <div className="flex items-center justify-between">
                    <span className="grid size-11 place-items-center rounded-2xl bg-brand-500/20 text-brand-200 transition group-hover:bg-brand-400 group-hover:text-brand-950">
                      <s.icon className="size-5" />
                    </span>
                    <span className="font-display text-4xl font-bold text-white/10">0{i + 1}</span>
                  </div>
                  <h3 className="mt-6 text-lg font-bold text-white">{s.title}</h3>
                  <p className="mt-1.5 text-sm leading-relaxed text-white/65">{s.body}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------- PORTALS */}
      <section className="mx-auto max-w-7xl px-4 py-28 sm:px-6">
        <Reveal>
          <SectionHeading
            eyebrow="Built for everyone in the loop"
            title="Three portals, one source of truth."
            align="center"
          />
        </Reveal>
        <div className="mt-14 grid gap-5 lg:grid-cols-3">
          {[
            {
              to: "/dashboard",
              icon: HeartPulse,
              title: "Patients",
              body: "Upload, verify, understand, track trends over time and book the right specialist.",
              cta: "Open patient portal",
            },
            {
              to: "/doctor/dashboard",
              icon: Stethoscope,
              title: "Doctors",
              body: "A clinical summary, the raw values, the model's reasoning and your patient's history — in one view.",
              cta: "Open doctor portal",
            },
            {
              to: "/admin/dashboard",
              icon: BadgeCheck,
              title: "Administrators",
              body: "Verify licences, watch the platform's health and revenue, and keep every access audited.",
              cta: "Open admin portal",
            },
          ].map((p, i) => (
            <Reveal key={p.title} delay={i * 0.1}>
              <Link
                to={p.to}
                className="group glass block h-full p-8 transition-all duration-500 hover:-translate-y-1.5 hover:shadow-glow"
              >
                <span className="grid size-12 place-items-center rounded-2xl bg-gradient-to-br from-brand-500 to-brand-700 text-white">
                  <p.icon className="size-6" />
                </span>
                <h3 className="mt-6 text-2xl font-bold">{p.title}</h3>
                <p className="mt-2 leading-relaxed text-ink-soft">{p.body}</p>
                <span className="mt-6 inline-flex items-center gap-2 text-sm font-bold text-brand-700">
                  {p.cta} <ArrowRight className="size-4 transition group-hover:translate-x-1" />
                </span>
              </Link>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ---------------------------------------------------------- TRUST */}
      <section className="mx-auto max-w-7xl px-4 sm:px-6">
        <div className="grid gap-5 md:grid-cols-3">
          {[
            {
              icon: Lock,
              title: "Private by design",
              body: "Files live in encrypted storage behind signed links. Every view is audited. Names never reach the language model.",
            },
            {
              icon: Languages,
              title: "Made for Nepal",
              body: "Trained on local report layouts; Nepali and Hindi explanations on the roadmap.",
            },
            {
              icon: BadgeCheck,
              title: "Not a doctor — a bridge",
              body: "MediScan supports decisions. Every result ends with a real clinician.",
            },
          ].map((t, i) => (
            <Reveal key={t.title} delay={i * 0.08}>
              <GlassCard className="h-full">
                <Eyebrow className="mb-4">{t.title}</Eyebrow>
                <p className="text-sm leading-relaxed text-ink-soft">{t.body}</p>
                <t.icon className="mt-6 size-8 text-brand-300" />
              </GlassCard>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------ FAQ */}
      <section className="mx-auto max-w-7xl px-4 pt-28 sm:px-6">
        <div className="grid gap-10 lg:grid-cols-[1fr_1.4fr]">
          <Reveal>
            <SectionHeading
              eyebrow="Questions"
              title={
                <>
                  Straight answers, <span className="text-gradient">no fine print.</span>
                </>
              }
              lead="The things people ask us before they upload their first report."
            />
            <Button to="/contact" variant="glass" className="mt-8" icon={ArrowRight} trailing>
              Ask something else
            </Button>
          </Reveal>
          <div className="space-y-3">
            {faqs.map((f, i) => (
              <Reveal key={f.q} delay={i * 0.05}>
                <FaqItem {...f} />
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------ CTA */}
      <section className="mx-auto max-w-7xl px-4 pt-28 sm:px-6">
        <Reveal>
          <div className="relative overflow-hidden rounded-xl3 bg-gradient-to-br from-brand-600 via-brand-700 to-brand-900 px-8 py-16 text-center text-white sm:px-16">
            <div className="absolute inset-0 grid-dots opacity-30 [--tw-mask:none]" />
            <h2 className="relative text-4xl font-bold sm:text-5xl">
              Ready to understand your next report?
            </h2>
            <p className="relative mx-auto mt-4 max-w-xl text-white/80">
              Upload a PDF or a photo. Verified values, honest risk, a real doctor — in minutes.
            </p>
            <div className="relative mt-8 flex flex-wrap justify-center gap-3">
              <Button to="/reports/upload" size="lg" variant="glass" className="text-brand-900">
                Upload a report
              </Button>
              <Button to="/doctors" size="lg" variant="secondary">
                Find a doctor
              </Button>
            </div>
          </div>
        </Reveal>
      </section>
    </PageTransition>
  );
}
