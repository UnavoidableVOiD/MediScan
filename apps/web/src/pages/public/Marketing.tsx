import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ArrowRight,
  BadgeCheck,
  BookOpenCheck,
  Brain,
  CalendarClock,
  FileSearch,
  Filter,
  Mail,
  MapPin,
  Phone,
  ScanLine,
  ShieldAlert,
  Star,
  Stethoscope,
  Target,
  Users,
} from "lucide-react";
import { Avatar, Badge, Button, GlassCard, Input, SectionHeading, Textarea } from "@/components/ui";
import { PageTransition, Reveal } from "@/components/motion";
import { OrbScene } from "@/components/three/HeroScene";
import { doctors, specializationLabel, type Specialization } from "@/mocks/data";
import { fmtMoney } from "@/lib/format";
import { cn } from "@/lib/cn";

/* ------------------------------------------------------------ shared hero */
function Hero({
  eyebrow,
  title,
  lead,
  children,
}: {
  eyebrow: string;
  title: React.ReactNode;
  lead: string;
  children?: React.ReactNode;
}) {
  return (
    <section className="relative overflow-hidden pt-36 pb-16">
      <div className="pointer-events-none absolute inset-0 grid-dots opacity-60" />
      <OrbScene className="pointer-events-none absolute -right-20 top-10 hidden h-[420px] w-[520px] lg:block" />
      <div className="relative mx-auto max-w-7xl px-4 sm:px-6">
        <Reveal>
          <SectionHeading eyebrow={eyebrow} title={title} lead={lead} />
        </Reveal>
        {children}
      </div>
    </section>
  );
}

/* --------------------------------------------------------------- Services */
export function Services() {
  const cards = [
    {
      icon: ScanLine,
      title: "Report extraction",
      body: "PDF or photo → structured values with units, the lab's reference ranges and a confidence score per field. You confirm before analysis.",
      tag: "OCR + review",
    },
    {
      icon: ShieldAlert,
      title: "Critical-value alerts",
      body: "Deterministic rules sourced from clinical guidelines flag life-threatening values first. They are never edited away by automation.",
      tag: "Safety",
    },
    {
      icon: Brain,
      title: "Risk assessment",
      body: "Anemia, CKD, liver, thyroid and glycaemic risk from versioned model bundles with published metrics. Missing tests → 'not assessable'.",
      tag: "6 conditions",
    },
    {
      icon: BookOpenCheck,
      title: "Plain-language explanation",
      body: "A patient summary and a clinician summary, grounded in WHO / KDIGO / ESC guidance, every number validated against your report.",
      tag: "Grounded LLM",
    },
    {
      icon: Stethoscope,
      title: "Specialist matching",
      body: "Your results route you to the right speciality — nephrologist, endocrinologist, hepatologist, hematologist, cardiologist.",
      tag: "Consultation",
    },
    {
      icon: CalendarClock,
      title: "Booking & payment",
      body: "Verified doctors publish slots; pay with Khalti; doctors comment directly on your report.",
      tag: "Khalti",
    },
  ];
  return (
    <PageTransition>
      <Hero
        eyebrow="Services"
        title={
          <>
            Everything between a <span className="text-gradient">report</span> and a decision.
          </>
        }
        lead="Each capability is a separate, testable step. That is what makes the whole trustworthy."
      />
      <section className="mx-auto max-w-7xl px-4 pb-28 sm:px-6">
        <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {cards.map((c, i) => (
            <Reveal key={c.title} delay={i * 0.06}>
              <GlassCard hover className="group h-full">
                <div className="flex items-start justify-between">
                  <span className="grid size-12 place-items-center rounded-2xl bg-brand-100 text-brand-700 transition group-hover:bg-gradient-to-br group-hover:from-brand-500 group-hover:to-brand-700 group-hover:text-white">
                    <c.icon className="size-5" />
                  </span>
                  <Badge tone="neutral">{c.tag}</Badge>
                </div>
                <h3 className="mt-6 text-xl font-bold">{c.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-soft">{c.body}</p>
              </GlassCard>
            </Reveal>
          ))}
        </div>
        <Reveal className="mt-16">
          <div className="glass-strong flex flex-col items-center justify-between gap-6 px-8 py-8 md:flex-row">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.2em] text-brand-700">Try it</p>
              <h3 className="mt-2 text-2xl font-bold">
                Walk through the full flow with a sample report.
              </h3>
            </div>
            <Button to="/reports/upload" size="lg" icon={ArrowRight} trailing>
              Start the flow
            </Button>
          </div>
        </Reveal>
      </section>
    </PageTransition>
  );
}

/* ------------------------------------------------------------------ About */
export function About() {
  const team: [string, string, string][] = [
    ["Aashish Sharma", "Frontend & product", "Pixels, motion and the words people actually read."],
    ["Prabhat Acharya", "Machine learning", "Models that admit what they do not know."],
    [
      "Pratik Chapagain",
      "Backend & AI systems",
      "Pipelines, contracts and everything that has to be boring and reliable.",
    ],
    ["Supreme Badal", "Frontend & quality", "If it can break, he finds it first."],
  ];
  return (
    <PageTransition>
      <Hero
        eyebrow="About"
        title={
          <>
            We started with a report <span className="text-gradient">nobody could read.</span>
          </>
        }
        lead="MediScan is built by four BE IT students who got tired of watching people photograph a lab report and then Google every line of it. We are making the kind of tool we wish our own families had."
      />
      <section className="mx-auto max-w-7xl px-4 pb-28 sm:px-6">
        <div className="grid gap-5 lg:grid-cols-3">
          {[
            {
              icon: Target,
              title: "Mission",
              body: "Close the gap between receiving a medical result and understanding it — for people with low health literacy first.",
            },
            {
              icon: FileSearch,
              title: "Principles",
              body: "Fail loudly, never guess. Say 'not assessable' rather than impute. Keep the lab's own ranges. Cite the guideline. Always end with a doctor.",
            },
            {
              icon: Users,
              title: "Who we serve",
              body: "Patients in Nepal, the clinicians who review their results, and the labs whose formats we learn to read.",
            },
          ].map((c, i) => (
            <Reveal key={c.title} delay={i * 0.08}>
              <GlassCard hover className="h-full">
                <c.icon className="size-7 text-brand-600" />
                <h3 className="mt-5 text-xl font-bold">{c.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-soft">{c.body}</p>
              </GlassCard>
            </Reveal>
          ))}
        </div>
        <Reveal className="mt-20">
          <SectionHeading
            eyebrow="The team"
            title={
              <>
                Four BE IT students. <span className="text-gradient">One stubborn idea.</span>
              </>
            }
            lead="We are Bachelor of Engineering in Information Technology students who believe medical data should make sense to the person it belongs to. MediScan is what happens when you take that seriously for long enough."
          />
        </Reveal>
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {team.map(([name, role, line], i) => (
            <Reveal key={name} delay={i * 0.06}>
              <GlassCard hover className="flex h-full flex-col">
                <div className="flex items-center gap-4">
                  <Avatar name={name} size={52} />
                  <div>
                    <p className="font-bold">{name}</p>
                    <p className="text-xs font-semibold text-brand-700">{role}</p>
                  </div>
                </div>
                <p className="mt-4 text-sm leading-relaxed text-ink-soft">{line}</p>
                <span className="mt-4 inline-flex w-fit rounded-full bg-brand-50 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-brand-800">
                  BE IT
                </span>
              </GlassCard>
            </Reveal>
          ))}
        </div>
      </section>
    </PageTransition>
  );
}

/* ---------------------------------------------------------------- Contact */
export function Contact() {
  return (
    <PageTransition>
      <Hero
        eyebrow="Contact"
        title={
          <>
            Talk to us — <span className="text-gradient">humans answer.</span>
          </>
        }
        lead="Labs who want their formats supported, clinicians who want to join, and anyone with a question."
      />
      <section className="mx-auto max-w-7xl px-4 pb-28 sm:px-6">
        <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
          <div className="space-y-4">
            {[
              [Mail, "hello@mediscan.health", "General enquiries"],
              [Phone, "+977 01 4XXXXXX", "Mon–Fri, 9am–6pm NPT"],
              [MapPin, "Kathmandu, Nepal", "Remote-first team"],
            ].map(([Icon, v, s], i) => {
              const I = Icon as typeof Mail;
              return (
                <Reveal key={v as string} delay={i * 0.06}>
                  <GlassCard className="flex items-center gap-4">
                    <span className="grid size-11 place-items-center rounded-2xl bg-brand-100 text-brand-700">
                      <I className="size-5" />
                    </span>
                    <div>
                      <p className="font-bold">{v as string}</p>
                      <p className="text-xs text-ink-muted">{s as string}</p>
                    </div>
                  </GlassCard>
                </Reveal>
              );
            })}
          </div>
          <Reveal delay={0.1}>
            <GlassCard strong className="p-8">
              <form className="grid gap-4 sm:grid-cols-2" onSubmit={(e) => e.preventDefault()}>
                <Input label="Name" name="name" placeholder="Your name" />
                <Input label="Email" name="email" type="email" placeholder="you@example.com" />
                <div className="sm:col-span-2">
                  <Input label="Subject" name="subject" placeholder="How can we help?" />
                </div>
                <div className="sm:col-span-2">
                  <Textarea label="Message" name="message" placeholder="Tell us a little more…" />
                </div>
                <div className="sm:col-span-2">
                  <Button type="submit" size="lg" icon={ArrowRight} trailing>
                    Send message
                  </Button>
                </div>
              </form>
            </GlassCard>
          </Reveal>
        </div>
      </section>
    </PageTransition>
  );
}

/* ---------------------------------------------------------------- Privacy */
export function Privacy() {
  const sections = [
    [
      "What we store",
      "Your account, the report files you upload, the values extracted from them, model assessments, explanations, conversations and consultation records.",
    ],
    [
      "Where it lives",
      "Files are in encrypted object storage and reachable only through short-lived signed links. Structured data is in a database in the region you use the service.",
    ],
    [
      "Who can see it",
      "You; a doctor you have booked or linked; administrators for verification tasks. Every access is written to an audit log you can request.",
    ],
    [
      "What leaves the platform",
      "When a language model is used, only de-identified values are sent — never your name, ID, date of birth or contact details.",
    ],
    [
      "Your rights",
      "Download or delete everything at any time from your profile. Deletion removes files, values, assessments and conversations.",
    ],
  ];
  return (
    <PageTransition>
      <Hero
        eyebrow="Privacy"
        title={
          <>
            Your data, <span className="text-gradient">on your terms.</span>
          </>
        }
        lead="Draft policy for the building phase. Final wording will be reviewed before public launch."
      />
      <section className="mx-auto max-w-4xl px-4 pb-28 sm:px-6">
        <div className="space-y-4">
          {sections.map(([t, b], i) => (
            <Reveal key={t} delay={i * 0.05}>
              <GlassCard>
                <h3 className="text-lg font-bold">{t}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-soft">{b}</p>
              </GlassCard>
            </Reveal>
          ))}
        </div>
      </section>
    </PageTransition>
  );
}

/* ------------------------------------------------------------ Doctor list */
const specs: (Specialization | "ALL")[] = [
  "ALL",
  "NEPHROLOGIST",
  "ENDOCRINOLOGIST",
  "HEPATOLOGIST",
  "CARDIOLOGIST",
  "HEMATOLOGIST",
  "GENERAL_PHYSICIAN",
];

export function DoctorCard({ d, compact }: { d: (typeof doctors)[number]; compact?: boolean }) {
  return (
    <GlassCard hover className={cn("flex h-full flex-col", compact && "p-5")}>
      <div className="flex items-start gap-4">
        <Avatar name={d.name} size={56} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="truncate text-lg font-bold">{d.name}</h3>
            {d.status === "VERIFIED" && <BadgeCheck className="size-4 text-brand-600" />}
          </div>
          <p className="text-sm font-semibold text-brand-700">
            {specializationLabel[d.specialization]}
          </p>
          <p className="truncate text-xs text-ink-muted">{d.hospital}</p>
        </div>
      </div>
      {!compact && <p className="mt-4 text-sm leading-relaxed text-ink-soft">{d.bio}</p>}
      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-soft">
        <span className="inline-flex items-center gap-1 font-semibold text-ink">
          <Star className="size-3.5 fill-amber-400 text-amber-400" /> {d.rating}{" "}
          <span className="font-normal text-ink-muted">({d.reviews})</span>
        </span>
        <span>{d.experience} yrs experience</span>
        <span className="font-semibold text-ink">{fmtMoney(d.fee)}</span>
      </div>
      <div className="mt-5 flex items-center justify-between gap-3 border-t border-line pt-4">
        <span className="text-xs text-ink-muted">
          Next: <span className="font-semibold text-ink">{d.nextSlot}</span>
        </span>
        <Button
          to={`/doctors/${d.id}`}
          size="sm"
          variant={d.status === "VERIFIED" ? "primary" : "glass"}
          disabled={d.status !== "VERIFIED"}
        >
          {d.status === "VERIFIED" ? "Book" : "Pending"}
        </Button>
      </div>
    </GlassCard>
  );
}

export function Doctors() {
  const [spec, setSpec] = useState<Specialization | "ALL">("ALL");
  const list = doctors.filter((d) => spec === "ALL" || d.specialization === spec);
  return (
    <PageTransition>
      <Hero
        eyebrow="Find a doctor"
        title={
          <>
            Verified specialists, <span className="text-gradient">matched to your results.</span>
          </>
        }
        lead="Every doctor here has had their licence reviewed by our team. Fees are shown up front."
      >
        <Reveal delay={0.1} className="mt-10">
          <div className="flex flex-wrap items-center gap-2">
            <Filter className="mr-1 size-4 text-ink-muted" />
            {specs.map((s) => (
              <button
                key={s}
                onClick={() => setSpec(s)}
                className={cn(
                  "rounded-full px-4 py-2 text-sm font-semibold transition",
                  spec === s
                    ? "bg-gradient-to-br from-brand-500 to-brand-700 text-white"
                    : "glass-pill text-ink-soft hover:text-ink",
                )}
              >
                {s === "ALL" ? "All" : specializationLabel[s]}
              </button>
            ))}
          </div>
        </Reveal>
      </Hero>
      <section className="mx-auto max-w-7xl px-4 pb-28 sm:px-6">
        <motion.div layout className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {list.map((d, i) => (
            <motion.div
              key={d.id}
              layout
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
            >
              <DoctorCard d={d} />
            </motion.div>
          ))}
        </motion.div>
      </section>
    </PageTransition>
  );
}

/* --------------------------------------------------------- Doctor profile */
export function DoctorProfilePublic() {
  const { id } = useParams();
  const d = doctors.find((x) => x.id === id) ?? doctors[0];
  const slots = [
    "09:00",
    "09:20",
    "09:40",
    "10:00",
    "10:20",
    "11:00",
    "16:00",
    "16:20",
    "16:40",
    "17:00",
  ];
  const booked = new Set(["09:40", "16:20"]);
  const [slot, setSlot] = useState<string | null>(null);
  const days = Array.from({ length: 6 }, (_, i) => {
    const dt = new Date();
    dt.setDate(dt.getDate() + i);
    return dt;
  });
  const [day, setDay] = useState(0);

  return (
    <PageTransition>
      <section className="relative pt-36 pb-28">
        <div className="pointer-events-none absolute inset-0 grid-dots opacity-60" />
        <div className="relative mx-auto grid max-w-7xl gap-6 px-4 sm:px-6 lg:grid-cols-[1.1fr_1fr]">
          <Reveal>
            <GlassCard strong className="p-8">
              <Link to="/doctors" className="text-xs font-semibold text-brand-700 hover:underline">
                ← All doctors
              </Link>
              <div className="mt-6 flex items-start gap-5">
                <Avatar name={d.name} size={84} />
                <div>
                  <div className="flex items-center gap-2">
                    <h1 className="text-3xl font-bold">{d.name}</h1>
                    <BadgeCheck className="size-5 text-brand-600" />
                  </div>
                  <p className="text-brand-700 font-semibold">
                    {specializationLabel[d.specialization]}
                  </p>
                  <p className="text-sm text-ink-muted">{d.hospital}</p>
                </div>
              </div>
              <p className="mt-6 leading-relaxed text-ink-soft">{d.bio}</p>
              <div className="mt-6 grid grid-cols-3 gap-3">
                {[
                  ["Rating", `${d.rating} ★`],
                  ["Experience", `${d.experience} yrs`],
                  ["Fee", fmtMoney(d.fee)],
                ].map(([k, v]) => (
                  <div key={k} className="rounded-2xl bg-white/70 p-4">
                    <p className="text-[11px] font-bold uppercase tracking-wider text-ink-muted">
                      {k}
                    </p>
                    <p className="mt-1 font-display text-xl font-bold">{v}</p>
                  </div>
                ))}
              </div>
              <div className="mt-6 rounded-2xl border border-brand-200 bg-brand-50 p-4 text-sm text-brand-900">
                <p className="font-bold">Licence verified by MediScan</p>
                <p className="mt-1 text-brand-800/80">
                  Reviewed against the Nepal Medical Council register. Verification ID on request.
                </p>
              </div>
            </GlassCard>
          </Reveal>
          <Reveal delay={0.1}>
            <GlassCard strong className="p-8">
              <h2 className="text-xl font-bold">Book a consultation</h2>
              <p className="mt-1 text-sm text-ink-soft">
                20-minute video or in-person visit. Pay securely with Khalti.
              </p>
              <div className="mt-6 flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
                {days.map((dt, i) => (
                  <button
                    key={i}
                    onClick={() => setDay(i)}
                    className={cn(
                      "min-w-[72px] rounded-2xl px-3 py-3 text-center transition",
                      day === i
                        ? "bg-gradient-to-br from-brand-500 to-brand-700 text-white"
                        : "bg-white/70 text-ink hover:bg-white",
                    )}
                  >
                    <p className="text-[11px] font-bold uppercase opacity-80">
                      {dt.toLocaleDateString("en", { weekday: "short" })}
                    </p>
                    <p className="font-display text-xl font-bold">{dt.getDate()}</p>
                  </button>
                ))}
              </div>
              <div className="mt-6 grid grid-cols-3 gap-2 sm:grid-cols-5">
                {slots.map((s) => {
                  const isBooked = booked.has(s);
                  return (
                    <button
                      key={s}
                      disabled={isBooked}
                      onClick={() => setSlot(s)}
                      className={cn(
                        "rounded-xl py-2.5 text-sm font-semibold transition",
                        isBooked
                          ? "cursor-not-allowed bg-ink/5 text-ink-muted line-through"
                          : slot === s
                            ? "bg-brand-600 text-white shadow-glow"
                            : "bg-white/70 text-ink hover:bg-white",
                      )}
                    >
                      {s}
                    </button>
                  );
                })}
              </div>
              <div className="mt-6 flex items-center justify-between rounded-2xl bg-white/70 p-4">
                <div>
                  <p className="text-[11px] font-bold uppercase tracking-wider text-ink-muted">
                    Total
                  </p>
                  <p className="font-display text-2xl font-bold">{fmtMoney(d.fee)}</p>
                </div>
                <Button
                  to={`/book-appointment/${d.id}?slot=${slot ?? ""}&day=${day}`}
                  size="lg"
                  disabled={!slot}
                  icon={ArrowRight}
                  trailing
                >
                  Continue
                </Button>
              </div>
            </GlassCard>
          </Reveal>
        </div>
      </section>
    </PageTransition>
  );
}
