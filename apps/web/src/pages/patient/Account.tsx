import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ArrowRight,
  CalendarClock,
  CheckCircle2,
  CreditCard,
  Download,
  FileText,
  Mail,
  MapPin,
  Phone,
  ShieldCheck,
  Trash2,
  UserRound,
  Wallet,
} from "lucide-react";
import {
  Avatar,
  Badge,
  Button,
  EmptyState,
  GlassCard,
  Input,
  PageHead,
  Tabs,
} from "@/components/ui";
import { Reveal } from "@/components/motion";
import { appointments, doctors, me, specializationLabel } from "@/mocks/data";
import { fmtDate, fmtMoney, fmtTime } from "@/lib/format";
import { cn } from "@/lib/cn";

/* ------------------------------------------------------- appointments */
export function PatientAppointments() {
  const [tab, setTab] = useState<"upcoming" | "past">("upcoming");
  const mine = appointments.filter(
    (a) => a.patient === me.name || ["a1", "a3", "a4"].includes(a.id),
  );
  const list = mine.filter((a) =>
    tab === "upcoming"
      ? ["PAID", "PENDING"].includes(a.status)
      : ["COMPLETED", "CANCELLED"].includes(a.status),
  );
  return (
    <>
      <PageHead
        title="Appointments"
        lead="Your consultations, payments and cancellations."
        actions={
          <Button to="/doctors" icon={CalendarClock}>
            Book new
          </Button>
        }
      />
      <Tabs
        tabs={[
          {
            id: "upcoming",
            label: "Upcoming",
            count: mine.filter((a) => ["PAID", "PENDING"].includes(a.status)).length,
          },
          { id: "past", label: "Past" },
        ]}
        value={tab}
        onChange={setTab}
      />
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        {list.length === 0 && (
          <EmptyState
            icon={CalendarClock}
            title="Nothing here"
            body="No appointments in this view."
          />
        )}
        {list.map((a, i) => {
          const d = doctors.find((x) => x.id === a.doctorId)!;
          return (
            <motion.div
              key={a.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
            >
              <GlassCard hover className="flex flex-col gap-4">
                <div className="flex items-start gap-4">
                  <Avatar name={d.name} size={52} />
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-bold">{d.name}</p>
                    <p className="text-xs font-semibold text-brand-700">
                      {specializationLabel[d.specialization]}
                    </p>
                    <p className="text-xs text-ink-muted">{d.hospital}</p>
                  </div>
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
                </div>
                <div className="grid grid-cols-3 gap-3 rounded-2xl bg-white/70 p-3 text-sm">
                  <div>
                    <p className="text-[10px] font-bold uppercase text-ink-muted">Date</p>
                    <p className="font-semibold">{fmtDate(a.date)}</p>
                  </div>
                  <div>
                    <p className="text-[10px] font-bold uppercase text-ink-muted">Time</p>
                    <p className="font-semibold">{fmtTime(a.start)}</p>
                  </div>
                  <div>
                    <p className="text-[10px] font-bold uppercase text-ink-muted">Fee</p>
                    <p className="font-semibold">{fmtMoney(a.amount)}</p>
                  </div>
                </div>
                <div className="flex gap-2">
                  {a.status === "PENDING" && (
                    <Button to={`/book-appointment/${a.doctorId}`} size="sm" icon={CreditCard}>
                      Pay now
                    </Button>
                  )}
                  {a.reportId && (
                    <Button
                      to={`/reports/${a.reportId}/result`}
                      size="sm"
                      variant="glass"
                      icon={FileText}
                    >
                      Linked report
                    </Button>
                  )}
                  {["PAID", "PENDING"].includes(a.status) && (
                    <Button size="sm" variant="ghost">
                      Cancel
                    </Button>
                  )}
                </div>
              </GlassCard>
            </motion.div>
          );
        })}
      </div>
      <p className="mt-6 text-xs text-ink-muted">
        Cancellation more than 24 hours before the slot refunds 60%. Within 24 hours no refund
        applies.
      </p>
    </>
  );
}

/* ------------------------------------------------------------ profile */
export function PatientProfile() {
  return (
    <>
      <PageHead title="Profile" lead="Your details, security and data controls." />
      <div className="grid gap-6 lg:grid-cols-[1fr_1.6fr]">
        <Reveal>
          <GlassCard strong className="text-center">
            <Avatar name={me.name} size={96} className="mx-auto text-3xl" />
            <h2 className="mt-4 text-2xl font-bold">{me.name}</h2>
            <p className="text-sm text-ink-muted">{me.email}</p>
            <Badge tone="brand" className="mt-3">
              Patient
            </Badge>
            <div className="mt-6 space-y-2 text-left text-sm">
              {[
                [Mail, me.email],
                [Phone, me.phone],
                [MapPin, me.city],
              ].map(([I, v]) => {
                const Icon = I as typeof Mail;
                return (
                  <p
                    key={v as string}
                    className="flex items-center gap-3 rounded-xl bg-white/70 px-3 py-2"
                  >
                    <Icon className="size-4 text-brand-600" /> {v as string}
                  </p>
                );
              })}
            </div>
          </GlassCard>
        </Reveal>
        <div className="space-y-6">
          <Reveal delay={0.05}>
            <GlassCard>
              <h3 className="flex items-center gap-2 text-lg font-bold">
                <UserRound className="size-5 text-brand-600" /> Personal details
              </h3>
              <form className="mt-5 grid gap-4 sm:grid-cols-2" onSubmit={(e) => e.preventDefault()}>
                <Input label="First name" name="first" defaultValue="Prabhat" />
                <Input label="Last name" name="last" defaultValue="Acharya" />
                <Input label="Date of birth" name="dob" type="date" defaultValue={me.dob} />
                <Input label="Phone" name="phone" defaultValue={me.phone} />
                <Input label="City" name="city" defaultValue={me.city} />
                <Input
                  label="Sex"
                  name="sex"
                  defaultValue="Male"
                  hint="Used for sex-specific reference ranges (e.g. hemoglobin)."
                />
                <div className="sm:col-span-2">
                  <Button type="submit">Save changes</Button>
                </div>
              </form>
            </GlassCard>
          </Reveal>
          <Reveal delay={0.1}>
            <GlassCard>
              <h3 className="flex items-center gap-2 text-lg font-bold">
                <ShieldCheck className="size-5 text-brand-600" /> Your data
              </h3>
              <p className="mt-2 text-sm text-ink-soft">
                Every access to your reports is logged. You can export or permanently delete
                everything.
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Button variant="glass" icon={Download}>
                  Export all data
                </Button>
                <Button variant="glass" icon={FileText}>
                  Access log
                </Button>
                <Button variant="danger" icon={Trash2}>
                  Delete account
                </Button>
              </div>
            </GlassCard>
          </Reveal>
        </div>
      </div>
    </>
  );
}

/* ---------------------------------------------------------- booking */
export function BookAppointment() {
  const { id } = useParams();
  const [params] = useSearchParams();
  const d = doctors.find((x) => x.id === id) ?? doctors[0];
  const slot = params.get("slot") || "16:00";
  const [step, setStep] = useState<1 | 2>(1);
  const commission = Math.round(d.fee * 0.25);
  return (
    <div className="mx-auto max-w-4xl">
      <PageHead
        crumbs={["Doctors", d.name, "Book"]}
        title="Confirm and pay"
        lead="Your slot is held for 10 minutes while you complete payment."
      />
      <div className="grid gap-6 md:grid-cols-[1fr_1fr]">
        <Reveal>
          <GlassCard strong>
            <div className="flex items-center gap-4">
              <Avatar name={d.name} size={60} />
              <div>
                <p className="text-lg font-bold">{d.name}</p>
                <p className="text-sm font-semibold text-brand-700">
                  {specializationLabel[d.specialization]}
                </p>
                <p className="text-xs text-ink-muted">{d.hospital}</p>
              </div>
            </div>
            <div className="mt-6 space-y-2 text-sm">
              {[
                ["Date", fmtDate(new Date().toISOString())],
                ["Time", `${fmtTime(slot)} · 20 min`],
                ["Mode", "Video consultation"],
                ["Linked report", "r1042 — Bir Hospital"],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between rounded-xl bg-white/70 px-3 py-2">
                  <span className="text-ink-muted">{k}</span>
                  <span className="font-semibold">{v}</span>
                </div>
              ))}
            </div>
            <div className="mt-6 border-t border-line pt-4 text-sm">
              <div className="flex justify-between">
                <span className="text-ink-muted">Consultation fee</span>
                <span>{fmtMoney(d.fee)}</span>
              </div>
              <div className="flex justify-between text-xs text-ink-muted">
                <span>incl. platform share (25%)</span>
                <span>{fmtMoney(commission)}</span>
              </div>
              <div className="mt-2 flex justify-between font-display text-xl font-bold">
                <span>Total</span>
                <span>{fmtMoney(d.fee)}</span>
              </div>
            </div>
          </GlassCard>
        </Reveal>
        <Reveal delay={0.1}>
          <GlassCard strong className="flex h-full flex-col">
            <h3 className="flex items-center gap-2 text-lg font-bold">
              <Wallet className="size-5 text-brand-600" /> Payment
            </h3>
            <div className="mt-5 space-y-3">
              {[
                { id: 1, name: "Khalti", sub: "Wallet, mobile banking, cards", active: true },
                { id: 2, name: "eSewa", sub: "Coming soon", active: false },
                { id: 3, name: "Pay at clinic", sub: "In-person visits only", active: false },
              ].map((m) => (
                <div
                  key={m.id}
                  className={cn(
                    "flex items-center gap-3 rounded-2xl border p-4",
                    m.active
                      ? "border-brand-400 bg-brand-50 shadow-glow"
                      : "border-white/80 bg-white/50 opacity-60",
                  )}
                >
                  <span
                    className={cn(
                      "size-4 rounded-full border-4",
                      m.active ? "border-brand-600 bg-white" : "border-ink/20",
                    )}
                  />
                  <div>
                    <p className="font-bold">{m.name}</p>
                    <p className="text-xs text-ink-muted">{m.sub}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-auto pt-6">
              {step === 1 ? (
                <Button
                  size="lg"
                  className="w-full"
                  icon={ArrowRight}
                  trailing
                  onClick={() => setStep(2)}
                >
                  Pay {fmtMoney(d.fee)} with Khalti
                </Button>
              ) : (
                <div className="rounded-2xl bg-[#5C2D91] p-5 text-center text-white">
                  <p className="font-display text-xl font-bold">Khalti checkout (sandbox)</p>
                  <p className="mt-1 text-sm text-white/80">
                    In production this redirects to Khalti's ePayment page.
                  </p>
                  <Button
                    to="/payment/success"
                    size="lg"
                    variant="glass"
                    className="mt-4 w-full text-[#5C2D91]"
                  >
                    Simulate successful payment
                  </Button>
                </div>
              )}
              <p className="mt-3 text-center text-xs text-ink-muted">
                Payment is verified server-side before the slot is confirmed.
              </p>
            </div>
          </GlassCard>
        </Reveal>
      </div>
    </div>
  );
}

/* --------------------------------------------------- payment success */
export function PaymentSuccess() {
  return (
    <div className="mx-auto max-w-lg py-10 text-center">
      <motion.div
        initial={{ scale: 0.6, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: "spring", stiffness: 260, damping: 18 }}
        className="relative mx-auto grid size-24 place-items-center rounded-full bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-glow"
      >
        <span className="absolute inset-0 rounded-full bg-brand-400/50 animate-pulse-ring" />
        <CheckCircle2 className="relative size-12" />
      </motion.div>
      <h1 className="mt-8 text-3xl font-bold">You're booked</h1>
      <p className="mt-2 text-ink-soft">
        Payment verified. Dr. Anjali Thapa will see you tomorrow at 6:00 PM. A confirmation is on
        its way to your inbox.
      </p>
      <GlassCard strong className="mt-8 text-left">
        {[
          ["Transaction", "pidx_8f3a…c21e"],
          ["Amount", fmtMoney(1500)],
          ["Status", "Completed"],
        ].map(([k, v]) => (
          <div
            key={k}
            className="flex justify-between border-b border-line py-2 text-sm last:border-0"
          >
            <span className="text-ink-muted">{k}</span>
            <span className="font-semibold">{v}</span>
          </div>
        ))}
      </GlassCard>
      <div className="mt-8 flex justify-center gap-2">
        <Button to="/appointments" icon={CalendarClock}>
          My appointments
        </Button>
        <Button to="/dashboard" variant="glass">
          Back to dashboard
        </Button>
      </div>
      <Link
        to="/reports/r1042/result"
        className="mt-6 inline-block text-sm font-semibold text-brand-700 hover:underline"
      >
        Share your latest report with the doctor →
      </Link>
    </div>
  );
}
