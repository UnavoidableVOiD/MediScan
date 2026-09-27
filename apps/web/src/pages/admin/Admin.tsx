import { useState } from "react";
import { motion } from "framer-motion";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis } from "recharts";
import {
  BadgeCheck,
  Ban,
  Check,
  Eye,
  FileText,
  Mail,
  Search,
  ShieldCheck,
  Stethoscope,
  Trash2,
  UserCog,
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
} from "@/components/ui";
import { Reveal } from "@/components/motion";
import { adminStats, doctors, patients, specializationLabel, type Doctor } from "@/mocks/data";
import { fmtDate, fmtMoney } from "@/lib/format";

const perDay = adminStats.perDay.map((n, i) => ({ d: `${i + 5} Sep`, n }));

/* ---------------------------------------------------------- dashboard */
export function AdminDashboard() {
  return (
    <>
      <PageHead
        title="Platform overview"
        lead="Health, growth and money — every number here is auditable."
        actions={
          <Button to="/admin/doctors" icon={BadgeCheck}>
            Review licences ({adminStats.pendingLicences})
          </Button>
        }
      />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat
          label="Patients"
          value={adminStats.patients.toLocaleString()}
          icon={Users}
          delta="+84 this month"
        />
        <Stat
          label="Verified doctors"
          value={adminStats.doctors}
          icon={Stethoscope}
          delta={`${adminStats.pendingLicences} pending review`}
        />
        <Stat
          label="Reports this month"
          value={adminStats.reportsThisMonth.toLocaleString()}
          icon={FileText}
          delta="p95 pipeline 48 s"
        />
        <Stat
          label="Platform revenue"
          value={fmtMoney(adminStats.commission)}
          icon={Wallet}
          delta={`of ${fmtMoney(adminStats.revenue)} gross`}
        />
      </div>
      <div className="mt-6 grid gap-6 lg:grid-cols-[1.5fr_1fr]">
        <Reveal>
          <GlassCard className="h-full">
            <h3 className="text-xl font-bold">Reports processed per day</h3>
            <div className="mt-4 h-60">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={perDay} margin={{ left: 0, right: 0 }}>
                  <defs>
                    <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10b981" stopOpacity={0.5} />
                      <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="d"
                    tick={{ fontSize: 11, fill: "#7b918a" }}
                    axisLine={false}
                    tickLine={false}
                    interval={2}
                  />
                  <Tooltip contentStyle={{ borderRadius: 12, fontSize: 12 }} />
                  <Area
                    type="monotone"
                    dataKey="n"
                    stroke="#047857"
                    strokeWidth={2.5}
                    fill="url(#g2)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </GlassCard>
        </Reveal>
        <Reveal delay={0.1}>
          <GlassCard className="h-full">
            <h3 className="text-xl font-bold">Money</h3>
            <div className="mt-4 space-y-3 text-sm">
              {[
                ["Gross paid", adminStats.revenue],
                ["Refunded", adminStats.refunds],
                [
                  "Doctor payouts (75%)",
                  adminStats.revenue - adminStats.refunds - adminStats.commission,
                ],
                ["Platform share (25%)", adminStats.commission],
              ].map(([k, v]) => (
                <div
                  key={k as string}
                  className="flex justify-between rounded-xl bg-white/70 px-3 py-2.5"
                >
                  <span className="text-ink-soft">{k}</span>
                  <span className="font-semibold">{fmtMoney(v as number)}</span>
                </div>
              ))}
            </div>
            <div className="mt-4 flex items-center gap-2 rounded-2xl bg-brand-50 p-3 text-xs text-brand-900">
              <ShieldCheck className="size-4" /> Every PHI access this month: 12,408 events, 0
              policy denials.
            </div>
          </GlassCard>
        </Reveal>
      </div>
      <Reveal className="mt-6">
        <h3 className="mb-3 text-xl font-bold">Licences awaiting review</h3>
        <Table head={["Doctor", "Specialization", "Hospital", "Submitted", ""]}>
          {doctors
            .filter((d) => d.status !== "VERIFIED")
            .concat(doctors.slice(0, 2).map((d) => ({ ...d, status: "PENDING" as const })))
            .map((d, i) => (
              <tr key={d.id + i} className="hover:bg-white/50">
                <Td>
                  <span className="flex items-center gap-3">
                    <Avatar name={d.name} size={32} />
                    <span className="font-semibold">{d.name}</span>
                  </span>
                </Td>
                <Td>{specializationLabel[d.specialization]}</Td>
                <Td className="text-ink-soft">{d.hospital}</Td>
                <Td>{fmtDate("2026-09-16")}</Td>
                <Td className="text-right">
                  <Button to="/admin/doctors" size="sm" variant="glass" icon={Eye}>
                    Review
                  </Button>
                </Td>
              </tr>
            ))}
        </Table>
      </Reveal>
    </>
  );
}

/* ------------------------------------------------------------ doctors */
export function AdminDoctors() {
  const [tab, setTab] = useState<"pending" | "verified" | "rejected">("pending");
  const [selected, setSelected] = useState<Doctor | null>(null);
  const [rows, setRows] = useState(
    doctors.map((d, i) => (i < 2 ? { ...d, status: "PENDING" as const } : d)),
  );
  const list = rows.filter((d) =>
    tab === "pending"
      ? ["PENDING", "UNVERIFIED"].includes(d.status)
      : d.status === tab.toUpperCase(),
  );
  const decide = (id: string, status: Doctor["status"]) => {
    setRows((r) => r.map((d) => (d.id === id ? { ...d, status } : d)));
    setSelected(null);
  };
  return (
    <>
      <PageHead
        title="Doctors"
        lead="Review licences, verify, or revoke. Decisions are logged with your account."
      />
      <Tabs
        tabs={[
          {
            id: "pending",
            label: "Pending",
            count: rows.filter((d) => ["PENDING", "UNVERIFIED"].includes(d.status)).length,
          },
          {
            id: "verified",
            label: "Verified",
            count: rows.filter((d) => d.status === "VERIFIED").length,
          },
          { id: "rejected", label: "Rejected" },
        ]}
        value={tab}
        onChange={setTab}
      />
      <div className="mt-6 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <Table head={["Doctor", "Specialization", "Fee", "Status", ""]}>
          {list.map((d) => (
            <tr key={d.id} className="hover:bg-white/50">
              <Td>
                <span className="flex items-center gap-3">
                  <Avatar name={d.name} size={32} />
                  <span>
                    <span className="block font-semibold">{d.name}</span>
                    <span className="block text-xs text-ink-muted">{d.hospital}</span>
                  </span>
                </span>
              </Td>
              <Td>{specializationLabel[d.specialization]}</Td>
              <Td>{fmtMoney(d.fee)}</Td>
              <Td>
                <Badge
                  tone={
                    d.status === "VERIFIED"
                      ? "brand"
                      : d.status === "REJECTED"
                        ? "critical"
                        : "warn"
                  }
                >
                  {d.status}
                </Badge>
              </Td>
              <Td className="text-right">
                <Button size="sm" variant="glass" onClick={() => setSelected(d)} icon={Eye}>
                  Open
                </Button>
              </Td>
            </tr>
          ))}
        </Table>
        <div>
          {selected ? (
            <motion.div
              key={selected.id}
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <GlassCard strong className="sticky top-24">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <Avatar name={selected.name} size={48} />
                    <div>
                      <p className="font-bold">{selected.name}</p>
                      <p className="text-xs text-brand-700 font-semibold">
                        {specializationLabel[selected.specialization]}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setSelected(null)}
                    className="grid size-8 place-items-center rounded-full hover:bg-brand-50"
                    aria-label="Close"
                  >
                    <X className="size-4" />
                  </button>
                </div>
                <div className="mt-5 space-y-2 text-sm">
                  {[
                    ["Licence no.", "NMC-1" + selected.id.slice(1) + "874"],
                    ["Hospital", selected.hospital],
                    ["Experience", `${selected.experience} years`],
                    ["Submitted", fmtDate("2026-09-16")],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between rounded-xl bg-white/70 px-3 py-2">
                      <span className="text-ink-muted">{k}</span>
                      <span className="font-semibold">{v}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-4 rounded-2xl border border-line bg-white p-3">
                  <p className="text-xs font-bold uppercase tracking-wider text-ink-muted">
                    Documents
                  </p>
                  <div className="mt-2 flex items-center gap-2 text-sm">
                    <FileText className="size-4 text-brand-600" /> nmc_certificate.pdf
                    <Button size="sm" variant="ghost" className="ml-auto" icon={Eye}>
                      View
                    </Button>
                  </div>
                </div>
                <div className="mt-5 flex gap-2">
                  <Button
                    className="flex-1"
                    icon={Check}
                    onClick={() => decide(selected.id, "VERIFIED")}
                  >
                    Approve
                  </Button>
                  <Button
                    className="flex-1"
                    variant="danger"
                    icon={Ban}
                    onClick={() => decide(selected.id, "REJECTED")}
                  >
                    Reject
                  </Button>
                </div>
              </GlassCard>
            </motion.div>
          ) : (
            <GlassCard className="flex h-64 flex-col items-center justify-center text-center">
              <BadgeCheck className="size-8 text-brand-300" />
              <p className="mt-3 text-sm text-ink-soft">Select a doctor to review their licence.</p>
            </GlassCard>
          )}
        </div>
      </div>
    </>
  );
}

/* ----------------------------------------------------------- patients */
export function AdminPatients() {
  const [q, setQ] = useState("");
  const list = patients.filter(
    (p) => p.name.toLowerCase().includes(q.toLowerCase()) || p.email.includes(q.toLowerCase()),
  );
  return (
    <>
      <PageHead
        title="Patients"
        lead="Accounts, activity and support actions."
        actions={
          <div className="w-64">
            <Input
              name="q"
              icon={Search}
              placeholder="Search by name or email"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
        }
      />
      <Table head={["Patient", "Contact", "City", "Last report", "Risk", ""]}>
        {list.map((p) => (
          <tr key={p.id} className="hover:bg-white/50">
            <Td>
              <span className="flex items-center gap-3">
                <Avatar name={p.name} size={32} />
                <span>
                  <span className="block font-semibold">{p.name}</span>
                  <span className="block text-xs text-ink-muted">
                    {p.age} y · {p.sex}
                  </span>
                </span>
              </span>
            </Td>
            <Td>
              <span className="block text-xs">{p.email}</span>
              <span className="block text-xs text-ink-muted">{p.phone}</span>
            </Td>
            <Td>{p.city}</Td>
            <Td>{fmtDate(p.lastReport)}</Td>
            <Td>
              <Badge tone={p.risk === "High" ? "critical" : p.risk === "Medium" ? "warn" : "brand"}>
                {p.risk}
              </Badge>
            </Td>
            <Td className="text-right">
              <span className="inline-flex gap-1">
                <Button size="sm" variant="ghost" icon={Mail} aria-label="Email" />
                <Button size="sm" variant="ghost" icon={Trash2} aria-label="Delete" />
              </span>
            </Td>
          </tr>
        ))}
      </Table>
    </>
  );
}

/* -------------------------------------------------------- create admin */
export function CreateAdmin() {
  return (
    <div className="mx-auto max-w-2xl">
      <PageHead
        title="Create admin"
        lead="Sub-admins can only review doctor licences. Only a super-admin can create them."
      />
      <Reveal>
        <GlassCard strong>
          <form className="grid gap-4 sm:grid-cols-2" onSubmit={(e) => e.preventDefault()}>
            <Input label="First name" name="first" placeholder="Asha" />
            <Input label="Last name" name="last" placeholder="Rai" />
            <div className="sm:col-span-2">
              <Input
                label="Email"
                name="email"
                type="email"
                icon={Mail}
                placeholder="asha@mediscan.health"
              />
            </div>
            <div className="sm:col-span-2">
              <Input
                label="Temporary password"
                name="pw"
                type="password"
                hint="They will be asked to change it on first login."
              />
            </div>
            <div className="sm:col-span-2 rounded-2xl bg-brand-50 p-4 text-sm text-brand-900">
              <p className="font-bold">Permissions</p>
              <ul className="mt-2 space-y-1 text-xs">
                <li>✓ Review and decide doctor licences</li>
                <li>✓ View patient list (no report contents)</li>
                <li>✗ Financial dashboards</li>
                <li>✗ Create other admins</li>
              </ul>
            </div>
            <div className="sm:col-span-2">
              <Button type="submit" variant="secondary" icon={UserCog}>
                Create admin account
              </Button>
            </div>
          </form>
        </GlassCard>
      </Reveal>
    </div>
  );
}
