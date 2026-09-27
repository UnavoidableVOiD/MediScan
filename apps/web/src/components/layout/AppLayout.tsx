import { useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  BadgeCheck,
  Bell,
  CalendarClock,
  ChevronLeft,
  FileText,
  Home,
  LayoutDashboard,
  LogOut,
  Menu,
  Search,
  ShieldCheck,
  Stethoscope,
  UploadCloud,
  UserCog,
  UserRound,
  Users,
  X,
  type LucideIcon,
} from "lucide-react";
import { Logo, LogoMark } from "@/components/brand/Logo";
import { Avatar, Badge } from "@/components/ui";
import { ScrollProgress } from "@/components/motion";
import { ChatWidget } from "@/components/chat/ChatWidget";
import { cn } from "@/lib/cn";

type Role = "patient" | "doctor" | "admin";
type Item = { to: string; label: string; icon: LucideIcon; end?: boolean };

const menus: Record<Role, { title: string; user: string; items: Item[] }> = {
  patient: {
    title: "Patient",
    user: "Prabhat Acharya",
    items: [
      { to: "/dashboard", label: "Overview", icon: LayoutDashboard, end: true },
      { to: "/reports/upload", label: "Upload report", icon: UploadCloud },
      { to: "/reports", label: "My reports", icon: FileText, end: true },
      { to: "/doctors", label: "Find a doctor", icon: Stethoscope },
      { to: "/appointments", label: "Appointments", icon: CalendarClock },
      { to: "/profile", label: "Profile", icon: UserRound },
    ],
  },
  doctor: {
    title: "Doctor",
    user: "Dr. Sunita Karki",
    items: [
      { to: "/doctor/dashboard", label: "Overview", icon: LayoutDashboard },
      { to: "/doctor/appointments", label: "Appointments", icon: CalendarClock },
      { to: "/doctor/patients", label: "Patients", icon: Users },
      { to: "/doctor/availability", label: "Availability", icon: Activity },
      { to: "/doctor/verify", label: "Verification", icon: BadgeCheck },
      { to: "/doctor/profile", label: "Profile", icon: UserRound },
    ],
  },
  admin: {
    title: "Admin",
    user: "System Admin",
    items: [
      { to: "/admin/dashboard", label: "Overview", icon: LayoutDashboard },
      { to: "/admin/doctors", label: "Doctors", icon: Stethoscope },
      { to: "/admin/patients", label: "Patients", icon: Users },
      { to: "/admin/create-admin", label: "Create admin", icon: UserCog },
    ],
  },
};

type SidebarProps = { role: Role; compact: boolean; onCollapse: () => void };

function Sidebar({ role, compact, onCollapse }: SidebarProps) {
  const menu = menus[role];
  return (
    <div className="flex h-full flex-col">
      <div
        className={cn(
          "flex items-center px-3 pb-6 pt-2",
          compact ? "justify-center" : "justify-between",
        )}
      >
        <Link to="/">{compact ? <LogoMark size={34} /> : <Logo />}</Link>
        {!compact && (
          <button
            onClick={() => onCollapse()}
            className="hidden size-8 place-items-center rounded-full text-ink-muted hover:bg-brand-50 lg:grid"
            aria-label="Collapse"
          >
            <ChevronLeft className="size-4" />
          </button>
        )}
      </div>
      {!compact && (
        <div className="mx-3 mb-4 rounded-2xl bg-brand-900 p-3 text-white">
          <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-brand-300">
            {menu.title} portal
          </p>
          <p className="mt-1 text-sm font-semibold">{menu.user}</p>
        </div>
      )}
      <nav className="flex-1 space-y-1 px-2">
        {menu.items.map((it) => (
          <NavLink
            key={it.to}
            to={it.to}
            end={it.end}
            title={it.label}
            className={({ isActive }) =>
              cn(
                "relative flex items-center gap-3 rounded-2xl px-3 py-2.5 text-sm font-semibold transition",
                compact && "justify-center px-0",
                isActive ? "text-brand-900" : "text-ink-soft hover:bg-white/70 hover:text-ink",
              )
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.span
                    layoutId={`side-${role}`}
                    className="absolute inset-0 rounded-2xl bg-brand-100"
                    transition={{ type: "spring", stiffness: 380, damping: 32 }}
                  />
                )}
                <it.icon className="relative z-10 size-[18px]" />
                {!compact && <span className="relative z-10">{it.label}</span>}
              </>
            )}
          </NavLink>
        ))}
      </nav>
      <div className="px-2 pb-2">
        <Link
          to="/"
          className={cn(
            "flex items-center gap-3 rounded-2xl px-3 py-2.5 text-sm font-semibold text-ink-soft hover:bg-white/70",
            compact && "justify-center px-0",
          )}
        >
          <Home className="size-[18px]" />
          {!compact && "Public site"}
        </Link>
        <Link
          to="/login"
          className={cn(
            "flex items-center gap-3 rounded-2xl px-3 py-2.5 text-sm font-semibold text-ink-soft hover:bg-white/70",
            compact && "justify-center px-0",
          )}
        >
          <LogOut className="size-[18px]" />
          {!compact && "Sign out"}
        </Link>
      </div>
    </div>
  );
}

export function AppLayout({ role }: { role: Role }) {
  const menu = menus[role];
  const [collapsed, setCollapsed] = useState(false);
  const [mobile, setMobile] = useState(false);
  const { pathname } = useLocation();

  return (
    <div className="min-h-screen">
      <ScrollProgress />
      {/* desktop sidebar */}
      <motion.aside
        animate={{ width: collapsed ? 76 : 264 }}
        transition={{ type: "spring", stiffness: 260, damping: 30 }}
        className="glass-strong fixed left-4 top-4 z-40 hidden h-[calc(100vh-2rem)] overflow-hidden py-4 lg:block"
      >
        <Sidebar role={role} compact={collapsed} onCollapse={() => setCollapsed(true)} />
        {collapsed && (
          <button
            onClick={() => setCollapsed(false)}
            className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full bg-brand-100 p-1.5 text-brand-800"
            aria-label="Expand"
          >
            <Menu className="size-4" />
          </button>
        )}
      </motion.aside>

      {/* mobile drawer */}
      <AnimatePresence>
        {mobile && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 bg-ink/30 backdrop-blur-sm lg:hidden"
              onClick={() => setMobile(false)}
            />
            <motion.aside
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
              className="glass-strong fixed left-3 top-3 z-50 h-[calc(100vh-1.5rem)] w-64 py-4 lg:hidden"
            >
              <button
                onClick={() => setMobile(false)}
                className="absolute right-3 top-3 grid size-8 place-items-center rounded-full hover:bg-brand-50"
                aria-label="Close"
              >
                <X className="size-4" />
              </button>
              <Sidebar role={role} compact={false} onCollapse={() => setCollapsed(true)} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      <div
        className={cn(
          "transition-[padding] duration-500",
          collapsed ? "lg:pl-[108px]" : "lg:pl-[296px]",
        )}
      >
        {/* top bar */}
        <header className="sticky top-0 z-30 px-4 pt-4 sm:px-6">
          <div className="glass-strong flex items-center justify-between gap-3 px-4 py-2.5">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setMobile(true)}
                className="grid size-9 place-items-center rounded-full hover:bg-brand-50 lg:hidden"
                aria-label="Menu"
              >
                <Menu className="size-5" />
              </button>
              <div className="hidden items-center gap-2 rounded-full bg-white/70 px-3 py-2 text-sm text-ink-muted sm:flex">
                <Search className="size-4" />
                <span className="w-56">Search reports, doctors…</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge tone="info" dot>
                Building phase · no auth
              </Badge>
              <button
                className="relative grid size-9 place-items-center rounded-full hover:bg-brand-50"
                aria-label="Notifications"
              >
                <Bell className="size-[18px]" />
                <span className="absolute right-2 top-2 size-2 rounded-full bg-brand-500" />
              </button>
              <Avatar name={menu.user} size={36} />
            </div>
          </div>
        </header>
        <AnimatePresence mode="wait">
          <motion.main
            key={pathname}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.4 }}
            className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6"
          >
            <Outlet />
          </motion.main>
        </AnimatePresence>
        <div className="h-8" />
      </div>
      {role === "patient" && <ChatWidget />}
      {role !== "patient" && (
        <span className="fixed bottom-4 right-4 hidden items-center gap-1 text-[10px] font-semibold text-ink-muted sm:flex">
          <ShieldCheck className="size-3" /> PHI access is audited
        </span>
      )}
    </div>
  );
}
