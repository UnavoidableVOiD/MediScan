import { useEffect, useState } from "react";
import { Link, NavLink, Outlet } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Menu, X } from "lucide-react";
import { Logo } from "@/components/brand/Logo";
import { Button } from "@/components/ui";
import { ScrollProgress } from "@/components/motion";
import { ChatWidget } from "@/components/chat/ChatWidget";
import { cn } from "@/lib/cn";

const nav = [
  { to: "/services", label: "Services" },
  { to: "/doctors", label: "Doctors" },
  { to: "/about", label: "About" },
  { to: "/contact", label: "Contact" },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className="fixed inset-x-0 top-0 z-50 px-4 pt-4 sm:px-6">
      <motion.nav
        layout
        className={cn(
          "mx-auto flex max-w-7xl items-center justify-between rounded-full px-4 py-2.5 transition-all duration-500 sm:px-5",
          scrolled ? "glass-strong shadow-glass" : "bg-transparent",
        )}
      >
        <Link to="/" aria-label="MediScan home">
          <Logo />
        </Link>
        <div className="hidden items-center gap-1 md:flex">
          {nav.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              className={({ isActive }) =>
                cn(
                  "rounded-full px-4 py-2 text-sm font-semibold transition",
                  isActive
                    ? "bg-brand-100 text-brand-800"
                    : "text-ink-soft hover:bg-white/70 hover:text-ink",
                )
              }
            >
              {n.label}
            </NavLink>
          ))}
        </div>
        <div className="hidden items-center gap-2 md:flex">
          <Button to="/login" variant="ghost" size="sm">
            Sign in
          </Button>
          <Button to="/dashboard" size="sm" icon={ArrowRight} trailing>
            Open app
          </Button>
        </div>
        <button
          className="grid size-10 place-items-center rounded-full md:hidden"
          onClick={() => setOpen((o) => !o)}
          aria-label="Menu"
        >
          {open ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </motion.nav>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="glass-strong mx-auto mt-2 flex max-w-7xl flex-col gap-1 p-3 md:hidden"
          >
            {nav.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                onClick={() => setOpen(false)}
                className="rounded-2xl px-4 py-3 text-sm font-semibold text-ink hover:bg-brand-50"
              >
                {n.label}
              </NavLink>
            ))}
            <div className="mt-2 flex gap-2">
              <Button to="/login" variant="glass" className="flex-1" onClick={() => setOpen(false)}>
                Sign in
              </Button>
              <Button to="/dashboard" className="flex-1">
                Open app
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="relative mt-32 overflow-hidden">
      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <div className="glass-dark relative overflow-hidden px-8 py-14 sm:px-14">
          <div className="absolute -right-24 -top-24 size-72 rounded-full bg-brand-500/30 blur-3xl" />
          <div className="absolute -bottom-24 -left-16 size-72 rounded-full bg-teal/20 blur-3xl" />
          <div className="relative grid gap-10 md:grid-cols-[1.4fr_1fr_1fr_1fr]">
            <div>
              <Logo dark />
              <p className="mt-5 max-w-sm text-sm leading-relaxed text-white/70">
                Understand your lab report in plain language, know when something needs attention,
                and reach a verified doctor — in minutes.
              </p>
              <p className="mt-6 text-xs text-white/50">
                MediScan is a decision-support tool. It does not replace a clinician's judgement.
              </p>
            </div>
            <FooterCol
              title="Product"
              links={[
                ["Services", "/services"],
                ["Find doctors", "/doctors"],
                ["Upload a report", "/reports/upload"],
                ["Pages index", "/pages"],
              ]}
            />
            <FooterCol
              title="Company"
              links={[
                ["About", "/about"],
                ["Contact", "/contact"],
                ["Privacy", "/privacy"],
              ]}
            />
            <FooterCol
              title="Portals"
              links={[
                ["Patient", "/dashboard"],
                ["Doctor", "/doctor/dashboard"],
                ["Admin", "/admin/dashboard"],
              ]}
            />
          </div>
          <div className="relative mt-12 flex flex-col gap-3 border-t border-white/10 pt-6 text-xs text-white/50 sm:flex-row sm:items-center sm:justify-between">
            <span>© 2026 MediScan. Kathmandu, Nepal.</span>
            <span>Built with care for people who deserve to understand their own health.</span>
          </div>
        </div>
      </div>
      <div className="h-10" />
    </footer>
  );
}

function FooterCol({ title, links }: { title: string; links: [string, string][] }) {
  return (
    <div>
      <p className="text-xs font-bold uppercase tracking-[0.2em] text-brand-300">{title}</p>
      <ul className="mt-4 space-y-2.5">
        {links.map(([label, to]) => (
          <li key={to}>
            <Link to={to} className="text-sm text-white/75 transition hover:text-white">
              {label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function PublicLayout() {
  return (
    <>
      <ScrollProgress />
      <Navbar />
      <Outlet />
      <Footer />
      <ChatWidget />
    </>
  );
}
