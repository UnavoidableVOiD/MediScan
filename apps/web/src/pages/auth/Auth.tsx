import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ArrowRight,
  KeyRound,
  Lock,
  Mail,
  Phone,
  ShieldCheck,
  Stethoscope,
  UserRound,
} from "lucide-react";
import { Logo } from "@/components/brand/Logo";
import { Button, Input } from "@/components/ui";
import { PageTransition } from "@/components/motion";
import { OrbScene } from "@/components/three/HeroScene";
import { cn } from "@/lib/cn";

function Shell({ children, aside }: { children: React.ReactNode; aside: React.ReactNode }) {
  return (
    <PageTransition className="min-h-screen">
      <div className="mx-auto grid min-h-screen max-w-7xl lg:grid-cols-2">
        <div className="relative hidden overflow-hidden lg:block">
          <div className="absolute inset-6 rounded-xl3 bg-gradient-to-br from-brand-700 via-brand-800 to-brand-950" />
          <OrbScene className="absolute inset-6" />
          <div className="absolute inset-6 flex flex-col justify-between p-12 text-white">
            <Link to="/">
              <Logo dark />
            </Link>
            {aside}
          </div>
        </div>
        <div className="flex items-center justify-center px-4 py-16 sm:px-10">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="w-full max-w-md"
          >
            <Link to="/" className="mb-8 inline-block lg:hidden">
              <Logo />
            </Link>
            {children}
          </motion.div>
        </div>
      </div>
    </PageTransition>
  );
}

const Aside = ({ title, body }: { title: string; body: string }) => (
  <div>
    <p className="text-xs font-bold uppercase tracking-[0.2em] text-brand-300">MediScan</p>
    <h2 className="mt-3 text-4xl font-bold leading-tight text-white">{title}</h2>
    <p className="mt-4 max-w-sm text-white/70">{body}</p>
    <div className="mt-8 flex gap-2 text-[11px] font-semibold text-white/60">
      <span className="rounded-full border border-white/20 px-3 py-1">OTP on every login</span>
      <span className="rounded-full border border-white/20 px-3 py-1">HttpOnly sessions</span>
    </div>
  </div>
);

export function Login() {
  const nav = useNavigate();
  return (
    <Shell
      aside={
        <Aside
          title="Welcome back."
          body="Sign in to see your reports, trends and upcoming consultations."
        />
      }
    >
      <h1 className="text-3xl font-bold">Sign in</h1>
      <p className="mt-2 text-sm text-ink-soft">
        New here?{" "}
        <Link to="/signup" className="font-semibold text-brand-700 hover:underline">
          Create an account
        </Link>
      </p>
      <form
        className="mt-8 space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          nav("/verify-otp");
        }}
      >
        <Input label="Email" name="email" type="email" icon={Mail} placeholder="you@example.com" />
        <Input
          label="Password"
          name="password"
          type="password"
          icon={Lock}
          placeholder="••••••••"
        />
        <Button type="submit" size="lg" className="w-full" icon={ArrowRight} trailing>
          Continue
        </Button>
        <button
          type="button"
          className="glass-pill flex h-12 w-full items-center justify-center gap-2 text-sm font-semibold text-ink hover:bg-white"
        >
          <svg viewBox="0 0 24 24" className="size-4" aria-hidden>
            <path
              fill="#EA4335"
              d="M12 10.2v3.9h5.5c-.2 1.3-1.6 3.9-5.5 3.9-3.3 0-6-2.7-6-6s2.7-6 6-6c1.9 0 3.1.8 3.9 1.5l2.6-2.6C16.9 3.3 14.7 2.4 12 2.4 6.7 2.4 2.4 6.7 2.4 12s4.3 9.6 9.6 9.6c5.5 0 9.2-3.9 9.2-9.4 0-.6-.1-1.1-.2-1.6H12z"
            />
          </svg>
          Continue with Google
        </button>
      </form>
      <p className="mt-6 text-center text-xs text-ink-muted">
        Building phase: any submission continues to the OTP screen.{" "}
        <Link to="/dashboard" className="font-semibold text-brand-700">
          Skip to dashboard →
        </Link>
      </p>
    </Shell>
  );
}

export function Signup() {
  const [role, setRole] = useState<"PATIENT" | "DOCTOR">("PATIENT");
  const nav = useNavigate();
  return (
    <Shell
      aside={
        <Aside
          title="Understand your health."
          body="Create an account to upload reports, track your values over time and reach verified doctors."
        />
      }
    >
      <h1 className="text-3xl font-bold">Create your account</h1>
      <p className="mt-2 text-sm text-ink-soft">
        Already registered?{" "}
        <Link to="/login" className="font-semibold text-brand-700 hover:underline">
          Sign in
        </Link>
      </p>
      <div className="mt-6 grid grid-cols-2 gap-2">
        {(["PATIENT", "DOCTOR"] as const).map((r) => (
          <button
            key={r}
            type="button"
            onClick={() => setRole(r)}
            className={cn(
              "flex items-center gap-3 rounded-2xl border p-4 text-left transition",
              role === r
                ? "border-brand-400 bg-brand-50 shadow-glow"
                : "border-white/80 bg-white/60 hover:bg-white",
            )}
          >
            <span
              className={cn(
                "grid size-10 place-items-center rounded-xl",
                role === r ? "bg-brand-600 text-white" : "bg-brand-100 text-brand-700",
              )}
            >
              {r === "PATIENT" ? (
                <UserRound className="size-5" />
              ) : (
                <Stethoscope className="size-5" />
              )}
            </span>
            <span>
              <span className="block text-sm font-bold">
                {r === "PATIENT" ? "Patient" : "Doctor"}
              </span>
              <span className="block text-[11px] text-ink-muted">
                {r === "PATIENT" ? "Upload & understand" : "Licence review required"}
              </span>
            </span>
          </button>
        ))}
      </div>
      <form
        className="mt-6 space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          nav("/verify-otp");
        }}
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <Input label="First name" name="first" placeholder="Prabhat" />
          <Input label="Last name" name="last" placeholder="Acharya" />
        </div>
        <Input label="Email" name="email" type="email" icon={Mail} placeholder="you@example.com" />
        <Input label="Phone" name="phone" icon={Phone} placeholder="+977 98XXXXXXXX" />
        <Input
          label="Password"
          name="password"
          type="password"
          icon={Lock}
          placeholder="At least 8 characters"
        />
        {role === "DOCTOR" && (
          <div className="rounded-2xl border border-brand-200 bg-brand-50 p-4 text-xs text-brand-900">
            After sign-up you will upload your NMC licence. Clinical features unlock once an
            administrator approves it.
          </div>
        )}
        <Button type="submit" size="lg" className="w-full" icon={ArrowRight} trailing>
          Create account
        </Button>
      </form>
    </Shell>
  );
}

export function VerifyOtp() {
  const [code, setCode] = useState(Array(6).fill(""));
  const refs = useRef<(HTMLInputElement | null)[]>([]);
  const nav = useNavigate();
  const { state } = useLocation() as { state?: { next?: string } };
  useEffect(() => refs.current[0]?.focus(), []);
  const onChange = (i: number, v: string) => {
    const d = v.replace(/\D/g, "").slice(-1);
    const next = [...code];
    next[i] = d;
    setCode(next);
    if (d && i < 5) refs.current[i + 1]?.focus();
  };
  return (
    <Shell
      aside={
        <Aside
          title="Check your inbox."
          body="We sent a six-digit code. It expires in 10 minutes."
        />
      }
    >
      <span className="grid size-14 place-items-center rounded-2xl bg-brand-100 text-brand-700">
        <KeyRound className="size-6" />
      </span>
      <h1 className="mt-6 text-3xl font-bold">Enter the code</h1>
      <p className="mt-2 text-sm text-ink-soft">Sent to you@example.com</p>
      <form
        className="mt-8"
        onSubmit={(e) => {
          e.preventDefault();
          nav(state?.next ?? "/dashboard");
        }}
      >
        <div className="flex justify-between gap-2">
          {code.map((c, i) => (
            <input
              key={i}
              ref={(el) => {
                refs.current[i] = el;
              }}
              value={c}
              onChange={(e) => onChange(i, e.target.value)}
              onKeyDown={(e) => e.key === "Backspace" && !c && refs.current[i - 1]?.focus()}
              inputMode="numeric"
              className="h-14 w-12 rounded-2xl border border-white/80 bg-white/70 text-center font-display text-2xl font-bold ring-focus focus:bg-white sm:w-14"
            />
          ))}
        </div>
        <Button type="submit" size="lg" className="mt-8 w-full" icon={ArrowRight} trailing>
          Verify
        </Button>
        <p className="mt-4 text-center text-xs text-ink-muted">
          Didn't get it?{" "}
          <button type="button" className="font-semibold text-brand-700">
            Resend in 0:42
          </button>
        </p>
      </form>
    </Shell>
  );
}

export function AdminLogin() {
  const nav = useNavigate();
  return (
    <Shell
      aside={
        <Aside
          title="Administrator access."
          body="Verify doctors, oversee the platform, and keep every access audited."
        />
      }
    >
      <span className="grid size-14 place-items-center rounded-2xl bg-brand-900 text-white">
        <ShieldCheck className="size-6" />
      </span>
      <h1 className="mt-6 text-3xl font-bold">Admin sign in</h1>
      <p className="mt-2 text-sm text-ink-soft">Staff accounts only. Attempts are logged.</p>
      <form
        className="mt-8 space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          nav("/admin/dashboard");
        }}
      >
        <Input
          label="Email"
          name="email"
          type="email"
          icon={Mail}
          placeholder="admin@mediscan.health"
        />
        <Input
          label="Password"
          name="password"
          type="password"
          icon={Lock}
          placeholder="••••••••"
        />
        <Button
          type="submit"
          size="lg"
          variant="secondary"
          className="w-full"
          icon={ArrowRight}
          trailing
        >
          Sign in
        </Button>
      </form>
    </Shell>
  );
}
