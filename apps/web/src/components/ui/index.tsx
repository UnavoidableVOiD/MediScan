import {
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  forwardRef,
} from "react";
import { Link } from "react-router-dom";
import { motion, type HTMLMotionProps } from "framer-motion";
import { ArrowUpRight, ChevronRight, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/cn";

/* ------------------------------------------------------------------ Button */
type Variant = "primary" | "secondary" | "ghost" | "danger" | "glass";
type Size = "sm" | "md" | "lg";

const variants: Record<Variant, string> = {
  primary:
    "bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-[0_10px_30px_-10px_rgb(16_185_129/0.6)] hover:from-brand-400 hover:to-brand-600",
  secondary: "bg-ink text-white hover:bg-brand-900",
  ghost: "bg-transparent text-ink-soft hover:bg-brand-50 hover:text-brand-800",
  danger: "bg-critical text-white hover:bg-red-700",
  glass: "glass-pill text-ink hover:bg-white/90",
};
const sizes: Record<Size, string> = {
  sm: "h-9 px-4 text-sm gap-1.5",
  md: "h-11 px-5 text-sm gap-2",
  lg: "h-13 px-7 text-base gap-2.5",
};

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
  to?: string;
  icon?: LucideIcon;
  trailing?: boolean;
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { className, variant = "primary", size = "md", to, icon: Icon, trailing, children, ...rest },
    ref,
  ) => {
    const cls = cn(
      "inline-flex items-center justify-center rounded-full font-semibold transition-all duration-300 ring-focus disabled:opacity-50 disabled:pointer-events-none active:scale-[0.98]",
      variants[variant],
      sizes[size],
      className,
    );
    const inner = (
      <>
        {Icon && !trailing && <Icon className="size-4" />}
        {children}
        {Icon && trailing && <Icon className="size-4" />}
      </>
    );
    if (to) {
      return (
        <Link to={to} className={cls}>
          {inner}
        </Link>
      );
    }
    return (
      <button ref={ref} className={cls} {...rest}>
        {inner}
      </button>
    );
  },
);
Button.displayName = "Button";

/* --------------------------------------------------------------- GlassCard */
type CardProps = HTMLMotionProps<"div"> & {
  strong?: boolean;
  dark?: boolean;
  hover?: boolean;
  padded?: boolean;
};

export function GlassCard({
  className,
  strong,
  dark,
  hover,
  padded = true,
  children,
  ...rest
}: CardProps) {
  return (
    <motion.div
      className={cn(
        dark ? "glass-dark" : strong ? "glass-strong" : "glass",
        padded && "p-6",
        hover && "transition-transform duration-500 hover:-translate-y-1 hover:shadow-glow",
        className,
      )}
      {...rest}
    >
      {children}
    </motion.div>
  );
}

/* ------------------------------------------------------------------- Badge */
type Tone = "brand" | "neutral" | "critical" | "warn" | "info";
const tones: Record<Tone, string> = {
  brand: "bg-brand-100 text-brand-800 border-brand-200",
  neutral: "bg-white/70 text-ink-soft border-line",
  critical: "bg-red-50 text-red-700 border-red-200",
  warn: "bg-amber-50 text-amber-800 border-amber-200",
  info: "bg-teal-50 text-teal-800 border-teal-200",
};
export function Badge({
  tone = "brand",
  className,
  children,
  dot,
}: {
  tone?: Tone;
  className?: string;
  children: ReactNode;
  dot?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wider",
        tones[tone],
        className,
      )}
    >
      {dot && <span className="size-1.5 rounded-full bg-current" />}
      {children}
    </span>
  );
}

/* ------------------------------------------------------------------- Input */
type InputProps = InputHTMLAttributes<HTMLInputElement> & {
  label?: string;
  hint?: string;
  icon?: LucideIcon;
};
export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, hint, icon: Icon, className, id, ...rest }, ref) => {
    const inputId = id ?? rest.name;
    return (
      <label className="block" htmlFor={inputId}>
        {label && (
          <span className="mb-1.5 block text-xs font-bold uppercase tracking-wider text-ink-muted">
            {label}
          </span>
        )}
        <span className="relative block">
          {Icon && (
            <Icon className="pointer-events-none absolute left-4 top-1/2 size-4 -translate-y-1/2 text-ink-muted" />
          )}
          <input
            ref={ref}
            id={inputId}
            className={cn(
              "h-12 w-full rounded-2xl border border-white/80 bg-white/70 px-4 text-sm text-ink placeholder:text-ink-muted/70 backdrop-blur ring-focus transition focus:bg-white",
              Icon && "pl-11",
              className,
            )}
            {...rest}
          />
        </span>
        {hint && <span className="mt-1.5 block text-xs text-ink-muted">{hint}</span>}
      </label>
    );
  },
);
Input.displayName = "Input";

export function Textarea({
  label,
  className,
  ...rest
}: React.TextareaHTMLAttributes<HTMLTextAreaElement> & { label?: string }) {
  return (
    <label className="block">
      {label && (
        <span className="mb-1.5 block text-xs font-bold uppercase tracking-wider text-ink-muted">
          {label}
        </span>
      )}
      <textarea
        className={cn(
          "min-h-28 w-full rounded-2xl border border-white/80 bg-white/70 px-4 py-3 text-sm text-ink placeholder:text-ink-muted/70 backdrop-blur ring-focus transition focus:bg-white",
          className,
        )}
        {...rest}
      />
    </label>
  );
}

/* -------------------------------------------------------------------- Stat */
export function Stat({
  label,
  value,
  delta,
  icon: Icon,
  className,
}: {
  label: string;
  value: ReactNode;
  delta?: string;
  icon?: LucideIcon;
  className?: string;
}) {
  return (
    <GlassCard className={cn("relative overflow-hidden", className)} hover>
      <div className="flex items-start justify-between">
        <p className="text-xs font-bold uppercase tracking-wider text-ink-muted">{label}</p>
        {Icon && (
          <span className="grid size-9 place-items-center rounded-xl bg-brand-100 text-brand-700">
            <Icon className="size-4" />
          </span>
        )}
      </div>
      <p className="mt-3 font-display text-3xl font-bold text-ink">{value}</p>
      {delta && <p className="mt-1 text-xs font-semibold text-brand-700">{delta}</p>}
      <span className="pointer-events-none absolute -bottom-8 -right-8 size-28 rounded-full bg-brand-200/40 blur-2xl" />
    </GlassCard>
  );
}

/* ---------------------------------------------------------- SectionHeading */
export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <p
      className={cn(
        "inline-flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.2em] text-brand-700",
        className,
      )}
    >
      <span className="h-px w-6 bg-brand-500" />
      {children}
    </p>
  );
}

export function SectionHeading({
  eyebrow,
  title,
  lead,
  align = "left",
  dark,
  className,
}: {
  eyebrow?: string;
  title: ReactNode;
  lead?: ReactNode;
  align?: "left" | "center";
  dark?: boolean;
  className?: string;
}) {
  return (
    <div className={cn("max-w-2xl", align === "center" && "mx-auto text-center", className)}>
      {eyebrow && (
        <Eyebrow
          className={cn("mb-4", align === "center" && "justify-center", dark && "text-brand-300")}
        >
          {eyebrow}
        </Eyebrow>
      )}
      <h2
        className={cn(
          "text-3xl font-bold leading-[1.1] sm:text-4xl lg:text-5xl",
          dark && "text-white",
        )}
      >
        {title}
      </h2>
      {lead && (
        <p
          className={cn(
            "mt-5 text-base leading-relaxed sm:text-lg",
            dark ? "text-white/70" : "text-ink-soft",
          )}
        >
          {lead}
        </p>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------- PageHead */
export function PageHead({
  title,
  lead,
  actions,
  crumbs,
}: {
  title: ReactNode;
  lead?: ReactNode;
  actions?: ReactNode;
  crumbs?: string[];
}) {
  return (
    <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        {crumbs && (
          <p className="mb-2 flex items-center gap-1 text-xs font-semibold text-ink-muted">
            {crumbs.map((c, i) => (
              <span key={c} className="flex items-center gap-1">
                {i > 0 && <ChevronRight className="size-3" />}
                {c}
              </span>
            ))}
          </p>
        )}
        <h1 className="text-3xl font-bold sm:text-4xl">{title}</h1>
        {lead && <p className="mt-2 max-w-2xl text-ink-soft">{lead}</p>}
      </div>
      {actions && <div className="flex shrink-0 gap-2">{actions}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------- Table */
export function Table({
  head,
  children,
  className,
}: {
  head: ReactNode[];
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("glass overflow-hidden p-0", className)}>
      <div className="overflow-x-auto scrollbar-thin">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-line bg-white/50 text-[11px] font-bold uppercase tracking-wider text-ink-muted">
              {head.map((h, i) => (
                <th key={i} className="px-5 py-3.5 font-bold">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-line">{children}</tbody>
        </table>
      </div>
    </div>
  );
}
export const Td = ({ children, className }: { children: ReactNode; className?: string }) => (
  <td className={cn("px-5 py-3.5 align-middle", className)}>{children}</td>
);

/* ------------------------------------------------------------------ Avatar */
export function Avatar({
  name,
  size = 40,
  className,
}: {
  name: string;
  size?: number;
  className?: string;
}) {
  const initials = name
    .split(" ")
    .map((p) => p[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
  return (
    <span
      style={{ width: size, height: size, fontSize: size / 2.6 }}
      className={cn(
        "grid shrink-0 place-items-center rounded-full bg-gradient-to-br from-brand-200 to-brand-400 font-display font-bold text-brand-900",
        className,
      )}
    >
      {initials}
    </span>
  );
}

/* ---------------------------------------------------------------- Progress */
export function Progress({
  value,
  tone = "brand",
  className,
}: {
  value: number;
  tone?: "brand" | "critical" | "warn";
  className?: string;
}) {
  const bar =
    tone === "critical"
      ? "from-red-400 to-red-600"
      : tone === "warn"
        ? "from-amber-300 to-amber-500"
        : "from-brand-400 to-brand-600";
  return (
    <div className={cn("h-2 w-full overflow-hidden rounded-full bg-brand-900/10", className)}>
      <motion.div
        initial={{ width: 0 }}
        whileInView={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        viewport={{ once: true }}
        transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
        className={cn("h-full rounded-full bg-gradient-to-r", bar)}
      />
    </div>
  );
}

/* -------------------------------------------------------------- EmptyState */
export function EmptyState({
  icon: Icon,
  title,
  body,
  action,
}: {
  icon: LucideIcon;
  title: string;
  body?: string;
  action?: ReactNode;
}) {
  return (
    <div className="glass flex flex-col items-center px-6 py-14 text-center">
      <span className="grid size-14 place-items-center rounded-2xl bg-brand-100 text-brand-700">
        <Icon className="size-6" />
      </span>
      <h3 className="mt-5 text-lg font-bold">{title}</h3>
      {body && <p className="mt-1 max-w-sm text-sm text-ink-soft">{body}</p>}
      {action && <div className="mt-6">{action}</div>}
    </div>
  );
}

/* ---------------------------------------------------------------- LinkCard */
export function LinkCard({
  to,
  title,
  body,
  icon: Icon,
}: {
  to: string;
  title: string;
  body: string;
  icon: LucideIcon;
}) {
  return (
    <Link
      to={to}
      className="group glass block p-6 transition-all duration-500 hover:-translate-y-1 hover:shadow-glow"
    >
      <div className="flex items-start justify-between">
        <span className="grid size-11 place-items-center rounded-2xl bg-brand-100 text-brand-700 transition group-hover:bg-brand-600 group-hover:text-white">
          <Icon className="size-5" />
        </span>
        <ArrowUpRight className="size-5 text-ink-muted transition group-hover:text-brand-700" />
      </div>
      <h3 className="mt-5 text-lg font-bold">{title}</h3>
      <p className="mt-1 text-sm text-ink-soft">{body}</p>
    </Link>
  );
}

/* ------------------------------------------------------------------- Tabs */
export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
}: {
  tabs: { id: T; label: string; count?: number }[];
  value: T;
  onChange: (t: T) => void;
}) {
  return (
    <div className="glass-pill inline-flex p-1">
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={cn(
            "relative rounded-full px-4 py-2 text-sm font-semibold transition",
            value === t.id ? "text-white" : "text-ink-soft hover:text-ink",
          )}
        >
          {value === t.id && (
            <motion.span
              layoutId="tab-pill"
              className="absolute inset-0 rounded-full bg-gradient-to-br from-brand-500 to-brand-700"
              transition={{ type: "spring", stiffness: 400, damping: 32 }}
            />
          )}
          <span className="relative z-10 inline-flex items-center gap-2">
            {t.label}
            {t.count !== undefined && (
              <span
                className={cn(
                  "rounded-full px-1.5 text-[10px]",
                  value === t.id ? "bg-white/25" : "bg-brand-100 text-brand-800",
                )}
              >
                {t.count}
              </span>
            )}
          </span>
        </button>
      ))}
    </div>
  );
}
