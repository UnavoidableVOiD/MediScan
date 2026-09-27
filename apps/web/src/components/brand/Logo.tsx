import { cn } from "@/lib/cn";

type Props = { className?: string; withWordmark?: boolean; dark?: boolean; size?: number };

/** MediScan mark: scan-frame corners + pulse line. Swap the SVG when a brand asset exists. */
export function LogoMark({ size = 36, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      className={className}
      aria-hidden
    >
      <defs>
        <linearGradient id="ms-g" x1="8" y1="6" x2="56" y2="58" gradientUnits="userSpaceOnUse">
          <stop stopColor="#34D399" />
          <stop offset="1" stopColor="#047857" />
        </linearGradient>
      </defs>
      <rect x="6" y="6" width="52" height="52" rx="16" fill="url(#ms-g)" />
      <path
        d="M18 24v-4a2 2 0 0 1 2-2h4M46 24v-4a2 2 0 0 0-2-2h-4M18 40v4a2 2 0 0 0 2 2h4M46 40v4a2 2 0 0 1-2 2h-4"
        stroke="#fff"
        strokeOpacity=".85"
        strokeWidth="2.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M15 32h9l3.2-8 5.6 16 3.8-10 2.4 4H49"
        stroke="#fff"
        strokeWidth="3.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function Logo({ className, withWordmark = true, dark = false, size = 36 }: Props) {
  return (
    <span className={cn("inline-flex items-center gap-2.5 select-none", className)}>
      <LogoMark size={size} />
      {withWordmark && (
        <span
          className={cn(
            "font-display text-xl font-bold tracking-tight leading-none",
            dark ? "text-white" : "text-ink",
          )}
        >
          Medi<span className={dark ? "text-brand-300" : "text-brand-600"}>Scan</span>
        </span>
      )}
    </span>
  );
}
