import { cn } from "@/lib/utils";

interface ProgressProps {
  /**
   * Null renders an indeterminate bar. The server reports null until the
   * diffusion process emits its first step, and showing a made-up percentage
   * there is what made the old UI feel hung.
   */
  value: number | null;
  className?: string;
  label?: string;
  tone?: "accent" | "gain" | "fault" | "muted";
}

const TONE_CLASS = {
  accent: "bg-accent",
  gain: "bg-gain",
  fault: "bg-fault",
  muted: "bg-ink-faint",
} as const;

export function Progress({ value, className, label, tone = "accent" }: ProgressProps) {
  const indeterminate = value === null;
  const percent = indeterminate ? 0 : Math.round(Math.min(Math.max(value, 0), 1) * 100);

  return (
    <div
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-canvas", className)}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : percent}
      aria-valuetext={indeterminate ? "Working, progress not yet reported" : `${percent}%`}
    >
      {indeterminate ? (
        <div className="h-full w-1/3 rounded-full bg-accent/70 motion-safe:animate-[indeterminate-sweep_1.4s_ease-in-out_infinite]" />
      ) : (
        <div
          className={cn("h-full rounded-full transition-[width] duration-300", TONE_CLASS[tone])}
          style={{ width: `${percent}%` }}
        />
      )}
    </div>
  );
}
