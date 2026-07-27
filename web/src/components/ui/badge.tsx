import { cva, type VariantProps } from "class-variance-authority";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import type { JobStatus } from "@/lib/types";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-medium whitespace-nowrap",
  {
    variants: {
      tone: {
        neutral: "border-stroke bg-surface-raised text-ink-muted",
        accent: "border-accent/40 bg-accent/10 text-accent",
        gain: "border-gain/40 bg-gain/10 text-gain",
        caution: "border-caution/40 bg-caution/10 text-caution",
        fault: "border-fault/40 bg-fault/10 text-fault",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export interface BadgeProps extends VariantProps<typeof badgeVariants> {
  children: ReactNode;
  className?: string;
}

export function Badge({ tone, className, children }: BadgeProps) {
  return <span className={cn(badgeVariants({ tone }), className)}>{children}</span>;
}

const STATUS_TONE: Record<JobStatus, NonNullable<BadgeProps["tone"]>> = {
  queued: "neutral",
  running: "accent",
  completed: "gain",
  failed: "fault",
  cancelled: "neutral",
  interrupted: "caution",
};

const STATUS_LABEL: Record<JobStatus, string> = {
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
  interrupted: "Interrupted",
};

export function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <Badge tone={STATUS_TONE[status]}>
      {/* The dot is redundant with the label on purpose: colour is never the
          only carrier of state. */}
      <span
        aria-hidden
        className={cn(
          "size-1.5 rounded-full",
          status === "running" ? "bg-accent" : "bg-current",
          status === "running" && "motion-safe:animate-pulse",
        )}
      />
      {STATUS_LABEL[status]}
    </Badge>
  );
}
