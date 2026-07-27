import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface SegmentedOption<T extends string> {
  value: T;
  label: ReactNode;
  /** Announced to screen readers when the visible label is an icon or glyph. */
  srLabel?: string;
  hint?: string;
}

interface SegmentedProps<T extends string> {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel: string;
  className?: string;
  size?: "sm" | "md";
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  className,
  size = "md",
}: SegmentedProps<T>) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn(
        "inline-flex items-center gap-1 rounded-lg border border-stroke bg-canvas p-1",
        className,
      )}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={option.hint}
            onClick={() => onChange(option.value)}
            className={cn(
              "cursor-pointer rounded-md font-medium transition-colors duration-150",
              size === "md" ? "min-h-9 px-3 text-sm" : "min-h-8 px-2.5 text-xs",
              active
                ? "bg-accent text-canvas"
                : "text-ink-muted hover:bg-surface-raised hover:text-ink",
            )}
          >
            {option.srLabel ? <span className="sr-only">{option.srLabel}</span> : null}
            <span aria-hidden={option.srLabel ? true : undefined}>{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}
