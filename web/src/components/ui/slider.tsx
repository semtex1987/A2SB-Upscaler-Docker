import { useId, type ReactNode } from "react";
import { cn } from "@/lib/utils";

interface SliderProps {
  label: string;
  /** Consequence of the setting, shown next to it rather than in a preamble. */
  hint?: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  disabled?: boolean;
  onChange: (value: number) => void;
}

export function Slider({
  label,
  hint,
  value,
  min,
  max,
  step = 1,
  unit,
  disabled,
  onChange,
}: SliderProps) {
  const id = useId();
  const percent = ((value - min) / (max - min)) * 100;

  return (
    <div className={cn("space-y-2", disabled && "opacity-50")}>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-sm font-medium text-ink">
          {label}
        </label>
        <span className="tnum font-mono text-sm text-ink">
          {value}
          {unit ? <span className="ml-0.5 text-ink-muted">{unit}</span> : null}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className={cn(
          "range-input h-11 w-full cursor-pointer appearance-none bg-transparent",
          "[&::-webkit-slider-runnable-track]:h-1.5 [&::-webkit-slider-runnable-track]:rounded-full",
          "[&::-moz-range-track]:h-1.5 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-canvas",
          "[&::-webkit-slider-thumb]:mt-[-7px] [&::-webkit-slider-thumb]:size-5 [&::-webkit-slider-thumb]:appearance-none",
          "[&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2",
          "[&::-webkit-slider-thumb]:border-canvas [&::-webkit-slider-thumb]:bg-accent",
          "[&::-moz-range-thumb]:size-5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-2",
          "[&::-moz-range-thumb]:border-canvas [&::-moz-range-thumb]:bg-accent",
          "disabled:cursor-not-allowed",
        )}
        style={{ ["--fill" as string]: `${percent}%` }}
      />
      {hint ? <p className="text-xs leading-relaxed text-ink-muted">{hint}</p> : null}
    </div>
  );
}
