import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MetricRow } from "@/lib/types";
import { cn } from "@/lib/utils";

const HEIGHT = 200;
const PAD = { top: 12, right: 14, bottom: 26, left: 52 };
/** Width used before the container has been measured. */
const FALLBACK_WIDTH = 640;

const SERIES_STYLE = [
  { key: "train_loss", label: "train", color: "var(--color-accent)" },
  { key: "val_loss", label: "validation", color: "var(--color-gain)" },
] as const;

type Scale = "linear" | "log";

interface Point {
  step: number;
  value: number;
}

interface Series {
  key: string;
  label: string;
  color: string;
  points: Point[];
}

/** Pull one numeric column out of the CSV rows, paired with its step. */
function toSeries(rows: MetricRow[], column: string): Point[] {
  const points: Point[] = [];
  for (const row of rows) {
    const step = row.step;
    const value = row[column];
    if (typeof step !== "number" || typeof value !== "number") continue;
    if (!Number.isFinite(step) || !Number.isFinite(value)) continue;
    points.push({ step, value });
  }
  // Lightning appends rows in order, but a resumed run can interleave them.
  return points.sort((a, b) => a.step - b.step);
}

function formatLoss(value: number): string {
  if (!Number.isFinite(value)) return "--";
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude < 1e-3 || magnitude >= 1e5) return value.toExponential(1);
  return String(Number(value.toPrecision(3)));
}

function formatStep(value: number): string {
  if (Math.abs(value) >= 1000) return `${Number((value / 1000).toPrecision(3))}k`;
  return String(Math.round(value));
}

/** Round a range outward to human tick values. */
function ticks(min: number, max: number, count: number): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
  const raw = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? magnitude * 10;
  const out: number[] = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-6; t += step) out.push(t);
  return out;
}

interface LossCurveProps {
  rows: MetricRow[];
  /** Split tag, shown as the chart's caption. */
  label: string;
  className?: string;
}

export function LossCurve({ rows, label, className }: LossCurveProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(FALLBACK_WIDTH);
  const [scale, setScale] = useState<Scale>("linear");
  const [hoverStep, setHoverStep] = useState<number | null>(null);

  // Measured rather than scaled by preserveAspectRatio, so that a pointer's x
  // maps to a step without correcting for letterboxing, and labels are not
  // stretched.
  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      const next = entry.contentRect.width;
      if (next > 0) setWidth(next);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const series: Series[] = useMemo(
    () =>
      SERIES_STYLE.map((style) => ({
        ...style,
        points: toSeries(rows, style.key),
      })).filter((s) => s.points.length > 0),
    [rows],
  );

  const bounds = useMemo(() => {
    const all = series.flatMap((s) => s.points);
    if (all.length === 0) return null;
    const steps = all.map((p) => p.step);
    const values = all.map((p) => p.value);
    const yMin = Math.min(...values);
    const yMax = Math.max(...values);
    // A flat series would collapse to zero height; give it a band to sit in.
    const spread = yMax - yMin || Math.abs(yMax) || 1;
    return {
      xMin: Math.min(...steps),
      xMax: Math.max(...steps),
      yMin: yMin - spread * 0.08,
      yMax: yMax + spread * 0.08,
      allPositive: yMin > 0,
    };
  }, [series]);

  const logActive = scale === "log" && bounds?.allPositive === true;

  const plot = useMemo(() => {
    if (!bounds) return null;
    const innerWidth = Math.max(width - PAD.left - PAD.right, 1);
    const innerHeight = HEIGHT - PAD.top - PAD.bottom;

    const project = (value: number) => (logActive ? Math.log10(value) : value);
    const yLow = logActive ? Math.log10(Math.max(bounds.yMin, Number.MIN_VALUE)) : bounds.yMin;
    const yHigh = logActive ? Math.log10(bounds.yMax) : bounds.yMax;
    const ySpan = yHigh - yLow || 1;
    const xSpan = bounds.xMax - bounds.xMin || 1;

    return {
      innerWidth,
      innerHeight,
      x: (step: number) => PAD.left + ((step - bounds.xMin) / xSpan) * innerWidth,
      y: (value: number) =>
        PAD.top + innerHeight - ((project(value) - yLow) / ySpan) * innerHeight,
      stepAt: (px: number) =>
        bounds.xMin + ((px - PAD.left) / innerWidth) * xSpan,
      // Losses usually span well under a decade, where rounding the exponent to
      // 4 steps leaves only one or two labels. Subdividing further recovers them.
      yTicks: logActive
        ? ticks(yLow, yHigh, 8).map((t) => 10 ** t)
        : ticks(bounds.yMin, bounds.yMax, 4),
      xTicks: ticks(bounds.xMin, bounds.xMax, 4),
    };
  }, [bounds, width, logActive]);

  const onPointerMove = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!plot) return;
      const rect = event.currentTarget.getBoundingClientRect();
      setHoverStep(plot.stepAt(event.clientX - rect.left));
    },
    [plot],
  );

  // Nearest sample per series to the hovered step, for the readout.
  const readout = useMemo(() => {
    if (hoverStep === null || series.length === 0) return null;
    const nearest = series.map((s) => {
      let best = s.points[0];
      for (const point of s.points) {
        if (Math.abs(point.step - hoverStep) < Math.abs(best.step - hoverStep)) best = point;
      }
      return { series: s, point: best };
    });
    return { step: nearest[0].point.step, nearest };
  }, [hoverStep, series]);

  if (!bounds || !plot) {
    return (
      <div
        ref={containerRef}
        className={cn(
          "flex h-32 items-center justify-center rounded-lg border border-stroke bg-canvas",
          className,
        )}
      >
        <p className="text-sm text-ink-faint">
          No loss values logged for {label} yet.
        </p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className={cn("rounded-lg border border-stroke bg-canvas", className)}>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-stroke px-3 py-2">
        <p className="text-sm font-medium text-ink">Split {label}</p>
        <div className="flex items-center gap-3">
          {series.map((s) => {
            const shown = readout?.nearest.find((n) => n.series.key === s.key)?.point;
            const latest = s.points[s.points.length - 1];
            return (
              <span key={s.key} className="flex items-center gap-1.5 text-xs text-ink-muted">
                <span
                  className="inline-block size-2 rounded-full"
                  style={{ backgroundColor: s.color }}
                  aria-hidden
                />
                {s.label}
                <span className="tnum font-mono text-ink">
                  {formatLoss((shown ?? latest).value)}
                </span>
              </span>
            );
          })}
        </div>
        <div className="ml-auto flex items-center gap-1 rounded-md border border-stroke p-0.5">
          {(["linear", "log"] as const).map((option) => (
            <button
              key={option}
              type="button"
              role="radio"
              aria-checked={scale === option}
              disabled={option === "log" && !bounds.allPositive}
              title={
                option === "log" && !bounds.allPositive
                  ? "A log axis needs every value to be positive"
                  : undefined
              }
              onClick={() => setScale(option)}
              className={cn(
                "min-h-7 cursor-pointer rounded px-2 text-xs font-medium transition-colors duration-150",
                scale === option
                  ? "bg-accent text-canvas"
                  : "text-ink-muted hover:text-ink disabled:cursor-not-allowed disabled:text-ink-faint disabled:hover:text-ink-faint",
              )}
            >
              {option}
            </button>
          ))}
        </div>
      </div>

      <svg
        width={width}
        height={HEIGHT}
        viewBox={`0 0 ${width} ${HEIGHT}`}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHoverStep(null)}
        className="block touch-none"
        role="img"
        aria-label={
          `Loss curves for split ${label}. ` +
          series
            .map(
              (s) =>
                `${s.label} ranges from ${formatLoss(s.points[0].value)} at step ` +
                `${s.points[0].step} to ${formatLoss(s.points[s.points.length - 1].value)} ` +
                `at step ${s.points[s.points.length - 1].step}`,
            )
            .join(". ")
        }
      >
        {plot.yTicks.map((tick) => {
          const y = plot.y(tick);
          if (y < PAD.top - 1 || y > HEIGHT - PAD.bottom + 1) return null;
          return (
            <g key={`y${tick}`}>
              <line
                x1={PAD.left}
                x2={width - PAD.right}
                y1={y}
                y2={y}
                stroke="var(--color-stroke)"
                strokeWidth={1}
              />
              <text
                x={PAD.left - 8}
                y={y}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize={10}
                fill="var(--color-ink-faint)"
                className="tnum"
              >
                {formatLoss(tick)}
              </text>
            </g>
          );
        })}

        {plot.xTicks.map((tick) => {
          const x = plot.x(tick);
          if (x < PAD.left - 1 || x > width - PAD.right + 1) return null;
          return (
            <text
              key={`x${tick}`}
              x={x}
              y={HEIGHT - PAD.bottom + 16}
              textAnchor="middle"
              fontSize={10}
              fill="var(--color-ink-faint)"
              className="tnum"
            >
              {formatStep(tick)}
            </text>
          );
        })}

        {series.map((s) => (
          <path
            key={s.key}
            d={s.points
              .map((p, i) => `${i === 0 ? "M" : "L"}${plot.x(p.step)},${plot.y(p.value)}`)
              .join(" ")}
            fill="none"
            stroke={s.color}
            strokeWidth={1.75}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}

        {/* A run with one sample draws no visible path, so mark the points. */}
        {series
          .filter((s) => s.points.length === 1)
          .map((s) => (
            <circle
              key={`dot${s.key}`}
              cx={plot.x(s.points[0].step)}
              cy={plot.y(s.points[0].value)}
              r={3}
              fill={s.color}
            />
          ))}

        {readout ? (
          <g>
            <line
              x1={plot.x(readout.step)}
              x2={plot.x(readout.step)}
              y1={PAD.top}
              y2={HEIGHT - PAD.bottom}
              stroke="var(--color-stroke-strong)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            {readout.nearest.map(({ series: s, point }) => (
              <circle
                key={`hover${s.key}`}
                cx={plot.x(point.step)}
                cy={plot.y(point.value)}
                r={3.5}
                fill={s.color}
                stroke="var(--color-canvas)"
                strokeWidth={1.5}
              />
            ))}
            <text
              x={plot.x(readout.step)}
              y={HEIGHT - PAD.bottom + 16}
              textAnchor="middle"
              fontSize={10}
              fill="var(--color-ink)"
              className="tnum"
            >
              step {formatStep(readout.step)}
            </text>
          </g>
        ) : null}
      </svg>
    </div>
  );
}
