import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn, formatDuration, formatHz } from "@/lib/utils";
import { decodeGrid, frequencyToRow, renderToCanvas, sampleDb } from "@/lib/spectrogram";
import type { SpectrogramPayload } from "@/lib/types";

export type BandFocus = "full" | "high";

interface SpectrogramProps {
  /** Drawn as the base layer. On the Evaluate view this is the restored output. */
  primary: SpectrogramPayload;
  /** Revealed by the wipe handle. On the Evaluate view this is the filtered input. */
  comparison?: SpectrogramPayload | null;
  primaryLabel: string;
  comparisonLabel?: string;
  cutoffHz: number;
  /** Supplying this makes the cutoff line draggable. */
  onCutoffChange?: (hz: number) => void;
  cutoffBounds?: { min: number; max: number };
  bandFocus: BandFocus;
  /** Lower edge of the high-band focus view. */
  focusFloorHz?: number;
  className?: string;
}

interface Hover {
  x: number;
  y: number;
  timeSec: number;
  frequencyHz: number;
  primaryDb: number;
  comparisonDb: number | null;
}

export function Spectrogram({
  primary,
  comparison,
  primaryLabel,
  comparisonLabel,
  cutoffHz,
  onCutoffChange,
  cutoffBounds = { min: 1000, max: 20000 },
  bandFocus,
  focusFloorHz = 8000,
  className,
}: SpectrogramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const [wipe, setWipe] = useState(0.5);
  const [hover, setHover] = useState<Hover | null>(null);
  const [draggingCutoff, setDraggingCutoff] = useState(false);

  const primaryGrid = useMemo(() => decodeGrid(primary), [primary]);
  const comparisonGrid = useMemo(
    () => (comparison ? decodeGrid(comparison) : null),
    [comparison],
  );
  const primaryLayer = useMemo(
    () => renderToCanvas(primary, primaryGrid),
    [primary, primaryGrid],
  );
  const comparisonLayer = useMemo(
    () => (comparison && comparisonGrid ? renderToCanvas(comparison, comparisonGrid) : null),
    [comparison, comparisonGrid],
  );

  const viewTop = bandFocus === "high" ? primary.maxFrequencyHz : primary.maxFrequencyHz;
  const viewBottom = bandFocus === "high" ? focusFloorHz : 0;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const frame = frameRef.current;
    if (!canvas || !frame) return;

    const { width, height } = frame.getBoundingClientRect();
    if (width === 0 || height === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);

    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, width, height);

    const srcY = frequencyToRow(viewTop, primary);
    const srcHeight = frequencyToRow(viewBottom, primary) - srcY;

    context.drawImage(primaryLayer, 0, srcY, primary.width, srcHeight, 0, 0, width, height);

    if (comparisonLayer && comparison) {
      const wipeX = Math.round(width * wipe);
      context.save();
      context.beginPath();
      context.rect(0, 0, wipeX, height);
      context.clip();
      const compY = frequencyToRow(viewTop, comparison);
      const compHeight = frequencyToRow(viewBottom, comparison) - compY;
      context.drawImage(
        comparisonLayer,
        0,
        compY,
        comparison.width,
        compHeight,
        0,
        0,
        width,
        height,
      );
      context.restore();

      context.strokeStyle = "rgba(241, 245, 249, 0.85)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(wipeX + 0.5, 0);
      context.lineTo(wipeX + 0.5, height);
      context.stroke();
    }
  }, [comparison, comparisonLayer, primary, primaryLayer, viewBottom, viewTop, wipe]);

  useEffect(() => {
    draw();
    const frame = frameRef.current;
    if (!frame) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(frame);
    return () => observer.disconnect();
  }, [draw]);

  const cutoffOffset = useMemo(() => {
    if (cutoffHz > viewTop || cutoffHz < viewBottom) return null;
    return ((viewTop - cutoffHz) / (viewTop - viewBottom)) * 100;
  }, [cutoffHz, viewBottom, viewTop]);

  const frequencyAt = useCallback(
    (fractionY: number) => viewTop - fractionY * (viewTop - viewBottom),
    [viewBottom, viewTop],
  );

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const frame = frameRef.current;
    if (!frame) return;
    const rect = frame.getBoundingClientRect();
    const fractionX = Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1);
    const fractionY = Math.min(Math.max((event.clientY - rect.top) / rect.height, 0), 1);

    if (draggingCutoff && onCutoffChange) {
      const hz = Math.round(frequencyAt(fractionY) / 100) * 100;
      onCutoffChange(Math.min(Math.max(hz, cutoffBounds.min), cutoffBounds.max));
      return;
    }

    const frequencyHz = frequencyAt(fractionY);
    const gridFractionY = 1 - frequencyHz / primary.maxFrequencyHz;
    setHover({
      x: fractionX * rect.width,
      y: fractionY * rect.height,
      timeSec: fractionX * primary.durationSec,
      frequencyHz,
      primaryDb: sampleDb(primary, primaryGrid, fractionX, gridFractionY),
      comparisonDb:
        comparison && comparisonGrid
          ? sampleDb(comparison, comparisonGrid, fractionX, gridFractionY)
          : null,
    });
  };

  const stopDrag = () => setDraggingCutoff(false);

  const description =
    `Spectrogram of ${primaryLabel}, ${formatHz(viewBottom)} to ${formatHz(viewTop)}, ` +
    `${formatDuration(primary.durationSec)} long. Brighter means more energy.`;

  return (
    <figure className={cn("space-y-2", className)}>
      <div className="flex gap-2">
        <FrequencyAxis top={viewTop} bottom={viewBottom} />

        <div
          ref={frameRef}
          role="img"
          aria-label={description}
          onPointerMove={handlePointerMove}
          onPointerLeave={() => {
            setHover(null);
            stopDrag();
          }}
          onPointerUp={stopDrag}
          className={cn(
            "relative min-h-64 flex-1 overflow-hidden rounded-lg border border-stroke bg-canvas",
            draggingCutoff ? "cursor-grabbing" : "cursor-crosshair",
          )}
          style={{ aspectRatio: "16 / 7" }}
        >
          <canvas ref={canvasRef} className="block size-full" />

          {cutoffOffset !== null ? (
            <div
              className="pointer-events-none absolute inset-x-0 flex items-center"
              style={{ top: `${cutoffOffset}%` }}
            >
              <div className="h-px w-full bg-accent/90" />
              <span className="absolute right-2 -translate-y-1/2 rounded bg-canvas/85 px-1.5 py-0.5 font-mono text-[11px] text-accent tnum">
                cutoff {formatHz(cutoffHz)}
              </span>
              {onCutoffChange ? (
                <button
                  type="button"
                  aria-label={`Drag to change cutoff, currently ${formatHz(cutoffHz)}`}
                  onPointerDown={(event) => {
                    event.preventDefault();
                    setDraggingCutoff(true);
                  }}
                  className="pointer-events-auto absolute left-2 size-5 -translate-y-1/2 cursor-grab rounded-full border-2 border-canvas bg-accent active:cursor-grabbing"
                />
              ) : null}
            </div>
          ) : null}

          {hover && !draggingCutoff ? <HoverReadout hover={hover} comparison={!!comparison} /> : null}
        </div>
      </div>

      <TimeAxis duration={primary.durationSec} />

      {comparison ? (
        <div className="flex items-center gap-3 pl-16">
          <span className="text-xs font-medium whitespace-nowrap text-ink-muted">
            {comparisonLabel ?? "Input"}
          </span>
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(wipe * 100)}
            onChange={(event) => setWipe(Number(event.target.value) / 100)}
            aria-label="Wipe between input and restored spectrogram"
            className="range-input h-11 flex-1 cursor-pointer appearance-none bg-transparent [&::-webkit-slider-runnable-track]:h-1.5 [&::-webkit-slider-runnable-track]:rounded-full [&::-webkit-slider-thumb]:mt-[-7px] [&::-webkit-slider-thumb]:size-5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-canvas [&::-webkit-slider-thumb]:bg-accent [&::-moz-range-thumb]:size-5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-canvas [&::-moz-range-thumb]:bg-accent [&::-moz-range-track]:h-1.5 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-canvas"
            style={{ ["--fill" as string]: `${wipe * 100}%` }}
          />
          <span className="text-xs font-medium whitespace-nowrap text-ink-muted">
            {primaryLabel}
          </span>
        </div>
      ) : null}
    </figure>
  );
}

function FrequencyAxis({ top, bottom }: { top: number; bottom: number }) {
  const ticks = useMemo(() => {
    const span = top - bottom;
    const count = 6;
    return Array.from({ length: count + 1 }, (_, index) => {
      const hz = top - (span * index) / count;
      return { hz, offset: (index / count) * 100 };
    });
  }, [bottom, top]);

  return (
    <div className="relative w-14 shrink-0" aria-hidden>
      {ticks.map((tick) => (
        <span
          key={tick.hz}
          className="absolute right-0 -translate-y-1/2 font-mono text-[11px] text-ink-faint tnum"
          style={{ top: `${tick.offset}%` }}
        >
          {(tick.hz / 1000).toFixed(1)}k
        </span>
      ))}
    </div>
  );
}

function TimeAxis({ duration }: { duration: number }) {
  const ticks = Array.from({ length: 7 }, (_, index) => (duration * index) / 6);
  return (
    <div className="flex justify-between pl-16" aria-hidden>
      {ticks.map((seconds, index) => (
        <span key={index} className="font-mono text-[11px] text-ink-faint tnum">
          {formatDuration(seconds)}
        </span>
      ))}
    </div>
  );
}

function HoverReadout({ hover, comparison }: { hover: Hover; comparison: boolean }) {
  // Flip the tooltip across the pointer so it never leaves the frame.
  const flipX = hover.x > 220;
  const flipY = hover.y < 90;
  return (
    <>
      <div
        className="pointer-events-none absolute inset-y-0 w-px bg-ink/25"
        style={{ left: hover.x }}
      />
      <div
        className="pointer-events-none absolute inset-x-0 h-px bg-ink/25"
        style={{ top: hover.y }}
      />
      <div
        className="pointer-events-none absolute z-10 rounded-md border border-stroke bg-canvas/95 px-2.5 py-1.5 font-mono text-[11px] leading-relaxed text-ink tnum"
        style={{
          left: hover.x,
          top: hover.y,
          transform: `translate(${flipX ? "calc(-100% - 12px)" : "12px"}, ${flipY ? "12px" : "calc(-100% - 12px)"})`,
        }}
      >
        <div className="text-ink-muted">
          {formatDuration(hover.timeSec)} · {formatHz(hover.frequencyHz)}
        </div>
        <div>{hover.primaryDb.toFixed(1)} dB</div>
        {comparison && hover.comparisonDb !== null ? (
          <div className="text-ink-muted">input {hover.comparisonDb.toFixed(1)} dB</div>
        ) : null}
      </div>
    </>
  );
}
