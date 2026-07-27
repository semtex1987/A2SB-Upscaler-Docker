import { useCallback, useEffect, useRef } from "react";
import { cn, formatDuration } from "@/lib/utils";

interface WaveformProps {
  peaks: number[];
  duration: number;
  currentTime: number;
  onSeek: (seconds: number) => void;
  className?: string;
}

const BAR_WIDTH = 2;
const BAR_GAP = 1;

export function Waveform({ peaks, duration, currentTime, onSeek, className }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const progress = duration > 0 ? Math.min(currentTime / duration, 1) : 0;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || peaks.length === 0) return;

    const { width, height } = container.getBoundingClientRect();
    if (width === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);

    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);

    const step = BAR_WIDTH + BAR_GAP;
    const barCount = Math.max(1, Math.floor(width / step));
    const middle = height / 2;
    const playedX = width * progress;

    for (let index = 0; index < barCount; index += 1) {
      // Peak-pick within the bucket so transients survive the downsample.
      const from = Math.floor((index / barCount) * peaks.length);
      const to = Math.max(from + 1, Math.floor(((index + 1) / barCount) * peaks.length));
      let amplitude = 0;
      for (let p = from; p < to && p < peaks.length; p += 1) {
        if (peaks[p] > amplitude) amplitude = peaks[p];
      }

      const x = index * step;
      const barHeight = Math.max(2, amplitude * (height - 4));
      context.fillStyle = x + BAR_WIDTH <= playedX ? "#3b82f6" : "#475569";
      context.fillRect(x, middle - barHeight / 2, BAR_WIDTH, barHeight);
    }

    context.fillStyle = "#f1f5f9";
    context.fillRect(Math.min(playedX, width - 2), 0, 2, height);
  }, [peaks, progress]);

  useEffect(() => {
    draw();
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(container);
    return () => observer.disconnect();
  }, [draw]);

  const seekFromEvent = useCallback(
    (clientX: number) => {
      const container = containerRef.current;
      if (!container || duration <= 0) return;
      const rect = container.getBoundingClientRect();
      const fraction = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
      onSeek(fraction * duration);
    },
    [duration, onSeek],
  );

  return (
    <div
      ref={containerRef}
      role="slider"
      tabIndex={0}
      aria-label="Playback position"
      aria-valuemin={0}
      aria-valuemax={Math.round(duration)}
      aria-valuenow={Math.round(currentTime)}
      aria-valuetext={`${formatDuration(currentTime)} of ${formatDuration(duration)}`}
      onPointerDown={(event) => {
        draggingRef.current = true;
        event.currentTarget.setPointerCapture(event.pointerId);
        seekFromEvent(event.clientX);
      }}
      onPointerMove={(event) => {
        if (draggingRef.current) seekFromEvent(event.clientX);
      }}
      onPointerUp={(event) => {
        draggingRef.current = false;
        event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onKeyDown={(event) => {
        const jump = event.shiftKey ? 10 : 1;
        if (event.key === "ArrowRight") {
          event.preventDefault();
          onSeek(Math.min(currentTime + jump, duration));
        } else if (event.key === "ArrowLeft") {
          event.preventDefault();
          onSeek(Math.max(currentTime - jump, 0));
        } else if (event.key === "Home") {
          event.preventDefault();
          onSeek(0);
        }
      }}
      className={cn(
        "h-20 w-full cursor-pointer touch-none rounded-lg border border-stroke bg-canvas px-1",
        className,
      )}
    >
      <canvas ref={canvasRef} className="block size-full" />
    </div>
  );
}
