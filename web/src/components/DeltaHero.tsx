import { cn, formatDb, formatElapsed, formatHz } from "@/lib/utils";
import type { FileResult } from "@/lib/types";

/** Below this the model effectively added nothing worth keeping. */
const MEANINGFUL_GAIN_DB = 3;
const MARGINAL_GAIN_DB = 1;

function toneFor(delta: number) {
  if (delta >= MEANINGFUL_GAIN_DB) return "gain" as const;
  if (delta >= MARGINAL_GAIN_DB) return "caution" as const;
  return "fault" as const;
}

const TONE_TEXT = {
  gain: "text-gain",
  caution: "text-caution",
  fault: "text-fault",
} as const;

const TONE_VERDICT = {
  gain: "Content added",
  caution: "Marginal gain",
  fault: "Nothing added",
} as const;

/**
 * The high-band delta, presented as the primary result rather than a line of
 * summary text. It is the only number that answers "did this work".
 */
export function DeltaHero({ result }: { result: FileResult }) {
  const tone = toneFor(result.highBandDeltaDb);
  // Both levels are negative dBFS; map onto a shared floor so the two bars are
  // directly comparable rather than each scaled to itself.
  const floor = Math.min(result.highBandInDb, result.highBandOutDb, -90);
  const toPercent = (db: number) => Math.max(0, Math.min(100, ((db - floor) / -floor) * 100));

  return (
    <div className="grid gap-6 sm:grid-cols-[auto_1fr] sm:items-center">
      <div>
        <p className="text-xs font-medium tracking-wide text-ink-faint uppercase">
          Energy above {formatHz(result.cutoffHz)}
        </p>
        <p className={cn("mt-1 font-mono text-5xl leading-none font-bold tnum", TONE_TEXT[tone])}>
          {formatDb(result.highBandDeltaDb, true)}
          <span className="ml-1 text-2xl font-medium">dB</span>
        </p>
        <p className={cn("mt-2 text-sm font-medium", TONE_TEXT[tone])}>{TONE_VERDICT[tone]}</p>
      </div>

      <div className="space-y-3">
        <LevelBar
          label="Filtered input"
          db={result.highBandInDb}
          percent={toPercent(result.highBandInDb)}
          tone="muted"
        />
        <LevelBar
          label="Restored output"
          db={result.highBandOutDb}
          percent={toPercent(result.highBandOutDb)}
          tone={tone}
        />
        <p className="text-xs leading-relaxed text-ink-muted">
          RMS level of everything at or above the cutoff, measured on the rendered files.{" "}
          {result.channels === 2 ? "Stereo" : "Mono"} · {result.steps} steps · batch{" "}
          {result.batchSize} · took {formatElapsed(result.elapsedSec)}.
        </p>
      </div>
    </div>
  );
}

const BAR_TONE = {
  gain: "bg-gain",
  caution: "bg-caution",
  fault: "bg-fault",
  muted: "bg-ink-faint",
} as const;

function LevelBar({
  label,
  db,
  percent,
  tone,
}: {
  label: string;
  db: number;
  percent: number;
  tone: keyof typeof BAR_TONE;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm text-ink-muted">{label}</span>
        <span className="tnum font-mono text-sm text-ink">{formatDb(db)} dB</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-canvas">
        <div
          className={cn("h-full rounded-full transition-[width] duration-500", BAR_TONE[tone])}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
