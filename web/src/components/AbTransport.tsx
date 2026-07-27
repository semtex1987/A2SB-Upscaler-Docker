import { useEffect, useState } from "react";
import { Download, Pause, Play, Rewind, FastForward } from "lucide-react";
import { api } from "@/lib/api";
import { formatDuration, formatHz } from "@/lib/utils";
import { useAbPlayer, type AbSource } from "@/hooks/useAbPlayer";
import type { FileResult } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Segmented } from "@/components/ui/segmented";
import { Waveform } from "@/components/Waveform";

interface AbTransportProps {
  result: FileResult;
}

/**
 * The comparison surface: one playhead, one transport, and a switch between the
 * filtered input and the restored output that does not move your position in
 * the file.
 */
export function AbTransport({ result }: AbTransportProps) {
  const player = useAbPlayer({
    filteredUrl: api.audioUrl(result.filteredPath),
    restoredUrl: api.audioUrl(result.restoredPath),
    cutoffHz: result.cutoffHz,
  });
  const [peaks, setPeaks] = useState<number[]>([]);

  useEffect(() => {
    const controller = new AbortController();
    setPeaks([]);
    api
      .waveform(result.restoredPath, 1600, controller.signal)
      .then((payload) => setPeaks(payload.peaks))
      .catch(() => undefined);
    return () => controller.abort();
  }, [result.restoredPath]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (event.code === "Space") {
        event.preventDefault();
        player.toggle();
      } else if (event.key.toLowerCase() === "a") {
        player.setSource((current: AbSource) =>
          current === "filtered" ? "restored" : "filtered",
        );
      } else if (event.key.toLowerCase() === "s") {
        player.setSoloHighBand((current: boolean) => !current);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [player]);

  return (
    <div className="space-y-4">
      <Waveform
        peaks={peaks}
        duration={player.duration || result.durationSec}
        currentTime={player.currentTime}
        onSeek={player.seek}
      />

      <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            aria-label="Back 5 seconds"
            onClick={() => player.skip(-5)}
          >
            <Rewind className="size-5" aria-hidden />
          </Button>
          <Button
            variant="primary"
            size="icon"
            aria-label={player.playing ? "Pause" : "Play"}
            disabled={!player.ready}
            onClick={player.toggle}
          >
            {player.playing ? (
              <Pause className="size-5" aria-hidden />
            ) : (
              <Play className="size-5" aria-hidden />
            )}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Forward 5 seconds"
            onClick={() => player.skip(5)}
          >
            <FastForward className="size-5" aria-hidden />
          </Button>
        </div>

        <span className="tnum font-mono text-sm text-ink-muted">
          {formatDuration(player.currentTime)}
          <span className="mx-1 text-ink-faint">/</span>
          {formatDuration(player.duration || result.durationSec)}
        </span>

        <div className="flex items-center gap-2">
          <span className="text-xs font-medium tracking-wide text-ink-faint uppercase">
            Listening to
          </span>
          <Segmented<AbSource>
            ariaLabel="Choose which version you hear"
            value={player.source}
            onChange={player.setSource}
            options={[
              { value: "filtered", label: "Input", hint: "The lowpassed source (A)" },
              { value: "restored", label: "Restored", hint: "The A2SB output (B)" },
            ]}
          />
        </div>

        <Button
          variant={player.soloHighBand ? "primary" : "secondary"}
          aria-pressed={player.soloHighBand}
          onClick={() => player.setSoloHighBand(!player.soloHighBand)}
          title="Highpass playback at the cutoff so you hear only the reconstructed band"
        >
          Solo above {formatHz(result.cutoffHz)}
        </Button>

        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="ghost"
            onClick={() => window.open(api.downloadUrl(result.restoredPath), "_blank")}
          >
            <Download className="size-4" aria-hidden />
            Restored WAV
          </Button>
        </div>
      </div>

      <p className="text-xs leading-relaxed text-ink-muted">
        Both versions play in lockstep, so switching never moves the playhead.{" "}
        <kbd className="rounded border border-stroke bg-canvas px-1 font-mono">Space</kbd> plays,{" "}
        <kbd className="rounded border border-stroke bg-canvas px-1 font-mono">A</kbd> switches
        source, <kbd className="rounded border border-stroke bg-canvas px-1 font-mono">S</kbd>{" "}
        solos the reconstructed band. Solo is the honest test: if that band is silent or gritty,
        the model did not give you anything worth keeping.
      </p>

      {player.error ? <p className="text-sm text-fault">{player.error}</p> : null}
    </div>
  );
}
