import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { cn, formatClock, formatDb, formatHz } from "@/lib/utils";
import type { Job, RestoreJob, SpectrogramPayload } from "@/lib/types";
import { isRestoreJob } from "@/lib/types";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/card";
import { Segmented } from "@/components/ui/segmented";
import { AbTransport } from "@/components/AbTransport";
import { DeltaHero } from "@/components/DeltaHero";
import { Spectrogram, type BandFocus } from "@/components/Spectrogram";

interface CompletedEntry {
  key: string;
  jobId: string;
  finishedAt: number | null;
  result: NonNullable<RestoreJob["files"][number]["result"]>;
}

interface EvaluateViewProps {
  jobs: Job[];
  focusJobId: string | null;
}

export function EvaluateView({ jobs, focusJobId }: EvaluateViewProps) {
  const entries = useMemo<CompletedEntry[]>(
    () =>
      jobs.filter(isRestoreJob).flatMap((job) =>
        job.files
          .filter((file) => file.result !== null)
          .map((file) => ({
            key: `${job.id}:${file.sourcePath}`,
            jobId: job.id,
            finishedAt: file.finishedAt,
            result: file.result!,
          })),
      ),
    [jobs],
  );

  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  useEffect(() => {
    if (entries.length === 0) {
      setSelectedKey(null);
      return;
    }
    const stillPresent = entries.some((entry) => entry.key === selectedKey);
    if (stillPresent) return;
    const preferred = focusJobId
      ? (entries.find((entry) => entry.jobId === focusJobId) ?? entries[0])
      : entries[0];
    setSelectedKey(preferred.key);
  }, [entries, focusJobId, selectedKey]);

  const selected = entries.find((entry) => entry.key === selectedKey) ?? null;

  if (entries.length === 0) {
    return (
      <Panel>
        <PanelBody className="py-16 text-center">
          <p className="text-sm text-ink">Nothing to evaluate yet.</p>
          <p className="mt-1 text-sm text-ink-muted">
            Finished restorations appear here with an A/B player and a spectrogram.
          </p>
        </PanelBody>
      </Panel>
    );
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px] xl:items-start">
      {selected ? <ResultDetail key={selected.key} entry={selected} /> : null}
      <Panel className="xl:sticky xl:top-6">
        <PanelHeader title="History" description="Every completed run, newest first." />
        <PanelBody className="scrollbar-slim max-h-[70vh] space-y-2 overflow-auto">
          {entries.map((entry) => {
            const active = entry.key === selectedKey;
            const delta = entry.result.highBandDeltaDb;
            return (
              <button
                key={entry.key}
                type="button"
                onClick={() => setSelectedKey(entry.key)}
                aria-current={active}
                className={cn(
                  "w-full cursor-pointer rounded-lg border px-3 py-2.5 text-left transition-colors duration-150",
                  active
                    ? "border-accent bg-accent/10"
                    : "border-stroke bg-canvas hover:border-stroke-strong hover:bg-surface-raised",
                )}
              >
                <p className="truncate text-sm font-medium text-ink">{entry.result.name}</p>
                <p className="mt-0.5 font-mono text-[11px] text-ink-muted tnum">
                  {formatHz(entry.result.cutoffHz)} · {entry.result.steps} steps ·{" "}
                  <span
                    className={cn(
                      delta >= 3 ? "text-gain" : delta >= 1 ? "text-caution" : "text-fault",
                    )}
                  >
                    {formatDb(delta, true)} dB
                  </span>
                </p>
                <p className="mt-0.5 text-[11px] text-ink-faint">
                  {formatClock(entry.finishedAt)}
                </p>
              </button>
            );
          })}
        </PanelBody>
      </Panel>
    </div>
  );
}

function ResultDetail({ entry }: { entry: CompletedEntry }) {
  const { result } = entry;
  const [restored, setRestored] = useState<SpectrogramPayload | null>(null);
  const [filtered, setFiltered] = useState<SpectrogramPayload | null>(null);
  const [bandFocus, setBandFocus] = useState<BandFocus>("high");
  const [spectroError, setSpectroError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setRestored(null);
    setFiltered(null);
    setSpectroError(null);
    Promise.all([
      api.spectrogram(result.restoredPath, controller.signal),
      api.spectrogram(result.filteredPath, controller.signal),
    ])
      .then(([restoredPayload, filteredPayload]) => {
        setRestored(restoredPayload);
        setFiltered(filteredPayload);
      })
      .catch((error: Error) => {
        if (error.name !== "AbortError") setSpectroError(error.message);
      });
    return () => controller.abort();
  }, [result.filteredPath, result.restoredPath]);

  return (
    <div className="space-y-6">
      <Panel>
        <PanelHeader
          title={result.name}
          description={`Restored at ${formatHz(result.cutoffHz)} · run ${entry.jobId}`}
        />
        <PanelBody className="space-y-6">
          <DeltaHero result={result} />
          {result.warnings.map((warning) => (
            <p
              key={warning}
              className="flex items-start gap-2 rounded-md border border-caution/40 bg-caution/5 px-3 py-2 text-sm text-caution"
            >
              <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
              {warning}
            </p>
          ))}
        </PanelBody>
      </Panel>

      <Panel>
        <PanelHeader title="Listen" description="Input against output at the same playhead." />
        <PanelBody>
          <AbTransport result={result} />
        </PanelBody>
      </Panel>

      <Panel>
        <PanelHeader
          title="Spectrum"
          description="Drag the handle under the plot to wipe between the filtered input and the restored output. Hover anywhere for a time, frequency and level readout."
          actions={
            <Segmented<BandFocus>
              ariaLabel="Frequency range shown"
              size="sm"
              value={bandFocus}
              onChange={setBandFocus}
              options={[
                { value: "high", label: "8 kHz and up", hint: "Zoom on the reconstructed band" },
                { value: "full", label: "Full range" },
              ]}
            />
          }
        />
        <PanelBody>
          {spectroError ? (
            <p className="py-10 text-center text-sm text-fault">{spectroError}</p>
          ) : restored ? (
            <Spectrogram
              primary={restored}
              comparison={filtered}
              primaryLabel="Restored"
              comparisonLabel="Filtered input"
              cutoffHz={result.cutoffHz}
              bandFocus={bandFocus}
            />
          ) : (
            <div
              className="flex h-72 items-center justify-center gap-2 text-sm text-ink-muted"
              // Space is reserved so the page does not jump when the plot arrives.
            >
              <Loader2 className="size-4 motion-safe:animate-spin" aria-hidden />
              Computing spectrogram
            </div>
          )}
        </PanelBody>
      </Panel>
    </div>
  );
}
