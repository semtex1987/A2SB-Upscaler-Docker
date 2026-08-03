import { useCallback, useMemo, useRef, useState } from "react";
import { AlertTriangle, FolderSearch, Loader2, Trash2, Upload, Waves } from "lucide-react";
import { api } from "@/lib/api";
import { cn, formatBytes, formatDuration, formatHz } from "@/lib/utils";
import type { ServerConfig, SourceAnalysis, StagedFile } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";

interface StageViewProps {
  config: ServerConfig;
  files: StagedFile[];
  setFiles: React.Dispatch<React.SetStateAction<StagedFile[]>>;
  steps: number;
  setSteps: (value: number) => void;
  batchSize: number;
  setBatchSize: (value: number) => void;
  onSubmit: () => Promise<void>;
  submitting: boolean;
  submitError: string | null;
  queueBusy: boolean;
}

function toStaged(analysis: SourceAnalysis): StagedFile {
  return {
    ...analysis,
    id: analysis.path,
    cutoffHz: analysis.suggestedCutoffHz,
    selected: true,
  };
}

export function StageView({
  config,
  files,
  setFiles,
  steps,
  setSteps,
  batchSize,
  setBatchSize,
  onSubmit,
  submitting,
  submitError,
  queueBusy,
}: StageViewProps) {
  const [dragActive, setDragActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState<{ name: string; error: string }[]>([]);
  const [pattern, setPattern] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const merge = useCallback(
    (analyses: SourceAnalysis[]) => {
      setFiles((previous) => {
        const byId = new Map(previous.map((file) => [file.id, file]));
        for (const analysis of analyses) {
          if (!byId.has(analysis.path)) byId.set(analysis.path, toStaged(analysis));
        }
        return Array.from(byId.values());
      });
    },
    [setFiles],
  );

  const addUploads = useCallback(
    async (list: FileList | null) => {
      if (!list || list.length === 0) return;
      setBusy(true);
      setErrors([]);
      try {
        const response = await api.upload(Array.from(list));
        merge(response.files);
        setErrors(response.errors);
      } catch (error) {
        setErrors([{ name: "Upload", error: (error as Error).message }]);
      } finally {
        setBusy(false);
      }
    },
    [merge],
  );

  const addStagedPaths = useCallback(async () => {
    if (!pattern.trim()) return;
    setBusy(true);
    setErrors([]);
    try {
      const { entries } = await api.browse(pattern);
      if (entries.length === 0) {
        setErrors([{ name: pattern, error: "No audio files matched." }]);
        return;
      }
      const response = await api.analyze(entries.map((entry) => entry.path));
      merge(response.files);
      setErrors(response.errors);
    } catch (error) {
      setErrors([{ name: "Browse", error: (error as Error).message }]);
    } finally {
      setBusy(false);
    }
  }, [merge, pattern]);

  const selected = files.filter((file) => file.selected);

  // Two directories can hold the same filename, and identical-looking rows with
  // different cutoffs are impossible to tell apart. Show the path only then.
  const ambiguousNames = useMemo(() => {
    const counts = new Map<string, number>();
    for (const file of files) counts.set(file.name, (counts.get(file.name) ?? 0) + 1);
    return new Set([...counts].filter(([, count]) => count > 1).map(([name]) => name));
  }, [files]);

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px] xl:items-start">
      <div className="space-y-6">
        <Panel>
          <PanelHeader
            title="Sources"
            description="Every file is scanned on arrival, so the cutoff below starts from what the audio actually contains rather than a guess."
            actions={
              files.length > 0 ? (
                <Button variant="ghost" onClick={() => setFiles([])}>
                  <Trash2 className="size-4" aria-hidden />
                  Clear
                </Button>
              ) : null
            }
          />
          <PanelBody className="space-y-4">
            <div
              onDragOver={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragActive(false);
                void addUploads(event.dataTransfer.files);
              }}
              className={cn(
                "rounded-lg border-2 border-dashed p-8 text-center transition-colors duration-150",
                dragActive ? "border-accent bg-accent/5" : "border-stroke bg-canvas",
              )}
            >
              <Waves className="mx-auto size-8 text-ink-faint" aria-hidden />
              <p className="mt-3 text-sm text-ink">Drop audio files here</p>
              <p className="mt-1 text-xs text-ink-muted">
                {config.audioExtensions.join(" · ").replaceAll(".", "")}
              </p>
              <Button className="mt-4" onClick={() => inputRef.current?.click()} disabled={busy}>
                {busy ? (
                  <Loader2 className="size-4 motion-safe:animate-spin" aria-hidden />
                ) : (
                  <Upload className="size-4" aria-hidden />
                )}
                Choose files
              </Button>
              <input
                ref={inputRef}
                type="file"
                multiple
                accept="audio/*"
                className="sr-only"
                onChange={(event) => {
                  void addUploads(event.target.files);
                  event.target.value = "";
                }}
              />
            </div>

            <details className="group rounded-lg border border-stroke bg-canvas">
              <summary className="flex min-h-11 cursor-pointer list-none items-center gap-2 px-4 text-sm text-ink-muted hover:text-ink">
                <FolderSearch className="size-4" aria-hidden />
                Use files already staged on the pod
              </summary>
              <div className="space-y-3 border-t border-stroke px-4 py-3">
                <label htmlFor="staged-pattern" className="block text-xs text-ink-muted">
                  One path or glob per line, under {config.inputDir} or {config.outputDir}.
                </label>
                <textarea
                  id="staged-pattern"
                  rows={2}
                  value={pattern}
                  onChange={(event) => setPattern(event.target.value)}
                  placeholder={`${config.inputDir}/*.wav`}
                  className="w-full rounded-md border border-stroke bg-surface px-3 py-2 font-mono text-sm text-ink placeholder:text-ink-faint"
                />
                <Button onClick={() => void addStagedPaths()} disabled={busy || !pattern.trim()}>
                  Scan paths
                </Button>
              </div>
            </details>

            {errors.length > 0 ? (
              <ul className="space-y-1 rounded-lg border border-fault/40 bg-fault/5 px-4 py-3">
                {errors.map((entry) => (
                  <li key={entry.name} className="text-sm text-fault">
                    <span className="font-medium">{entry.name}</span>: {entry.error}
                  </li>
                ))}
              </ul>
            ) : null}

            {files.length === 0 ? (
              <p className="py-6 text-center text-sm text-ink-muted">
                Nothing staged yet. Restored audio and spectrograms are written to{" "}
                <code className="font-mono text-ink">{config.outputDir}</code>.
              </p>
            ) : (
              <ul className="space-y-3">
                {files.map((file) => (
                  <StagedFileRow
                    key={file.id}
                    file={file}
                    showPath={ambiguousNames.has(file.name)}
                    cutoffBounds={config.cutoffHz}
                    onChange={(next) =>
                      setFiles((previous) =>
                        previous.map((item) => (item.id === file.id ? next : item)),
                      )
                    }
                    onRemove={() =>
                      setFiles((previous) => previous.filter((item) => item.id !== file.id))
                    }
                  />
                ))}
              </ul>
            )}
          </PanelBody>
        </Panel>
      </div>

      <Panel className="xl:sticky xl:top-6">
        <PanelHeader title="Run settings" description="Applied to every selected file." />
        <PanelBody className="space-y-6">
          <Slider
            label="Diffusion steps"
            value={steps}
            min={config.steps.min}
            max={config.steps.max}
            step={10}
            onChange={setSteps}
            hint="More steps means more sampling time per channel. 50 to 100 is the practical range for bandwidth extension; beyond that the return is hard to hear."
          />
          <Slider
            label="Inference batch size"
            value={batchSize}
            min={config.batchSize.min}
            max={config.batchSize.max}
            onChange={setBatchSize}
            hint="Higher is faster but uses more VRAM. 16 to 32 suits an H100 or H200; drop it if inference dies with a CUDA out-of-memory error."
          />

          <div className="space-y-2 border-t border-stroke pt-4">
            <div className="flex items-baseline justify-between text-sm">
              <span className="text-ink-muted">Selected</span>
              <span className="tnum font-mono text-ink">
                {selected.length} of {files.length}
              </span>
            </div>
            <div className="flex items-baseline justify-between text-sm">
              <span className="text-ink-muted">Channels to process</span>
              <span className="tnum font-mono text-ink">
                {selected.reduce((total, file) => total + file.channels, 0)}
              </span>
            </div>
            <p className="pt-1 text-xs leading-relaxed text-ink-muted">
              Stereo files are split and run one channel at a time, so a stereo file costs two
              diffusion passes.
            </p>
          </div>

          {submitError ? (
            <p className="rounded-md border border-fault/40 bg-fault/5 px-3 py-2 text-sm text-fault">
              {submitError}
            </p>
          ) : null}

          <Button
            variant="primary"
            size="lg"
            className="w-full"
            disabled={selected.length === 0 || submitting}
            onClick={() => void onSubmit()}
          >
            {submitting ? (
              <Loader2 className="size-4 motion-safe:animate-spin" aria-hidden />
            ) : null}
            {queueBusy ? "Add to queue" : "Start restoration"}
          </Button>
          {queueBusy ? (
            <p className="text-center text-xs text-ink-muted">
              Another job is running. This one starts when the GPU frees up.
            </p>
          ) : null}
        </PanelBody>
      </Panel>
    </div>
  );
}

const VERDICT_COPY = {
  transcode: { tone: "caution" as const, label: "Lossy transcode" },
  "clean-fade": { tone: "gain" as const, label: "Genuine master" },
  "full-bandwidth": { tone: "neutral" as const, label: "Full bandwidth" },
  unknown: { tone: "fault" as const, label: "Unreadable" },
};

function StagedFileRow({
  file,
  showPath,
  cutoffBounds,
  onChange,
  onRemove,
}: {
  file: StagedFile;
  showPath: boolean;
  cutoffBounds: { min: number; max: number };
  onChange: (next: StagedFile) => void;
  onRemove: () => void;
}) {
  const verdict = VERDICT_COPY[file.verdict] ?? VERDICT_COPY.unknown;
  const overridden = file.cutoffHz !== file.suggestedCutoffHz;

  return (
    <li className="relative rounded-lg border border-stroke bg-canvas">
      <div className="flex items-start gap-3 px-4 py-3">
        <input
          id={`select-${file.id}`}
          type="checkbox"
          checked={file.selected}
          onChange={(event) => onChange({ ...file, selected: event.target.checked })}
          aria-label={`Include ${file.name}`}
          className="mt-1 size-4 shrink-0 cursor-pointer accent-[var(--color-accent)] relative z-10"
        />
        <label htmlFor={`select-${file.id}`} className="after:absolute after:inset-0 cursor-pointer text-[0px]">Select {file.name}</label>
        <div className="min-w-0 flex-1 pointer-events-none">
          <div className="flex flex-wrap items-center gap-2">
            <p className="truncate font-medium text-ink">{file.name}</p>
            <Badge tone={verdict.tone}>{verdict.label}</Badge>
            {file.shelf ? (
              <Badge tone="caution">
                <AlertTriangle className="size-3" aria-hidden />
                Brickwall
              </Badge>
            ) : null}
          </div>
          {showPath ? (
            <p className="mt-0.5 truncate font-mono text-xs text-ink-faint" title={file.path}>
              {file.path}
            </p>
          ) : null}
          <p className="mt-1 font-mono text-xs text-ink-muted tnum">
            {formatDuration(file.durationSec)} · {(file.sampleRate / 1000).toFixed(1)} kHz ·{" "}
            {file.channels === 2 ? "stereo" : "mono"} · {formatBytes(file.sizeBytes)} · content to{" "}
            {formatHz(file.hfEdgeHz)}
          </p>
          <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">{file.note}</p>
        </div>
        <Button variant="ghost" size="icon" aria-label={`Remove ${file.name}`} onClick={onRemove} className="relative z-10">
          <Trash2 className="size-4" aria-hidden />
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-3 border-t border-stroke px-4 py-3">
        <label
          htmlFor={`cutoff-${file.id}`}
          className="text-xs font-medium tracking-wide text-ink-faint uppercase"
        >
          Cutoff
        </label>
        <input
          id={`cutoff-${file.id}`}
          type="number"
          min={cutoffBounds.min}
          max={cutoffBounds.max}
          step={100}
          value={file.cutoffHz}
          onChange={(event) => onChange({ ...file, cutoffHz: Number(event.target.value) })}
          className="tnum w-28 rounded-md border border-stroke bg-surface px-2 py-1.5 font-mono text-sm text-ink"
        />
        <span className="text-xs text-ink-muted">Hz</span>
        {overridden ? (
          <button
            type="button"
            onClick={() => onChange({ ...file, cutoffHz: file.suggestedCutoffHz })}
            className="cursor-pointer text-xs text-accent underline-offset-2 hover:underline"
          >
            Reset to suggested {formatHz(file.suggestedCutoffHz)}
          </button>
        ) : (
          <span className="text-xs text-ink-faint">Measured suggestion</span>
        )}
      </div>
    </li>
  );
}
