import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  CircleSlash,
  ExternalLink,
  FolderSearch,
  Loader2,
  Play,
  RefreshCw,
  Terminal,
  XCircle,
} from "lucide-react";
import { api, formatApiError } from "@/lib/api";
import { cn, formatBytes, formatClock, formatDuration, formatElapsed, formatHz } from "@/lib/utils";
import type {
  BrowseEntry,
  CheckpointStatus,
  Job,
  TensorBoardStatus,
  TrainingMetrics,
  TrainJob,
  VetResult,
} from "@/lib/types";
import { isTrainJob } from "@/lib/types";
import type { TrainingConfig } from "@/lib/types";
import { LossCurve } from "@/components/LossCurve";
import { Badge, StatusBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Segmented } from "@/components/ui/segmented";
import { Slider } from "@/components/ui/slider";

interface TrainViewProps {
  config: TrainingConfig;
  jobs: Job[];
  logs: Record<string, string[]>;
  onHydrateLog: (jobId: string) => void;
}

export function TrainView({ config, jobs, logs, onHydrateLog }: TrainViewProps) {
  const trainJobs = jobs.filter(isTrainJob);
  const activeJob = trainJobs.find((j) => j.status === "running" || j.status === "queued") ?? null;
  const resumableJobs = trainJobs.filter((j) => j.status !== "running" && j.status !== "queued");
  const metricsJobId = activeJob?.id ?? trainJobs[0]?.id;

  // Dataset panel state
  const [pattern, setPattern] = useState(config.dataDir);
  const [browsing, setBrowsing] = useState(false);
  const [vetting, setVetting] = useState(false);
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [vetResults, setVetResults] = useState<VetResult[]>([]);
  const [browseError, setBrowseError] = useState<string | null>(null);

  // Configure panel state
  const [steps, setSteps] = useState(config.steps.default);
  const [batchSize, setBatchSize] = useState(config.batchSize.default);
  const [learningRate, setLearningRate] = useState(config.learningRateDefault);
  const [splits, setSplits] = useState("both");
  const [valFrac, setValFrac] = useState(0.1);
  const [valEvery, setValEvery] = useState<number | null>(null);
  const [valSamples, setValSamples] = useState<number | null>(null);
  const [restart, setRestart] = useState(false);
  const [resumeFromJobId, setResumeFromJobId] = useState("");
  const vetGeneration = useRef(0);
  const vettedPattern = useRef<string | null>(null);

  // Submit state
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Metrics + TensorBoard state
  const [metrics, setMetrics] = useState<TrainingMetrics>({});
  const [tbStatus, setTbStatus] = useState<TensorBoardStatus | null>(null);
  const [tbStarting, setTbStarting] = useState(false);
  const [tbError, setTbError] = useState<string | null>(null);
  const [tbEmbedded, setTbEmbedded] = useState(false);

  useEffect(() => {
    vetGeneration.current += 1;
    setVetResults([]);
    setEntries([]);
    vettedPattern.current = null;
  }, [pattern]);

  // Checkpoint state
  const [checkpoints, setCheckpoints] = useState<CheckpointStatus | null>(null);
  const [checkpointBusy, setCheckpointBusy] = useState(false);
  const [checkpointMsg, setCheckpointMsg] = useState<string | null>(null);

  const loadCheckpoints = useCallback(async () => {
    try {
      const status = await api.trainingCheckpoints();
      setCheckpoints(status);
    } catch {
      // Checkpoints endpoint may fail if ensemble YAML is missing on a fresh setup.
    }
  }, []);

  useEffect(() => {
    void loadCheckpoints();
  }, [loadCheckpoints]);

  const loadMetrics = useCallback(async () => {
    try {
      setMetrics(await api.trainingMetrics(undefined, metricsJobId));
    } catch {
      // Absent until Lightning writes its first metrics.csv; the panel hides itself.
    }
  }, [metricsJobId]);

  // Depends on the boolean rather than on activeJob, because that object is
  // rebuilt on every SSE tick and would restart the interval before it fired.
  const isTraining = activeJob !== null;
  useEffect(() => {
    void loadMetrics();
    if (!isTraining) return;
    const timer = setInterval(() => void loadMetrics(), 10_000);
    return () => clearInterval(timer);
  }, [loadMetrics, isTraining]);

  const refreshTensorboard = useCallback(async () => {
    try {
      const status = await api.tensorboardStatus();
      setTbStatus(status);
      if (status.ready) setTbEmbedded(true);
      return status;
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    void refreshTensorboard();
  }, [refreshTensorboard]);

  const startTensorboard = useCallback(async () => {
    setTbStarting(true);
    setTbError(null);
    try {
      setTbStatus(await api.startTensorboard());
      // The start call returns as soon as the process exists; TensorBoard needs
      // another 15-30s to scan the log tree before the iframe will load.
      const deadline = Date.now() + 90_000;
      while (Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        const status = await refreshTensorboard();
        if (status?.ready) return;
        if (status && !status.running) {
          setTbError(status.error ?? "TensorBoard stopped while starting up.");
          return;
        }
      }
      setTbError("TensorBoard did not finish starting up. Check the server logs.");
    } catch (err) {
      setTbError((err as Error).message);
    } finally {
      setTbStarting(false);
    }
  }, [refreshTensorboard]);

  // Splits are absent until their run starts, so an empty one is not shown.
  const splitsWithMetrics = Object.entries(metrics).filter(([, rows]) => rows.length > 0);

  const stopTensorboard = useCallback(async () => {
    setTbEmbedded(false);
    setTbError(null);
    try {
      setTbStatus(await api.stopTensorboard());
    } catch (err) {
      setTbError((err as Error).message);
    }
  }, []);

  const browseAndVet = useCallback(async () => {
    if (!pattern.trim()) return;
    const generation = ++vetGeneration.current;
    const requested = pattern.trim();
    setBrowsing(true);
    setBrowseError(null);
    setVetResults([]);
    vettedPattern.current = null;
    try {
      const { entries: found } = await api.trainingBrowse(pattern);
      if (generation !== vetGeneration.current) return;
      setEntries(found);
      if (found.length === 0) {
        setBrowseError("No audio files found. Check the path and try again.");
        return;
      }
      setVetting(true);
      const { files } = await api.trainingVet(found.map((e) => e.path));
      if (generation !== vetGeneration.current) return;
      setVetResults(files);
      vettedPattern.current = requested;
    } catch (err) {
      if (generation !== vetGeneration.current) return;
      setBrowseError(formatApiError(err));
    } finally {
      if (generation === vetGeneration.current) {
        setBrowsing(false);
        setVetting(false);
      }
    }
  }, [pattern]);

  const passCount = vetResults.filter((r) => r.verdict === "pass").length;
  const checkCount = vetResults.filter((r) => r.verdict === "check").length;
  const rejectCount = vetResults.filter((r) => r.verdict === "reject").length;
  const totalDuration = vetResults.reduce((sum, r) => sum + r.durationSec, 0);
  const passedDuration = vetResults
    .filter((r) => r.verdict !== "reject")
    .reduce((sum, r) => sum + r.durationSec, 0);

  const canSubmit =
    !activeJob &&
    vettedPattern.current === pattern.trim() &&
    vetResults.some((r) => r.verdict !== "reject") &&
    !submitting;

  const submit = useCallback(async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      await api.submitTrainingJob({
        dataDir: pattern.trim(),
        steps,
        batchSize,
        learningRate,
        splits,
        valFrac,
        valEvery,
        valSamples,
        restart: Boolean(resumeFromJobId) && restart,
        resumeFromJobId: resumeFromJobId || null,
        untilStep: null,
      });
    } catch (err) {
      setSubmitError(formatApiError(err));
    } finally {
      setSubmitting(false);
    }
  }, [canSubmit, batchSize, learningRate, pattern, restart, resumeFromJobId, splits, steps, valEvery, valFrac, valSamples]);

  return (
    <div className="space-y-6">
      {/* Queue warning when already training */}
      {activeJob ? (
        <div className="rounded-lg border border-caution/40 bg-caution/5 px-4 py-3 text-sm text-caution">
          <strong>GPU is busy training.</strong> Restoration jobs will queue behind the current
          fine-tune. Cancellation is available in the progress panel below.
        </div>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px] xl:items-start">
        {/* Left column */}
        <div className="space-y-6">
          {/* Dataset panel */}
          <Panel>
            <PanelHeader
              title="Dataset"
              description="Scan a directory and vet each file before spending GPU time."
            />
            <PanelBody className="space-y-4">
              <div>
                <label htmlFor="training-pattern" className="block text-xs text-ink-muted">
                  Directory path on this machine
                </label>
                <div className="mt-1.5 flex gap-2">
                  <textarea
                    id="training-pattern"
                    rows={1}
                    value={pattern}
                    onChange={(e) => setPattern(e.target.value)}
                    placeholder={config.dataDir}
                    className="flex-1 resize-none rounded-md border border-stroke bg-surface px-3 py-2 font-mono text-sm text-ink placeholder:text-ink-faint"
                  />
                  <Button
                    onClick={() => void browseAndVet()}
                    disabled={browsing || vetting || !pattern.trim()}
                  >
                    {browsing || vetting ? (
                      <Loader2 className="size-4 motion-safe:animate-spin" aria-hidden />
                    ) : (
                      <FolderSearch className="size-4" aria-hidden />
                    )}
                    {vetting ? "Vetting…" : browsing ? "Scanning…" : "Scan & vet"}
                  </Button>
                </div>
                <p className="mt-1 text-xs text-ink-faint">
                  Bandwidth verdicts are spectral heuristics (first two minutes), not proof of
                  provenance. Duration comes from the file header. Changing the path clears a
                  previous scan — re-run Scan &amp; vet before submitting. Only PASS and CHECK
                  files are usable training material.
                </p>
              </div>

              {browseError ? (
                <p className="text-sm text-fault">{browseError}</p>
              ) : null}

              {vetResults.length > 0 ? (
                <>
                  <div className="flex flex-wrap gap-2 rounded-lg border border-stroke bg-canvas px-3 py-2 text-sm">
                    <span className="text-gain font-medium">{passCount} PASS</span>
                    <span className="text-ink-muted">·</span>
                    <span className="text-caution font-medium">{checkCount} CHECK</span>
                    <span className="text-ink-muted">·</span>
                    <span className="text-fault font-medium">{rejectCount} REJECT</span>
                    <span className="ml-auto text-ink-muted tnum font-mono text-xs">
                      {formatDuration(passedDuration)} usable / {formatDuration(totalDuration)} total
                    </span>
                  </div>

                  {rejectCount > 0 ? (
                    <div className="flex gap-2 rounded-lg border border-fault/40 bg-fault/5 px-3 py-2 text-xs text-fault">
                      <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
                      <span>
                        {rejectCount} file{rejectCount !== 1 ? "s" : ""} rejected. Training on
                        band-limited files teaches the model to output silence above the cutoff.
                        Remove them from the directory, or finetune.py will silently discard them.
                      </span>
                    </div>
                  ) : null}

                  <ul className="space-y-2">
                    {vetResults.map((r) => (
                      <VetResultRow key={r.path} result={r} />
                    ))}
                  </ul>
                </>
              ) : entries.length > 0 && !browsing && !vetting ? (
                <p className="text-sm text-ink-muted">{entries.length} files found — vetting…</p>
              ) : null}
            </PanelBody>
          </Panel>

          {/* Training progress panel — only shown when jobs exist */}
          {trainJobs.length > 0 ? (
            <div className="space-y-4">
              {trainJobs.map((job) => (
                <TrainJobCard
                  key={job.id}
                  job={job}
                  log={logs[job.id]}
                  onHydrateLog={onHydrateLog}
                  onFinished={loadCheckpoints}
                />
              ))}
            </div>
          ) : null}

          {/* Loss curves, read from Lightning's metrics.csv */}
          {splitsWithMetrics.length > 0 ? (
            <Panel>
              <PanelHeader
                title="Loss curves"
                description={
                  isTraining
                    ? "Updating every 10 seconds while training runs."
                    : "From the most recent run in each split."
                }
                actions={
                  <Button variant="ghost" onClick={() => void loadMetrics()} title="Refresh">
                    <RefreshCw className="size-4" aria-hidden />
                  </Button>
                }
              />
              <PanelBody className="space-y-3">
                {splitsWithMetrics.map(([tag, rows]) => (
                  <LossCurve key={tag} label={tag} rows={rows} />
                ))}
              </PanelBody>
            </Panel>
          ) : null}

          {/* TensorBoard */}
          <Panel>
            <PanelHeader
              title="TensorBoard"
              description="Scalars, histograms and distributions for the full training history."
              actions={
                <>
                  {tbStatus?.running ? (
                    <Badge tone={tbStatus.ready ? "gain" : "neutral"}>
                      {tbStatus.ready ? "Running" : "Starting"}
                    </Badge>
                  ) : null}
                  {tbStatus?.ready ? (
                    <>
                      <a
                        href={tbStatus.url}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex min-h-9 items-center gap-1.5 rounded-lg px-3 text-sm text-ink-muted transition-colors duration-150 hover:bg-surface-raised hover:text-ink"
                      >
                        <ExternalLink className="size-4" aria-hidden />
                        New tab
                      </a>
                      <Button variant="secondary" onClick={() => void stopTensorboard()}>
                        Stop
                      </Button>
                    </>
                  ) : null}
                </>
              }
            />
            <PanelBody className="space-y-3">
              {tbStatus && !tbStatus.available ? (
                <p className="text-sm text-ink-muted">
                  {tbStatus.error ??
                    "TensorBoard is not installed in this image. Rebuild to enable it."}
                </p>
              ) : (
                <>
                  {!tbStatus?.running ? (
                    <>
                      <p className="text-sm text-ink-muted">
                        {tbStatus?.hasEventFiles
                          ? "Event files are ready to load."
                          : "No event files yet — start a fine-tune, then launch TensorBoard to watch it."}
                      </p>
                      <Button disabled={tbStarting} onClick={() => void startTensorboard()}>
                        {tbStarting ? (
                          <Loader2 className="size-4 motion-safe:animate-spin" aria-hidden />
                        ) : (
                          <Play className="size-4" aria-hidden />
                        )}
                        {tbStarting ? "Starting TensorBoard…" : "Launch TensorBoard"}
                      </Button>
                    </>
                  ) : null}

                  {tbStatus?.running && !tbStatus.ready ? (
                    <p className="flex items-center gap-2 text-sm text-ink-muted">
                      <Loader2 className="size-4 motion-safe:animate-spin" aria-hidden />
                      Scanning the log directory. This usually takes 15–30 seconds.
                    </p>
                  ) : null}

                  {tbError ? (
                    <div className="rounded-lg border border-fault/40 bg-fault/5 px-3 py-2">
                      <p className="font-mono text-xs whitespace-pre-wrap text-fault">{tbError}</p>
                    </div>
                  ) : null}

                  {tbEmbedded && tbStatus?.ready ? (
                    <iframe
                      src={tbStatus.url}
                      title="TensorBoard"
                      className="h-[640px] w-full rounded-lg border border-stroke bg-surface"
                    />
                  ) : null}
                </>
              )}
            </PanelBody>
          </Panel>
        </div>

        {/* Right column */}
        <div className="space-y-6">
          {/* Configure panel */}
          <Panel>
            <PanelHeader
              title="Configure"
              description="Applied to the next training run."
            />
            <PanelBody className="space-y-5">
              <Slider
                label="Additional steps per split"
                hint="Optimizer updates from the starting checkpoint (0 for a new run, or the resumed global step). Use Resume below to continue a previous job instead of sharing one output folder."
                min={config.steps.min}
                max={config.steps.max}
                value={steps}
                onChange={setSteps}
              />

              <Slider
                label="Batch size"
                hint="Larger batches are faster but use more VRAM. 2 is safe on a 16 GB GPU."
                min={config.batchSize.min}
                max={config.batchSize.max}
                value={batchSize}
                onChange={setBatchSize}
              />

              <div>
                <label htmlFor="train-lr" className="mb-1.5 block text-sm font-medium text-ink">
                  Learning rate
                </label>
                <input
                  id="train-lr"
                  type="number"
                  min={1e-7}
                  max={1e-3}
                  step={1e-6}
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value) || config.learningRateDefault)}
                  className="w-full rounded-md border border-stroke bg-surface px-3 py-2 font-mono text-sm text-ink"
                />
                <p className="mt-1 text-xs text-ink-faint">
                  Default {config.learningRateDefault}. Keep below 1e-4; too high will overfit in minutes.
                </p>
              </div>

              <div>
                <p className="mb-2 text-sm font-medium text-ink">Splits to train</p>
                <Segmented
                  options={config.splits.map((s) => ({
                    value: s,
                    label: s === "both" ? "Both" : s,
                  }))}
                  value={splits}
                  onChange={setSplits}
                  ariaLabel="Splits to train"
                />
                <p className="mt-1 text-xs text-ink-faint">
                  Both: full fine-tune across the diffusion trajectory. Train one split to test
                  faster.
                </p>
              </div>

              <details className="group rounded-lg border border-stroke bg-canvas">
                <summary className="flex min-h-10 cursor-pointer list-none items-center gap-2 px-3 text-sm text-ink-muted hover:text-ink">
                  <ChevronDown
                    className="size-4 transition-transform duration-150 group-open:rotate-180"
                    aria-hidden
                  />
                  Advanced options
                </summary>
                <div className="space-y-4 border-t border-stroke px-3 py-3">
                  <div>
                    <label htmlFor="val-frac" className="mb-1 block text-xs text-ink-muted">
                      Validation fraction ({Math.round(valFrac * 100)}%)
                    </label>
                    <input
                      id="val-frac"
                      type="number"
                      min={0.01}
                      max={0.5}
                      step={0.01}
                      value={valFrac}
                      onChange={(e) => setValFrac(parseFloat(e.target.value) || 0.1)}
                      className="w-full rounded-md border border-stroke bg-surface px-3 py-2 font-mono text-sm text-ink"
                    />
                  </div>

                  <div>
                    <label htmlFor="val-every" className="mb-1 block text-xs text-ink-muted">
                      Validate every N steps (blank = auto)
                    </label>
                    <input
                      id="val-every"
                      type="number"
                      min={1}
                      value={valEvery ?? ""}
                      onChange={(e) =>
                        setValEvery(e.target.value ? parseInt(e.target.value, 10) : null)
                      }
                      className="w-full rounded-md border border-stroke bg-surface px-3 py-2 font-mono text-sm text-ink"
                    />
                  </div>

                  <div>
                    <label htmlFor="val-samples" className="mb-1 block text-xs text-ink-muted">
                      Max validation samples (blank = all, 16–32 recommended)
                    </label>
                    <input
                      id="val-samples"
                      type="number"
                      min={1}
                      value={valSamples ?? ""}
                      onChange={(e) =>
                        setValSamples(e.target.value ? parseInt(e.target.value, 10) : null)
                      }
                      className="w-full rounded-md border border-stroke bg-surface px-3 py-2 font-mono text-sm text-ink"
                    />
                    <p className="mt-1 text-xs text-ink-faint">
                      Each sample runs a full diffusion pass. 256 (the default) can take an hour
                      per validation cycle; 16–32 gives the same signal in minutes.
                    </p>
                  </div>

                  <label className="flex cursor-pointer items-center gap-3">
                    <input
                      type="checkbox"
                      checked={restart}
                      onChange={(e) => setRestart(e.target.checked)}
                      disabled={!resumeFromJobId}
                      className="size-4 rounded border-stroke accent-accent"
                    />
                    <span className="text-sm text-ink">
                      Restart this run — ignore its checkpoints and start from release weights
                    </span>
                  </label>
                </div>
              </details>

              <div>
                <label htmlFor="resume-from" className="mb-1.5 block text-sm font-medium text-ink">
                  Resume from
                </label>
                <select
                  id="resume-from"
                  value={resumeFromJobId}
                  onChange={(e) => {
                    setResumeFromJobId(e.target.value);
                    if (!e.target.value) setRestart(false);
                  }}
                  className="w-full rounded-md border border-stroke bg-surface px-3 py-2 font-mono text-sm text-ink"
                >
                  <option value="">New run (fresh output directory)</option>
                  {resumableJobs.map((job) => (
                    <option key={job.id} value={job.id}>
                      {job.id} · {job.trainParams?.splits ?? "both"} · {job.status}
                    </option>
                  ))}
                </select>
                <p className="mt-1 text-xs text-ink-faint">
                  Each new job writes to its own folder under the training output directory.
                  Resume continues that job's checkpoints; additional steps are added to its
                  current global step. Leaving this on a new run never picks up an earlier job.
                </p>
              </div>

              {submitError ? (
                <div className="rounded-lg border border-fault/40 bg-fault/5 px-3 py-2">
                  <p className="text-sm font-medium text-fault">Cannot start training</p>
                  <pre className="mt-1 whitespace-pre-wrap font-mono text-xs text-fault">
                    {submitError}
                  </pre>
                </div>
              ) : null}

              {!activeJob && vetResults.length > 0 ? (
                <div className="rounded-lg border border-caution/30 bg-caution/5 px-3 py-2 text-xs text-caution">
                  Training takes hours. Restoration jobs will queue behind it. The GPU badge
                  in the header will read "Training" so you know not to wait.
                </div>
              ) : null}

              <Button
                className="w-full"
                disabled={!canSubmit || submitting}
                onClick={() => void submit()}
              >
                {submitting ? (
                  <Loader2 className="size-4 motion-safe:animate-spin" aria-hidden />
                ) : null}
                {activeJob
                  ? "Training already running"
                  : vetResults.length === 0
                    ? "Scan a dataset first"
                    : "Start fine-tuning"}
              </Button>
            </PanelBody>
          </Panel>

          {/* Checkpoints panel */}
          <Panel>
            <PanelHeader
              title="Checkpoints"
              description="Activate fine-tuned weights without restarting the container."
              actions={
                <Button variant="ghost" onClick={() => void loadCheckpoints()} title="Refresh">
                  <RefreshCw className="size-4" aria-hidden />
                </Button>
              }
            />
            <PanelBody className="space-y-4">
              {checkpoints ? (
                <>
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-ink-muted">Active:</span>
                    <Badge
                      tone={
                        checkpoints.active === "finetuned"
                          ? "gain"
                          : checkpoints.active === "mixed"
                            ? "caution"
                            : "neutral"
                      }
                    >
                      {checkpoints.active === "finetuned"
                        ? "Fine-tuned"
                        : checkpoints.active === "mixed"
                          ? "Mixed (one fine-tuned, one release)"
                          : "Release"}
                    </Badge>
                  </div>

                  <div className="space-y-1.5 rounded-lg border border-stroke bg-canvas px-3 py-2 font-mono text-xs text-ink-muted">
                    <p className="font-semibold text-ink">Fine-tuned</p>
                    {Object.keys(checkpoints.finetunedPaths).length > 0 ? (
                      Object.entries(checkpoints.finetunedPaths).map(([name, path]) => (
                        <p key={name} className="truncate" title={path}>
                          <CheckCircle2 className="mr-1 inline size-3 text-gain" aria-hidden />
                          {name}
                        </p>
                      ))
                    ) : (
                      <p className="italic text-ink-faint">None yet — run a fine-tune first.</p>
                    )}
                  </div>

                  {checkpointMsg ? (
                    <p className="text-sm text-gain">{checkpointMsg}</p>
                  ) : null}

                  <div className="flex gap-2">
                    <Button
                      className="flex-1"
                      disabled={
                        checkpointBusy ||
                        Object.keys(checkpoints.finetunedPaths).length === 0 ||
                        checkpoints.active === "finetuned"
                      }
                      onClick={async () => {
                        setCheckpointBusy(true);
                        setCheckpointMsg(null);
                        try {
                          const res = await api.activateCheckpoints();
                          setCheckpoints(res.checkpoints);
                          setCheckpointMsg(
                            `Activated ${res.activated} fine-tuned checkpoint${res.activated !== 1 ? "s" : ""}. Next restoration uses them.`,
                          );
                        } catch (err) {
                          setCheckpointMsg(`Error: ${(err as Error).message}`);
                        } finally {
                          setCheckpointBusy(false);
                        }
                      }}
                    >
                      Activate fine-tuned
                    </Button>
                    <Button
                      variant="secondary"
                      disabled={checkpointBusy || checkpoints.active === "release"}
                      onClick={async () => {
                        setCheckpointBusy(true);
                        setCheckpointMsg(null);
                        try {
                          const res = await api.revertCheckpoints();
                          setCheckpoints(res.checkpoints);
                          setCheckpointMsg("Reverted to release checkpoints.");
                        } catch (err) {
                          setCheckpointMsg(`Error: ${(err as Error).message}`);
                        } finally {
                          setCheckpointBusy(false);
                        }
                      }}
                    >
                      Revert to release
                    </Button>
                  </div>
                </>
              ) : (
                <p className="text-sm text-ink-muted">Loading checkpoint status…</p>
              )}
            </PanelBody>
          </Panel>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function VetResultRow({ result }: { result: VetResult }) {
  const toneClass =
    result.verdict === "pass"
      ? "text-gain"
      : result.verdict === "check"
        ? "text-caution"
        : "text-fault";

  const Icon =
    result.verdict === "pass"
      ? CheckCircle2
      : result.verdict === "check"
        ? AlertTriangle
        : XCircle;

  return (
    <li className="rounded-lg border border-stroke bg-canvas px-3 py-2.5">
      <div className="flex items-start gap-2">
        <Icon className={cn("mt-0.5 size-4 shrink-0", toneClass)} aria-hidden />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
            <p className="min-w-0 truncate font-medium text-ink">{result.name}</p>
            <span className={cn("font-mono text-xs font-medium uppercase", toneClass)}>
              {result.verdict}
            </span>
            <span className="ml-auto font-mono text-xs text-ink-muted tnum">
              {formatHz(result.hfEdgeHz)} · {formatDuration(result.durationSec)} ·{" "}
              {formatBytes(result.sizeBytes)}
            </span>
          </div>
          <p className="mt-0.5 text-xs text-ink-muted">{result.note}</p>
        </div>
      </div>
    </li>
  );
}

function TrainJobCard({
  job,
  log,
  onHydrateLog,
  onFinished,
}: {
  job: TrainJob;
  log?: string[];
  onHydrateLog: (jobId: string) => void;
  onFinished: () => void;
}) {
  const [showLog, setShowLog] = useState(job.status === "running");
  const [cancelling, setCancelling] = useState(false);
  const active = job.status === "running" || job.status === "queued";
  const wasFinished = useRef(false);

  useEffect(() => {
    if (showLog) onHydrateLog(job.id);
  }, [job.id, onHydrateLog, showLog]);

  useEffect(() => {
    if (!wasFinished.current && job.status === "completed") {
      wasFinished.current = true;
      onFinished();
    }
  }, [job.status, onFinished]);

  const percent = job.trainFraction === null ? null : Math.round(job.trainFraction * 100);
  const params = job.trainParams;

  return (
    <Panel>
      <PanelHeader
        title={
          params
            ? `${params.steps.toLocaleString()} steps · batch ${params.batchSize} · splits ${params.splits}`
            : `Training job · ${job.steps.toLocaleString()} steps`
        }
        description={
          <span className="font-mono text-xs tnum">
            {job.id} · started {formatClock(job.startedAt ?? job.createdAt)}
            {job.finishedAt && job.startedAt
              ? ` · took ${formatElapsed(job.finishedAt - job.startedAt)}`
              : ""}
          </span>
        }
        actions={
          <>
            <Badge tone="neutral">Training</Badge>
            <StatusBadge status={job.status} />
            {active ? (
              <Button
                variant="danger"
                disabled={cancelling}
                onClick={async () => {
                  setCancelling(true);
                  try {
                    await api.cancelJob(job.id);
                  } finally {
                    setCancelling(false);
                  }
                }}
              >
                <CircleSlash className="size-4" aria-hidden />
                Cancel
              </Button>
            ) : null}
          </>
        }
      />
      <PanelBody className="space-y-3">
        <div className="rounded-lg border border-stroke bg-canvas px-4 py-3">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
            <p className="flex-1 text-ink-muted">{job.trainStage || "Queued"}</p>
            {percent !== null ? (
              <span className="tnum font-mono text-xs text-ink-muted">
                {percent}%
                {job.trainEtaSec != null ? ` · ${formatElapsed(job.trainEtaSec)} left` : ""}
              </span>
            ) : null}
          </div>
          {(active || percent !== null) ? (
            <div className="mt-2">
              <Progress value={job.trainFraction} label="Training progress" />
            </div>
          ) : null}
        </div>

        {job.error ? (
          <div className="rounded-lg border border-fault/40 bg-fault/5 px-3 py-2">
            <p className="text-sm text-fault">{job.error}</p>
          </div>
        ) : null}

        <div className="border-t border-stroke pt-2">
          <button
            type="button"
            onClick={() => setShowLog((v) => !v)}
            aria-expanded={showLog}
            className="flex min-h-11 cursor-pointer items-center gap-2 text-sm text-ink-muted hover:text-ink"
          >
            <Terminal className="size-4" aria-hidden />
            Training log
            <ChevronDown
              className={cn("size-4 transition-transform duration-150", showLog && "rotate-180")}
              aria-hidden
            />
          </button>
          {showLog ? (
            <LogPanel lines={log ?? []} live={job.status === "running"} />
          ) : null}
        </div>
      </PanelBody>
    </Panel>
  );
}

function LogPanel({ lines, live }: { lines: string[]; live: boolean }) {
  const ref = useRef<HTMLPreElement>(null);
  const pinnedRef = useRef(true);

  useEffect(() => {
    const element = ref.current;
    if (!element || !pinnedRef.current) return;
    element.scrollTop = element.scrollHeight;
  }, [lines]);

  return (
    <pre
      ref={ref}
      onScroll={(event) => {
        const element = event.currentTarget;
        pinnedRef.current =
          element.scrollHeight - element.scrollTop - element.clientHeight < 24;
      }}
      className="scrollbar-slim mt-3 max-h-72 overflow-auto rounded-lg border border-stroke bg-canvas p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-ink-muted"
      aria-live={live ? "polite" : "off"}
      aria-label="Training log output"
    >
      {lines.length > 0 ? lines.join("\n") : "No output yet."}
    </pre>
  );
}
