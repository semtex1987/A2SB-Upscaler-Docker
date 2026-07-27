import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, AudioWaveform, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { ServerConfig, StagedFile } from "@/lib/types";
import { isRestoreJob, isTrainJob } from "@/lib/types";
import { useJobStream } from "@/hooks/useJobStream";
import { Badge } from "@/components/ui/badge";
import { EvaluateView } from "@/components/EvaluateView";
import { RunView } from "@/components/RunView";
import { StageView } from "@/components/StageView";
import { TrainView } from "@/components/TrainView";

type ViewId = "stage" | "run" | "evaluate" | "train";

const VIEWS: { id: ViewId; label: string; caption: string }[] = [
  { id: "stage", label: "Stage", caption: "Choose sources and settings" },
  { id: "run", label: "Run", caption: "Queue, progress and logs" },
  { id: "evaluate", label: "Evaluate", caption: "Compare and download" },
  { id: "train", label: "Train", caption: "Fine-tune and activate checkpoints" },
];

export default function App() {
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [view, setView] = useState<ViewId>("stage");
  const [staged, setStaged] = useState<StagedFile[]>([]);
  const [steps, setSteps] = useState(50);
  const [batchSize, setBatchSize] = useState(16);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [focusJobId, setFocusJobId] = useState<string | null>(null);

  const { jobs, logs, connected, ready, activeJob, queuedCount, hydrateLog } = useJobStream();

  useEffect(() => {
    api
      .config()
      .then((loaded) => {
        setConfig(loaded);
        setSteps(loaded.steps.default);
        setBatchSize(loaded.batchSize.default);
      })
      .catch((error: Error) => setConfigError(error.message));
  }, []);

  const submit = useCallback(async () => {
    const selected = staged.filter((file) => file.selected);
    if (selected.length === 0) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const job = await api.submitJob({
        files: selected.map((file) => ({
          name: file.name,
          sourcePath: file.path,
          cutoffHz: file.cutoffHz,
        })),
        steps,
        batchSize,
      });
      setFocusJobId(job.id);
      setView("run");
    } catch (error) {
      setSubmitError((error as Error).message);
    } finally {
      setSubmitting(false);
    }
  }, [batchSize, staged, steps]);

  const completedCount = useMemo(
    () =>
      jobs
        .filter(isRestoreJob)
        .reduce((total, job) => total + job.files.filter((f) => f.result).length, 0),
    [jobs],
  );

  const activeTrainJob = useMemo(() => jobs.find(isTrainJob), [jobs]);

  const counts: Record<ViewId, number | null> = {
    stage: staged.length || null,
    run:
      jobs.filter((job) => isRestoreJob(job) && (job.status === "running" || job.status === "queued"))
        .length || null,
    evaluate: completedCount || null,
    train:
      jobs.filter((job) => isTrainJob(job) && (job.status === "running" || job.status === "queued"))
        .length || null,
  };

  if (configError) {
    return (
      <main className="mx-auto flex min-h-full max-w-md flex-col justify-center px-6 text-center">
        <h1 className="text-lg font-semibold text-ink">Cannot reach the server</h1>
        <p className="mt-2 text-sm text-ink-muted">{configError}</p>
      </main>
    );
  }

  if (!config || !ready) {
    return (
      <main className="flex min-h-full items-center justify-center gap-2 text-sm text-ink-muted">
        <Loader2 className="size-4 motion-safe:animate-spin" aria-hidden />
        Connecting
      </main>
    );
  }

  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-20 border-b border-stroke bg-canvas/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center gap-2.5">
            <AudioWaveform className="size-6 text-accent" aria-hidden />
            <div>
              <h1 className="text-sm leading-tight font-semibold text-ink">A2SB Restoration</h1>
              <p className="text-xs leading-tight text-ink-muted">Bandwidth extension</p>
            </div>
          </div>

          <nav aria-label="Workflow" className="flex items-center gap-1">
            {VIEWS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setView(item.id)}
                aria-current={view === item.id ? "page" : undefined}
                title={item.caption}
                className={cn(
                  "flex min-h-11 cursor-pointer items-center gap-2 rounded-lg px-3 text-sm font-medium transition-colors duration-150",
                  view === item.id
                    ? "bg-surface text-ink"
                    : "text-ink-muted hover:bg-surface hover:text-ink",
                )}
              >
                {item.label}
                {counts[item.id] ? (
                  <span className="tnum rounded bg-canvas px-1.5 font-mono text-xs text-ink-muted">
                    {counts[item.id]}
                  </span>
                ) : null}
              </button>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            {activeJob ? (
              <Badge tone="accent">
                <Activity className="size-3 motion-safe:animate-pulse" aria-hidden />
                {activeTrainJob?.status === "running" ? "Training" : "GPU busy"}
              </Badge>
            ) : (
              <Badge tone="neutral">GPU idle</Badge>
            )}
            {queuedCount > 0 ? <Badge tone="neutral">{queuedCount} queued</Badge> : null}
            <span
              className={cn("size-2 rounded-full", connected ? "bg-gain" : "bg-caution")}
              title={connected ? "Live updates connected" : "Reconnecting"}
            />
            <span className="sr-only" role="status">
              {connected ? "Live updates connected" : "Live updates reconnecting"}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6 lg:px-8">
        {view === "stage" ? (
          <StageView
            config={config}
            files={staged}
            setFiles={setStaged}
            steps={steps}
            setSteps={setSteps}
            batchSize={batchSize}
            setBatchSize={setBatchSize}
            onSubmit={submit}
            submitting={submitting}
            submitError={submitError}
            queueBusy={activeJob !== null}
          />
        ) : null}

        {view === "run" ? (
          <RunView
            jobs={jobs}
            logs={logs}
            connected={connected}
            onHydrateLog={hydrateLog}
            onShowResults={(jobId) => {
              setFocusJobId(jobId);
              setView("evaluate");
            }}
          />
        ) : null}

        {view === "evaluate" ? <EvaluateView jobs={jobs} focusJobId={focusJobId} /> : null}

        {view === "train" ? (
          <TrainView
            config={config.training}
            jobs={jobs}
            logs={logs}
            onHydrateLog={hydrateLog}
          />
        ) : null}
      </main>
    </div>
  );
}
