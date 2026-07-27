import { useEffect, useRef, useState } from "react";
import { ChevronDown, CircleSlash, Terminal } from "lucide-react";
import { api } from "@/lib/api";
import { cn, formatClock, formatDb, formatElapsed, formatHz } from "@/lib/utils";
import type { Job, JobFile, RestoreJob } from "@/lib/types";
import { isRestoreJob } from "@/lib/types";
import { Badge, StatusBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface RunViewProps {
  jobs: Job[];
  logs: Record<string, string[]>;
  connected: boolean;
  onHydrateLog: (jobId: string) => void;
  onShowResults: (jobId: string) => void;
}

export function RunView({ jobs, logs, connected, onHydrateLog, onShowResults }: RunViewProps) {
  // RunView shows only restoration jobs; training jobs are tracked in TrainView.
  const restoreJobs = jobs.filter(isRestoreJob);
  if (restoreJobs.length === 0) {
    return (
      <Panel>
        <PanelBody className="py-16 text-center">
          <p className="text-sm text-ink">No jobs yet.</p>
          <p className="mt-1 text-sm text-ink-muted">
            Stage some audio and start a restoration. Jobs keep running if you close this tab.
          </p>
        </PanelBody>
      </Panel>
    );
  }

  return (
    <div className="space-y-4">
      {!connected ? (
        <p className="rounded-lg border border-caution/40 bg-caution/5 px-4 py-3 text-sm text-caution">
          Live updates disconnected. Any running job continues on the server; this page will
          reattach automatically.
        </p>
      ) : null}
      {restoreJobs.map((job) => (
        <JobCard
          key={job.id}
          job={job}
          log={logs[job.id]}
          onHydrateLog={onHydrateLog}
          onShowResults={onShowResults}
        />
      ))}
    </div>
  );
}

function JobCard({
  job,
  log,
  onHydrateLog,
  onShowResults,
}: {
  job: RestoreJob;
  log?: string[];
  onHydrateLog: (jobId: string) => void;
  onShowResults: (jobId: string) => void;
}) {
  const [showLog, setShowLog] = useState(job.status === "running");
  const [cancelling, setCancelling] = useState(false);
  const active = job.status === "running" || job.status === "queued";
  const completedCount = job.files.filter((file) => file.status === "completed").length;

  useEffect(() => {
    if (showLog) onHydrateLog(job.id);
  }, [job.id, onHydrateLog, showLog]);

  return (
    <Panel>
      <PanelHeader
        title={`${job.files.length} file${job.files.length === 1 ? "" : "s"} · ${job.steps} steps · batch ${job.batchSize}`}
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
            {completedCount > 0 ? (
              <Button variant="secondary" onClick={() => onShowResults(job.id)}>
                View results
              </Button>
            ) : null}
          </>
        }
      />
      <PanelBody className="space-y-3">
        {job.files.map((file) => (
          <FileProgressRow key={`${job.id}-${file.sourcePath}`} file={file} />
        ))}

        <div className="border-t border-stroke pt-3">
          <button
            type="button"
            onClick={() => setShowLog((current) => !current)}
            aria-expanded={showLog}
            className="flex min-h-11 cursor-pointer items-center gap-2 text-sm text-ink-muted hover:text-ink"
          >
            <Terminal className="size-4" aria-hidden />
            Inference log
            <ChevronDown
              className={cn("size-4 transition-transform duration-150", showLog && "rotate-180")}
              aria-hidden
            />
          </button>
          {showLog ? <LogPanel lines={log ?? []} live={job.status === "running"} /> : null}
        </div>
      </PanelBody>
    </Panel>
  );
}

function FileProgressRow({ file }: { file: JobFile }) {
  const running = file.status === "running";
  const percent = file.fraction === null ? null : Math.round(file.fraction * 100);

  return (
    <div className="rounded-lg border border-stroke bg-canvas px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <p className="min-w-0 flex-1 truncate font-medium text-ink">{file.name}</p>
        <Badge tone="neutral">{formatHz(file.cutoffHz)}</Badge>
        <StatusBadge status={file.status} />
      </div>

      {running || file.status === "queued" ? (
        <div className="mt-3 space-y-1.5">
          <Progress value={file.fraction} label={`${file.name} progress`} />
          <div className="flex items-center justify-between gap-3 text-xs">
            <span className="text-ink-muted">{file.stage}</span>
            <span className="tnum font-mono text-ink-muted">
              {percent === null ? "starting" : `${percent}%`}
              {file.etaSec != null ? ` · ${formatElapsed(file.etaSec)} left` : ""}
            </span>
          </div>
        </div>
      ) : null}

      {file.result ? (
        <p className="mt-2 font-mono text-xs text-ink-muted tnum">
          {formatDb(file.result.highBandInDb)} dB → {formatDb(file.result.highBandOutDb)} dB (
          <span
            className={cn(
              file.result.highBandDeltaDb >= 3
                ? "text-gain"
                : file.result.highBandDeltaDb >= 1
                  ? "text-caution"
                  : "text-fault",
            )}
          >
            {formatDb(file.result.highBandDeltaDb, true)} dB
          </span>
          ) in {formatElapsed(file.result.elapsedSec)}
        </p>
      ) : null}

      {file.error ? (
        <div className="mt-2 rounded-md border border-fault/40 bg-fault/5 px-3 py-2">
          <p className="text-sm text-fault">{file.error}</p>
          {file.errorDetail ? (
            <pre className="scrollbar-slim mt-2 max-h-40 overflow-auto font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-ink-muted">
              {file.errorDetail}
            </pre>
          ) : null}
        </div>
      ) : null}
    </div>
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
        // Stop auto-scrolling the moment the reader scrolls up to look at something.
        pinnedRef.current =
          element.scrollHeight - element.scrollTop - element.clientHeight < 24;
      }}
      className="scrollbar-slim mt-3 max-h-72 overflow-auto rounded-lg border border-stroke bg-canvas p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-ink-muted"
      aria-live={live ? "polite" : "off"}
      aria-label="Inference log output"
    >
      {lines.length > 0 ? lines.join("\n") : "No output recorded."}
    </pre>
  );
}
