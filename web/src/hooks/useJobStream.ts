import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { Job, StreamEvent } from "@/lib/types";

const LOG_LIMIT = 400;

/**
 * Subscribes to the server's job stream.
 *
 * The connection is the source of truth for live state: the server sends a
 * snapshot on connect, so a tab that was closed mid-job reattaches to the same
 * state on reopen rather than losing it.
 */
export function useJobStream() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [logs, setLogs] = useState<Record<string, string[]>>({});
  const [connected, setConnected] = useState(false);
  const [ready, setReady] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const source = new EventSource("/api/events");
    sourceRef.current = source;

    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as StreamEvent;
      if (event.type === "snapshot") {
        setJobs(event.jobs);
        setReady(true);
        setConnected(true);
        return;
      }
      if (event.type === "job") {
        setJobs((previous) => {
          const index = previous.findIndex((job) => job.id === event.job.id);
          if (index === -1) return [event.job, ...previous];
          const next = previous.slice();
          next[index] = event.job;
          return next;
        });
        return;
      }
      setLogs((previous) => {
        const existing = previous[event.jobId] ?? [];
        const appended = [...existing, event.line];
        return {
          ...previous,
          [event.jobId]: appended.length > LOG_LIMIT ? appended.slice(-LOG_LIMIT) : appended,
        };
      });
    };

    return () => {
      source.close();
      sourceRef.current = null;
    };
  }, []);

  /** Pull the persisted log for a job the stream did not witness live. */
  const hydrateLog = useCallback(
    async (jobId: string) => {
      if (logs[jobId]) return;
      try {
        const { log } = await api.getJob(jobId);
        setLogs((previous) => ({ ...previous, [jobId]: log }));
      } catch {
        // A missing log is not worth surfacing; the job row already shows status.
      }
    },
    [logs],
  );

  const activeJob = useMemo(
    () => jobs.find((job) => job.status === "running") ?? null,
    [jobs],
  );

  const queuedCount = useMemo(
    () => jobs.filter((job) => job.status === "queued").length,
    [jobs],
  );

  return { jobs, logs, connected, ready, activeJob, queuedCount, hydrateLog };
}
