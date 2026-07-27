import type {
  BrowseEntry,
  CheckpointStatus,
  Job,
  ServerConfig,
  SourceAnalysis,
  SpectrogramPayload,
  TrainJob,
  VetResult,
  WaveformPayload,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Non-JSON error bodies keep the status line as their message.
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  config: () => request<ServerConfig>("/api/config"),

  upload: (files: File[], signal?: AbortSignal) => {
    const body = new FormData();
    for (const file of files) body.append("files", file);
    return request<{ files: SourceAnalysis[]; errors: { name: string; error: string }[] }>(
      "/api/uploads",
      { method: "POST", body, signal },
    );
  },

  analyze: (paths: string[]) =>
    request<{ files: SourceAnalysis[]; errors: { name: string; error: string }[] }>(
      "/api/analyze",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths }),
      },
    ),

  browse: (pattern: string) =>
    request<{ entries: BrowseEntry[] }>(`/api/browse?pattern=${encodeURIComponent(pattern)}`),

  listJobs: () =>
    request<{ jobs: Job[]; activeJobId: string | null; queueDepth: number }>("/api/jobs"),

  getJob: (id: string) => request<{ job: Job; log: string[] }>(`/api/jobs/${id}`),

  submitJob: (payload: {
    files: { name: string; sourcePath: string; cutoffHz: number }[];
    steps: number;
    batchSize: number;
  }) =>
    request<Job>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  cancelJob: (id: string) => request<{ ok: boolean }>(`/api/jobs/${id}/cancel`, { method: "POST" }),

  spectrogram: (path: string, signal?: AbortSignal) =>
    request<SpectrogramPayload>(`/api/spectrogram?path=${encodeURIComponent(path)}`, { signal }),

  waveform: (path: string, buckets = 1600, signal?: AbortSignal) =>
    request<WaveformPayload>(
      `/api/waveform?path=${encodeURIComponent(path)}&buckets=${buckets}`,
      { signal },
    ),

  audioUrl: (path: string) => `/api/audio?path=${encodeURIComponent(path)}`,
  downloadUrl: (path: string) => `/api/download?path=${encodeURIComponent(path)}`,

  // Training
  trainingBrowse: (pattern: string) =>
    request<{ entries: BrowseEntry[] }>(
      `/api/training/browse?pattern=${encodeURIComponent(pattern)}`,
    ),

  trainingVet: (paths: string[]) =>
    request<{ files: VetResult[]; errors: { path: string; error: string }[] }>(
      "/api/training/vet",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths }),
      },
    ),

  submitTrainingJob: (payload: {
    dataDir: string;
    steps: number;
    batchSize: number;
    learningRate: number;
    splits: string;
    valFrac: number;
    valEvery: number | null;
    valSamples: number | null;
    restart: boolean;
  }) =>
    request<TrainJob>("/api/training/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  trainingCheckpoints: () => request<CheckpointStatus>("/api/training/checkpoints"),

  activateCheckpoints: () =>
    request<{ activated: number; checkpoints: CheckpointStatus }>("/api/training/activate", {
      method: "POST",
    }),

  revertCheckpoints: () =>
    request<{ checkpoints: CheckpointStatus }>("/api/training/revert", { method: "POST" }),

  trainingMetrics: (splits?: string) =>
    request<Record<string, Array<Record<string, number>>>>(
      `/api/training/metrics${splits ? `?splits=${encodeURIComponent(splits)}` : ""}`,
    ),
};
