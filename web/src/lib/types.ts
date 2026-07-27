export type JobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";

/** Result of the pre-flight spectral scan on a staged source file. */
export interface SourceAnalysis {
  path: string;
  name: string;
  sizeBytes: number;
  durationSec: number;
  sampleRate: number;
  channels: number;
  /** Highest frequency carrying real energy, from the 95th-percentile spectrum. */
  hfEdgeHz: number;
  /** True when the spectrum cliffs at hfEdgeHz, the signature of a lossy transcode. */
  shelf: boolean;
  verdict: "transcode" | "clean-fade" | "full-bandwidth" | "unknown";
  note: string;
  suggestedCutoffHz: number;
}

export interface FileResult {
  name: string;
  sourcePath: string;
  restoredPath: string;
  filteredPath: string;
  channels: number;
  durationSec: number;
  cutoffHz: number;
  steps: number;
  batchSize: number;
  highBandInDb: number;
  highBandOutDb: number;
  highBandDeltaDb: number;
  elapsedSec: number;
  warnings: string[];
}

export interface JobFile {
  name: string;
  sourcePath: string;
  cutoffHz: number;
  status: JobStatus;
  stage: string;
  /** Null while the diffusion process has not reported a step yet. */
  fraction: number | null;
  etaSec: number | null;
  startedAt: number | null;
  finishedAt: number | null;
  result: FileResult | null;
  error: string | null;
  errorDetail: string | null;
}

/** Training parameters mirroring server/jobs.py TrainParams. */
export interface TrainParams {
  dataDir: string;
  outputDir: string;
  steps: number;
  batchSize: number;
  learningRate: number;
  splits: string;
  valFrac: number;
  valEvery: number | null;
  valSamples: number | null;
  restart: boolean;
}

/** A restoration job — runs inference per file. */
export interface RestoreJob {
  kind: "restore";
  id: string;
  createdAt: number;
  steps: number;
  batchSize: number;
  files: JobFile[];
  status: JobStatus;
  startedAt: number | null;
  finishedAt: number | null;
  error: string | null;
  progress: number | null;
}

/** A training job — runs finetune.py for hours on the same GPU. */
export interface TrainJob {
  kind: "train";
  id: string;
  createdAt: number;
  steps: number;
  batchSize: number;
  files: [];
  status: JobStatus;
  startedAt: number | null;
  finishedAt: number | null;
  error: string | null;
  progress: number | null;
  trainParams: TrainParams | null;
  trainStage: string;
  trainFraction: number | null;
  trainEtaSec: number | null;
}

export type Job = RestoreJob | TrainJob;

/** Narrow to restoration jobs for RunView / EvaluateView. */
export function isRestoreJob(job: Job): job is RestoreJob {
  return job.kind === "restore";
}

/** Narrow to training jobs for TrainView. */
export function isTrainJob(job: Job): job is TrainJob {
  return job.kind === "train";
}

/** Type-safe switch over all job kinds — compile error if a new kind is unhandled. */
export function matchJob<T>(
  job: Job,
  handlers: { restore: (j: RestoreJob) => T; train: (j: TrainJob) => T },
): T {
  switch (job.kind) {
    case "restore":
      return handlers.restore(job);
    case "train":
      return handlers.train(job);
    default: {
      const _never: never = job;
      throw new Error(`Unhandled job kind: ${String((_never as Job).kind)}`);
    }
  }
}

export interface TrainingConfig {
  steps: { min: number; max: number; default: number };
  batchSize: { min: number; max: number; default: number };
  learningRateDefault: number;
  dataDir: string;
  outputDir: string;
  splits: string[];
}

export interface ServerConfig {
  steps: { min: number; max: number; default: number };
  batchSize: { min: number; max: number; default: number };
  cutoffHz: { min: number; max: number; default: number };
  inputDir: string;
  outputDir: string;
  audioExtensions: string[];
  training: TrainingConfig;
}

export interface SpectrogramPayload {
  width: number;
  height: number;
  sampleRate: number;
  maxFrequencyHz: number;
  durationSec: number;
  floorDb: number;
  /** base64 uint8 grid, row 0 = highest frequency. */
  data: string;
}

export interface WaveformPayload {
  peaks: number[];
  durationSec: number;
}

export interface BrowseEntry {
  path: string;
  name: string;
  sizeBytes: number;
}

/** A staged file plus the per-file cutoff the operator settled on. */
export interface StagedFile extends SourceAnalysis {
  id: string;
  cutoffHz: number;
  selected: boolean;
}

/** Dataset vetting result returned by /api/training/vet. */
export interface VetResult {
  path: string;
  name: string;
  sizeBytes: number;
  durationSec: number;
  sampleRate: number;
  hfEdgeHz: number;
  shelf: boolean;
  verdict: "pass" | "check" | "reject";
  note: string;
}

/** Checkpoint status returned by /api/training/checkpoints. */
export interface CheckpointStatus {
  active: "release" | "finetuned" | "mixed";
  finetunedPaths: Record<string, string>;
  releasePaths: Record<string, string>;
  ensembleConfig: string;
}

export type StreamEvent =
  | { type: "snapshot"; jobs: Job[]; activeJobId: string | null }
  | { type: "job"; job: Job }
  | { type: "log"; jobId: string; line: string };
