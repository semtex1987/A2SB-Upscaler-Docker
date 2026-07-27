import type { SpectrogramPayload } from "./types";

/**
 * Inferno control points, sampled from the matplotlib colormap. The previous
 * server-rendered plots used inferno, so keeping it means a spectrogram read in
 * this UI looks like one read anywhere else in the project.
 */
const INFERNO_STOPS: [number, number, number][] = [
  [0, 0, 4],
  [22, 11, 57],
  [66, 10, 104],
  [106, 23, 110],
  [147, 38, 103],
  [188, 55, 84],
  [221, 81, 58],
  [243, 120, 25],
  [252, 165, 10],
  [246, 215, 70],
  [252, 255, 164],
];

function buildLut(): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3);
  const segments = INFERNO_STOPS.length - 1;
  for (let i = 0; i < 256; i += 1) {
    const position = (i / 255) * segments;
    const index = Math.min(Math.floor(position), segments - 1);
    const t = position - index;
    const from = INFERNO_STOPS[index];
    const to = INFERNO_STOPS[index + 1];
    lut[i * 3] = from[0] + (to[0] - from[0]) * t;
    lut[i * 3 + 1] = from[1] + (to[1] - from[1]) * t;
    lut[i * 3 + 2] = from[2] + (to[2] - from[2]) * t;
  }
  return lut;
}

const LUT = buildLut();

export function decodeGrid(payload: SpectrogramPayload): Uint8Array {
  const binary = atob(payload.data);
  const grid = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) grid[i] = binary.charCodeAt(i);
  return grid;
}

/** Paint the grid into an offscreen canvas at its native resolution. */
export function renderToCanvas(payload: SpectrogramPayload, grid: Uint8Array): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = payload.width;
  canvas.height = payload.height;
  const context = canvas.getContext("2d");
  if (!context) return canvas;

  const image = context.createImageData(payload.width, payload.height);
  for (let i = 0; i < grid.length; i += 1) {
    const value = grid[i];
    image.data[i * 4] = LUT[value * 3];
    image.data[i * 4 + 1] = LUT[value * 3 + 1];
    image.data[i * 4 + 2] = LUT[value * 3 + 2];
    image.data[i * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  return canvas;
}

/** Row 0 of the grid is the highest frequency, so the axis runs top-down. */
export function frequencyToRow(hz: number, payload: SpectrogramPayload): number {
  return (1 - hz / payload.maxFrequencyHz) * payload.height;
}

/** Convert a stored uint8 back to the dB value it was quantised from. */
export function valueToDb(value: number, floorDb: number): number {
  return floorDb + (value / 255) * -floorDb;
}

export function sampleDb(
  payload: SpectrogramPayload,
  grid: Uint8Array,
  fractionX: number,
  fractionY: number,
): number {
  const column = Math.min(payload.width - 1, Math.max(0, Math.floor(fractionX * payload.width)));
  const row = Math.min(payload.height - 1, Math.max(0, Math.floor(fractionY * payload.height)));
  return valueToDb(grid[row * payload.width + column], payload.floorDb);
}
