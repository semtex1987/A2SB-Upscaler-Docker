import { useCallback, useEffect, useRef, useState } from "react";

export type AbSource = "filtered" | "restored";

interface Options {
  filteredUrl: string;
  restoredUrl: string;
  /** Corner frequency for the high-band solo, in Hz. */
  cutoffHz: number;
}

/** Transparent corner for the solo chain when solo is off. */
const BYPASS_HZ = 20;
/** Three cascaded biquads give ~36 dB/oct, steep enough to isolate the band. */
const SOLO_STAGES = 3;
const SWITCH_RAMP_SEC = 0.015;
/** Resync the muted element if it drifts further than this from the audible one. */
const MAX_DRIFT_SEC = 0.06;

/**
 * Plays the filtered input and the restored output in lockstep and switches
 * which one you hear without moving the playhead.
 *
 * Both elements run at once and one is muted, so an A/B switch is a gain change
 * rather than a seek. That is the whole point: a seek would break the comparison
 * by putting you somewhere else in the file at the moment you switch.
 */
export function useAbPlayer({ filteredUrl, restoredUrl, cutoffHz }: Options) {
  const [source, setSource] = useState<AbSource>("restored");
  const [soloHighBand, setSoloHighBand] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const filteredRef = useRef<HTMLAudioElement | null>(null);
  const restoredRef = useRef<HTMLAudioElement | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const gainsRef = useRef<{ filtered: GainNode; restored: GainNode } | null>(null);
  const filtersRef = useRef<BiquadFilterNode[]>([]);
  const frameRef = useRef<number | null>(null);
  const sourceRef = useRef<AbSource>(source);
  sourceRef.current = source;

  // -- element setup -------------------------------------------------------

  useEffect(() => {
    const filtered = new Audio();
    const restored = new Audio();
    for (const [element, url] of [
      [filtered, filteredUrl],
      [restored, restoredUrl],
    ] as const) {
      element.src = url;
      element.preload = "auto";
      element.crossOrigin = "anonymous";
    }
    filteredRef.current = filtered;
    restoredRef.current = restored;

    let loaded = 0;
    const onLoaded = () => {
      loaded += 1;
      if (loaded === 2) {
        setDuration(Math.max(filtered.duration || 0, restored.duration || 0));
        setReady(true);
      }
    };
    const onError = () => setError("Could not load the rendered audio.");

    filtered.addEventListener("loadedmetadata", onLoaded);
    restored.addEventListener("loadedmetadata", onLoaded);
    filtered.addEventListener("error", onError);
    restored.addEventListener("error", onError);

    return () => {
      filtered.removeEventListener("loadedmetadata", onLoaded);
      restored.removeEventListener("loadedmetadata", onLoaded);
      filtered.removeEventListener("error", onError);
      restored.removeEventListener("error", onError);
      filtered.pause();
      restored.pause();
      filtered.src = "";
      restored.src = "";
      filteredRef.current = null;
      restoredRef.current = null;
      setReady(false);
      setPlaying(false);
      setCurrentTime(0);
    };
  }, [filteredUrl, restoredUrl]);

  // -- audio graph, built on the first gesture (autoplay policy) -----------

  const ensureGraph = useCallback(() => {
    if (contextRef.current || !filteredRef.current || !restoredRef.current) return;
    const context = new AudioContext();
    const filteredGain = context.createGain();
    const restoredGain = context.createGain();
    filteredGain.gain.value = sourceRef.current === "filtered" ? 1 : 0;
    restoredGain.gain.value = sourceRef.current === "restored" ? 1 : 0;

    context.createMediaElementSource(filteredRef.current).connect(filteredGain);
    context.createMediaElementSource(restoredRef.current).connect(restoredGain);

    const mixer = context.createGain();
    filteredGain.connect(mixer);
    restoredGain.connect(mixer);

    // Always in circuit; a 20 Hz corner is inaudible, so toggling solo is a
    // frequency change rather than a reconnect that could click.
    const filters: BiquadFilterNode[] = [];
    let tail: AudioNode = mixer;
    for (let stage = 0; stage < SOLO_STAGES; stage += 1) {
      const filter = context.createBiquadFilter();
      filter.type = "highpass";
      filter.frequency.value = BYPASS_HZ;
      filter.Q.value = Math.SQRT1_2;
      tail.connect(filter);
      tail = filter;
      filters.push(filter);
    }
    tail.connect(context.destination);

    contextRef.current = context;
    gainsRef.current = { filtered: filteredGain, restored: restoredGain };
    filtersRef.current = filters;
  }, []);

  useEffect(() => {
    return () => {
      void contextRef.current?.close();
      contextRef.current = null;
      gainsRef.current = null;
      filtersRef.current = [];
    };
  }, []);

  // -- A/B switching -------------------------------------------------------

  useEffect(() => {
    const gains = gainsRef.current;
    const context = contextRef.current;
    if (!gains || !context) return;
    const now = context.currentTime;
    for (const key of ["filtered", "restored"] as const) {
      const node = gains[key].gain;
      node.cancelScheduledValues(now);
      node.setValueAtTime(node.value, now);
      node.linearRampToValueAtTime(source === key ? 1 : 0, now + SWITCH_RAMP_SEC);
    }
  }, [source]);

  useEffect(() => {
    const target = soloHighBand ? Math.max(cutoffHz, 100) : BYPASS_HZ;
    const context = contextRef.current;
    for (const filter of filtersRef.current) {
      if (context) {
        filter.frequency.setTargetAtTime(target, context.currentTime, 0.01);
      } else {
        filter.frequency.value = target;
      }
    }
  }, [soloHighBand, cutoffHz]);

  // -- transport -----------------------------------------------------------

  const tick = useCallback(() => {
    const audible = sourceRef.current === "filtered" ? filteredRef.current : restoredRef.current;
    const muted = sourceRef.current === "filtered" ? restoredRef.current : filteredRef.current;
    if (audible) {
      setCurrentTime(audible.currentTime);
      if (muted && Math.abs(muted.currentTime - audible.currentTime) > MAX_DRIFT_SEC) {
        muted.currentTime = audible.currentTime;
      }
      if (audible.ended) {
        filteredRef.current?.pause();
        restoredRef.current?.pause();
        setPlaying(false);
        return;
      }
    }
    frameRef.current = requestAnimationFrame(tick);
  }, []);

  useEffect(() => {
    if (!playing) {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
      return;
    }
    frameRef.current = requestAnimationFrame(tick);
    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    };
  }, [playing, tick]);

  const play = useCallback(async () => {
    ensureGraph();
    await contextRef.current?.resume();
    const filtered = filteredRef.current;
    const restored = restoredRef.current;
    if (!filtered || !restored) return;
    restored.currentTime = filtered.currentTime;
    try {
      await Promise.all([filtered.play(), restored.play()]);
      setPlaying(true);
    } catch {
      setError("Playback was blocked by the browser.");
    }
  }, [ensureGraph]);

  const pause = useCallback(() => {
    filteredRef.current?.pause();
    restoredRef.current?.pause();
    setPlaying(false);
  }, []);

  const toggle = useCallback(() => {
    if (playing) pause();
    else void play();
  }, [pause, play, playing]);

  const seek = useCallback((seconds: number) => {
    const clamped = Math.max(0, seconds);
    if (filteredRef.current) filteredRef.current.currentTime = clamped;
    if (restoredRef.current) restoredRef.current.currentTime = clamped;
    setCurrentTime(clamped);
  }, []);

  const skip = useCallback(
    (delta: number) => {
      const audible = source === "filtered" ? filteredRef.current : restoredRef.current;
      seek(Math.min((audible?.currentTime ?? 0) + delta, duration));
    },
    [duration, seek, source],
  );

  return {
    source,
    setSource,
    soloHighBand,
    setSoloHighBand,
    playing,
    currentTime,
    duration,
    ready,
    error,
    play,
    pause,
    toggle,
    seek,
    skip,
  };
}
