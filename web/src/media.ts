/**
 * Media engine: two stems (guitar / everything else) + muted video, kept in sync.
 * HTMLMediaElement.playbackRate keeps the pitch when slowing down, which is
 * what a beginner wants. A Karplus-Strong synth can play the transcription,
 * and a metronome can click on the detected beats.
 */

export type Listener = () => void;

export class MediaEngine {
  guitar = new Audio();
  backing = new Audio();
  video: HTMLVideoElement | null = null;
  rate = 1;
  private listeners: Listener[] = [];
  loop: [number, number] | null = null;
  ctx: AudioContext | null = null;
  synthGain: GainNode | null = null;
  clickGain: GainNode | null = null;
  synthOn = false;
  clickOn = false;
  private scheduledUntil = 0;
  private lastTime = 0;
  private ksCache = new Map<number, AudioBuffer>();
  notes: { t: number; end: number; p: number; amp: number }[] = [];
  beats: number[] = [];
  downbeatOffset = 0;
  meter = 4;

  constructor() {
    for (const a of [this.guitar, this.backing]) {
      a.preload = "auto";
      (a as any).preservesPitch = true;
      a.crossOrigin = "anonymous";
    }
    this.guitar.addEventListener("ended", () => this.emit());
  }

  async load(base: string, media: { video: string | null; guitar: string; backing: string }, video: HTMLVideoElement | null) {
    this.pause();
    this.guitar.src = `${base}/${media.guitar}`;
    this.backing.src = `${base}/${media.backing}`;
    this.video = video;
    if (video && media.video) {
      video.src = `${base}/${media.video}`;
      video.muted = true;
      video.playsInline = true;
      (video as any).preservesPitch = true;
    }
    await Promise.all([this.guitar, this.backing].map((a) => new Promise((r) => {
      if (a.readyState >= 1) r(null);
      a.addEventListener("loadedmetadata", () => r(null), { once: true });
      a.addEventListener("error", () => r(null), { once: true });
    })));
  }

  on(fn: Listener) {
    this.listeners.push(fn);
  }
  private emit() {
    this.listeners.forEach((f) => f());
  }

  get duration() {
    return this.guitar.duration || 0;
  }
  get time() {
    return this.guitar.currentTime;
  }
  get playing() {
    return !this.guitar.paused;
  }

  private ensureCtx() {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.synthGain = this.ctx.createGain();
      this.synthGain.gain.value = 0.55;
      this.synthGain.connect(this.ctx.destination);
      this.clickGain = this.ctx.createGain();
      this.clickGain.gain.value = 0.35;
      this.clickGain.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") this.ctx.resume();
  }

  async play() {
    this.ensureCtx();
    this.scheduledUntil = this.time;
    const ps = [this.guitar.play(), this.backing.play()];
    if (this.video) this.video.play().catch(() => {});
    await Promise.allSettled(ps);
    this.emit();
  }

  pause() {
    this.guitar.pause();
    this.backing.pause();
    this.video?.pause();
    this.emit();
  }

  toggle() {
    this.playing ? this.pause() : this.play();
  }

  seek(t: number) {
    t = Math.max(0, Math.min(this.duration || t, t));
    this.guitar.currentTime = t;
    this.backing.currentTime = t;
    if (this.video) this.video.currentTime = t;
    this.scheduledUntil = t;
    this.emit();
  }

  setRate(r: number) {
    this.rate = r;
    for (const m of [this.guitar, this.backing, this.video]) if (m) m.playbackRate = r;
    this.emit();
  }

  setVolumes(guitar: number, backing: number) {
    this.guitar.volume = guitar;
    this.backing.volume = backing;
  }

  setSynth(on: boolean, vol = 0.55) {
    this.synthOn = on;
    this.ensureCtx();
    this.synthGain!.gain.value = vol;
    this.scheduledUntil = this.time;
  }

  setClick(on: boolean) {
    this.clickOn = on;
    this.ensureCtx();
    this.scheduledUntil = this.time;
  }

  /** Called every animation frame: sync, loop, schedule synth + clicks. */
  tick() {
    const t = this.time;
    if (this.loop && this.playing && t >= this.loop[1]) {
      this.seek(this.loop[0]);
      return;
    }
    if (this.playing) {
      const drift = this.backing.currentTime - t;
      if (Math.abs(drift) > 0.06) this.backing.currentTime = t;
      if (this.video && this.video.readyState >= 2) {
        const vd = this.video.currentTime - t;
        if (Math.abs(vd) > 0.25) this.video.currentTime = t;
        else this.video.playbackRate = this.rate * (1 - Math.max(-0.1, Math.min(0.1, vd * 0.8)));
      }
      if (t < this.lastTime - 0.05) this.scheduledUntil = t;
      this.schedule(t);
    }
    this.lastTime = t;
  }

  private schedule(now: number) {
    if (!this.ctx || (!this.synthOn && !this.clickOn)) return;
    const lookahead = 0.35 * this.rate;
    const from = Math.max(this.scheduledUntil, now);
    const to = now + lookahead;
    if (to <= from) return;
    const base = this.ctx.currentTime;
    const at = (mt: number) => base + (mt - now) / this.rate;
    if (this.synthOn) {
      for (const n of this.notes) {
        if (n.t >= from && n.t < to) this.pluck(n.p, at(n.t), Math.min(2.5, (n.end - n.t) / this.rate + 0.25), n.amp);
      }
    }
    if (this.clickOn) {
      this.beats.forEach((b, i) => {
        if (b >= from && b < to) this.click(at(b), (i - this.downbeatOffset) % this.meter === 0);
      });
    }
    this.scheduledUntil = to;
  }

  private ks(midi: number): AudioBuffer {
    const hit = this.ksCache.get(midi);
    if (hit) return hit;
    const ctx = this.ctx!;
    const sr = ctx.sampleRate;
    const f = 440 * Math.pow(2, (midi - 69) / 12);
    const len = Math.floor(sr * 2.6);
    const buf = ctx.createBuffer(1, len, sr);
    const out = buf.getChannelData(0);
    const N = Math.max(2, Math.round(sr / f));
    const ring = new Float32Array(N);
    // pluck position + slightly filtered noise burst
    let prev = 0;
    for (let i = 0; i < N; i++) {
      const r = Math.random() * 2 - 1;
      prev = 0.6 * r + 0.4 * prev;
      ring[i] = prev;
    }
    const decay = 0.996 - Math.max(0, (midi - 40)) * 0.00004;
    let idx = 0;
    for (let i = 0; i < len; i++) {
      const a = ring[idx];
      const b = ring[(idx + 1) % N];
      const v = decay * 0.5 * (a + b);
      ring[idx] = v;
      out[i] = a;
      idx = (idx + 1) % N;
    }
    // soft attack de-click
    for (let i = 0; i < 64; i++) out[i] *= i / 64;
    this.ksCache.set(midi, buf);
    return buf;
  }

  private pluck(midi: number, when: number, dur: number, amp: number) {
    const ctx = this.ctx!;
    const src = ctx.createBufferSource();
    src.buffer = this.ks(midi);
    const g = ctx.createGain();
    const peak = 0.25 + 0.5 * Math.min(1, amp);
    g.gain.setValueAtTime(peak, when);
    g.gain.setTargetAtTime(0, when + dur, 0.08);
    src.connect(g).connect(this.synthGain!);
    src.start(Math.max(when, ctx.currentTime));
    src.stop(when + dur + 0.6);
  }

  private click(when: number, accent: boolean) {
    const ctx = this.ctx!;
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.frequency.value = accent ? 1600 : 1000;
    g.gain.setValueAtTime(0.0001, when);
    g.gain.exponentialRampToValueAtTime(accent ? 0.9 : 0.5, when + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0001, when + 0.06);
    o.connect(g).connect(this.clickGain!);
    o.start(Math.max(when, ctx.currentTime));
    o.stop(when + 0.08);
  }
}
