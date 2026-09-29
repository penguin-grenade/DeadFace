/**
 * Procedural WebAudio sound. No sample files: every sound is synthesized so the
 * prototype has zero asset dependencies. The master chain runs through a
 * waveshaper + compressor to mimic a cheap bodycam mic clipping on gunshots,
 * and a generated impulse response gives an indoor concrete reverb.
 */
export class Audio {
  private ctx: AudioContext | null = null;
  private master!: GainNode;
  private reverb!: ConvolverNode;
  private reverbSend!: GainNode;
  private noise!: AudioBuffer;

  init() {
    if (this.ctx) {
      void this.ctx.resume();
      return;
    }
    const ctx = new AudioContext();
    this.ctx = ctx;

    const clip = ctx.createWaveShaper();
    const curve = new Float32Array(1024);
    for (let i = 0; i < curve.length; i++) {
      const x = (i / (curve.length - 1)) * 2 - 1;
      curve[i] = Math.tanh(x * 2.2) / Math.tanh(2.2);
    }
    clip.curve = curve;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14;
    comp.ratio.value = 8;
    comp.attack.value = 0.002;
    comp.release.value = 0.25;
    const hp = ctx.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = 90; // tiny mic, no real sub-bass

    this.master = ctx.createGain();
    this.master.gain.value = 0.8;
    this.master.connect(clip).connect(comp).connect(hp).connect(ctx.destination);

    this.reverb = ctx.createConvolver();
    this.reverb.buffer = this.impulse(1.8, 2.6);
    this.reverbSend = ctx.createGain();
    this.reverbSend.gain.value = 0.35;
    this.reverbSend.connect(this.reverb).connect(this.master);

    this.noise = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
    const d = this.noise.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  }

  private impulse(seconds: number, decay: number) {
    const ctx = this.ctx!;
    const len = Math.floor(ctx.sampleRate * seconds);
    const buf = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let c = 0; c < 2; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < len; i++) {
        const t = i / len;
        // Sparse early reflections, then dense tail.
        const early = i < ctx.sampleRate * 0.06 && Math.random() < 0.004 ? 3 : 0;
        d[i] = ((Math.random() * 2 - 1) + early) * Math.pow(1 - t, decay);
      }
    }
    return buf;
  }

  private out(dry: number, wet: number) {
    const ctx = this.ctx!;
    const g = ctx.createGain();
    g.gain.value = dry;
    g.connect(this.master);
    if (wet > 0) {
      const s = ctx.createGain();
      s.gain.value = wet;
      g.connect(s).connect(this.reverbSend);
    }
    return g;
  }

  private noiseBurst(dest: AudioNode, t: number, dur: number, type: BiquadFilterType, freq: number, q = 0.7, gain = 1) {
    const ctx = this.ctx!;
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    src.playbackRate.value = 0.8 + Math.random() * 0.4;
    const f = ctx.createBiquadFilter();
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    const g = ctx.createGain();
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(f).connect(g).connect(dest);
    src.start(t, Math.random() * 1.5, dur + 0.05);
  }

  gunshot() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const t = ctx.currentTime;
    const out = this.out(1.0, 1.0);
    // Crack
    this.noiseBurst(out, t, 0.09, 'highpass', 1800, 0.5, 1.6);
    // Body
    this.noiseBurst(out, t, 0.35, 'lowpass', 1400, 0.8, 1.4);
    // Thump
    const o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.setValueAtTime(160, t);
    o.frequency.exponentialRampToValueAtTime(45, t + 0.18);
    const g = ctx.createGain();
    g.gain.setValueAtTime(1.2, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.25);
    o.connect(g).connect(out);
    o.start(t);
    o.stop(t + 0.3);
    // Slide/mechanism clack
    this.noiseBurst(out, t + 0.045, 0.03, 'bandpass', 3200, 4, 0.4);
  }

  dryFire() {
    if (!this.ctx) return;
    this.noiseBurst(this.out(0.6, 0.1), this.ctx.currentTime, 0.025, 'bandpass', 2600, 6, 0.8);
  }

  reload(stage: 'out' | 'in' | 'slide') {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const out = this.out(0.5, 0.15);
    if (stage === 'out') this.noiseBurst(out, t, 0.06, 'bandpass', 1500, 3, 0.7);
    if (stage === 'in') {
      this.noiseBurst(out, t, 0.04, 'bandpass', 2200, 5, 0.9);
      this.noiseBurst(out, t + 0.03, 0.05, 'bandpass', 900, 3, 0.6);
    }
    if (stage === 'slide') {
      this.noiseBurst(out, t, 0.08, 'bandpass', 2800, 2, 0.6);
      this.noiseBurst(out, t + 0.09, 0.05, 'bandpass', 3400, 5, 0.9);
    }
  }

  impact(kind: 'metal' | 'concrete' | 'wood' | 'flesh', distance: number) {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const t = ctx.currentTime + Math.min(distance / 343, 0.1);
    const atten = 1 / (1 + distance * 0.25);
    const out = this.out(0.7 * atten, 0.4 * atten);
    if (kind === 'metal') {
      // Steel plate ring: inharmonic partials.
      for (const [f, a, d] of [[820, 0.5, 0.9], [1370, 0.35, 0.7], [2190, 0.25, 0.5], [3310, 0.15, 0.3]]) {
        const o = ctx.createOscillator();
        o.frequency.value = f * (0.97 + Math.random() * 0.06);
        const g = ctx.createGain();
        g.gain.setValueAtTime(a, t);
        g.gain.exponentialRampToValueAtTime(0.0001, t + d);
        o.connect(g).connect(out);
        o.start(t);
        o.stop(t + d + 0.05);
      }
      this.noiseBurst(out, t, 0.03, 'highpass', 3000, 0.5, 0.8);
    } else if (kind === 'wood') {
      this.noiseBurst(out, t, 0.08, 'bandpass', 700, 2, 1);
    } else if (kind === 'flesh') {
      this.noiseBurst(out, t, 0.07, 'lowpass', 500, 1, 1.2);
    } else {
      this.noiseBurst(out, t, 0.06, 'bandpass', 1900, 1.2, 0.9);
    }
  }

  footstep(intensity: number) {
    if (!this.ctx) return;
    const out = this.out(0.25 * intensity, 0.08);
    const t = this.ctx.currentTime;
    this.noiseBurst(out, t, 0.07, 'bandpass', 420 + Math.random() * 200, 1.5, 1);
    this.noiseBurst(out, t + 0.01, 0.04, 'highpass', 3500, 0.7, 0.25);
  }

  shellCasing() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const t = ctx.currentTime;
    const out = this.out(0.12, 0.05);
    for (let i = 0; i < 3; i++) {
      const o = ctx.createOscillator();
      o.frequency.value = 5200 + Math.random() * 1800;
      const g = ctx.createGain();
      const s = t + i * (0.06 + Math.random() * 0.05);
      g.gain.setValueAtTime(0.5 / (i + 1), s);
      g.gain.exponentialRampToValueAtTime(0.0001, s + 0.08);
      o.connect(g).connect(out);
      o.start(s);
      o.stop(s + 0.1);
    }
  }
}
