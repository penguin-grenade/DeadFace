import * as THREE from 'three';
import { assetManifest } from '../engine/assets';

/**
 * Procedural, tileable PBR texture sets (albedo + normal + roughness) generated
 * on the CPU at startup. These are stand-ins: the Blender pipeline (see
 * docs/PIPELINE.md) can bake real scanned/authored textures to
 * public/textures/<name>_{albedo,normal,roughness}.jpg and they are picked up
 * automatically by loadOrGenerate().
 */

const SIZE = 512;

function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

/** Periodic value noise so every texture tiles seamlessly. */
class TileNoise {
  private lattice: Float32Array;
  private readonly N = 256;
  constructor(seed: number) {
    const r = rng(seed);
    this.lattice = new Float32Array(this.N * this.N);
    for (let i = 0; i < this.lattice.length; i++) this.lattice[i] = r();
  }
  private at(x: number, y: number) {
    return this.lattice[(y & 255) * this.N + (x & 255)];
  }
  /** u,v in [0,1), period = cells across the texture (power of two). */
  sample(u: number, v: number, period: number) {
    const x = u * period;
    const y = v * period;
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const xf = x - xi;
    const yf = y - yi;
    const sx = xf * xf * (3 - 2 * xf);
    const sy = yf * yf * (3 - 2 * yf);
    const m = period - 1;
    const x0 = xi & m, x1 = (xi + 1) & m, y0 = yi & m, y1 = (yi + 1) & m;
    const a = this.at(x0, y0), b = this.at(x1, y0), c = this.at(x0, y1), d = this.at(x1, y1);
    return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
  }
  fbm(u: number, v: number, base: number, octaves: number, gain = 0.5) {
    let amp = 1, sum = 0, norm = 0, p = base;
    for (let o = 0; o < octaves; o++) {
      sum += this.sample(u, v, p) * amp;
      norm += amp;
      amp *= gain;
      p *= 2;
    }
    return sum / norm;
  }
}

interface Layers {
  albedo: Float32Array; // rgb
  height: Float32Array;
  rough: Float32Array;
}

function makeLayers(): Layers {
  return {
    albedo: new Float32Array(SIZE * SIZE * 3),
    height: new Float32Array(SIZE * SIZE),
    rough: new Float32Array(SIZE * SIZE),
  };
}

function toTexture(data: Uint8ClampedArray, srgb: boolean) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = SIZE;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(SIZE, SIZE);
  img.data.set(data);
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.anisotropy = 8;
  return tex;
}

function finish(l: Layers, normalStrength: number): TextureSet {
  const a = new Uint8ClampedArray(SIZE * SIZE * 4);
  const n = new Uint8ClampedArray(SIZE * SIZE * 4);
  const r = new Uint8ClampedArray(SIZE * SIZE * 4);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = y * SIZE + x;
      a[i * 4] = Math.pow(l.albedo[i * 3], 1 / 2.2) * 255;
      a[i * 4 + 1] = Math.pow(l.albedo[i * 3 + 1], 1 / 2.2) * 255;
      a[i * 4 + 2] = Math.pow(l.albedo[i * 3 + 2], 1 / 2.2) * 255;
      a[i * 4 + 3] = 255;
      const hl = l.height[y * SIZE + ((x - 1 + SIZE) % SIZE)];
      const hr = l.height[y * SIZE + ((x + 1) % SIZE)];
      const hu = l.height[((y - 1 + SIZE) % SIZE) * SIZE + x];
      const hd = l.height[((y + 1) % SIZE) * SIZE + x];
      // Canvas rows go top->bottom while texture v goes bottom->top (flipY),
      // so the row above (hu) is +v. Normal = (-dh/du, -dh/dv, 1).
      const nx = (hl - hr) * normalStrength;
      const ny = (hd - hu) * normalStrength;
      const len = Math.hypot(nx, ny, 1);
      n[i * 4] = (nx / len * 0.5 + 0.5) * 255;
      n[i * 4 + 1] = (ny / len * 0.5 + 0.5) * 255;
      n[i * 4 + 2] = (1 / len * 0.5 + 0.5) * 255;
      n[i * 4 + 3] = 255;
      const rv = Math.min(1, Math.max(0.02, l.rough[i])) * 255;
      // roughnessMap reads the G channel.
      r[i * 4] = r[i * 4 + 1] = r[i * 4 + 2] = rv;
      r[i * 4 + 3] = 255;
    }
  }
  return { map: toTexture(a, true), normalMap: toTexture(n, false), roughnessMap: toTexture(r, false) };
}

export interface TextureSet {
  map: THREE.Texture;
  normalMap: THREE.Texture;
  roughnessMap: THREE.Texture;
}

function each(fn: (u: number, v: number, i: number, x: number, y: number) => void) {
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) fn(x / SIZE, y / SIZE, y * SIZE + x, x, y);
}

export function concrete(seed = 1): TextureSet {
  const n = new TileNoise(seed);
  const n2 = new TileNoise(seed + 7);
  const l = makeLayers();
  each((u, v, i) => {
    const base = n.fbm(u, v, 8, 7);
    const stain = n2.fbm(u, v, 4, 4);
    const fine = n.sample(u, v, 256);
    const pit = fine > 0.93 ? -0.6 : 0;
    const crackN = Math.abs(n2.fbm(u, v, 16, 3) - 0.5);
    const crack = crackN < 0.006 ? -0.8 : 0;
    const g = 0.36 + base * 0.14 - Math.max(0, stain - 0.55) * 0.35 + (crack + pit) * 0.08;
    l.albedo[i * 3] = g * 0.98;
    l.albedo[i * 3 + 1] = g * 0.97;
    l.albedo[i * 3 + 2] = g * 0.93;
    l.height[i] = base * 0.6 + fine * 0.15 + pit * 0.3 + crack;
    // Worn, polished patches are smoother.
    l.rough[i] = 0.78 + base * 0.15 - Math.max(0, stain - 0.6) * 0.8;
  });
  return finish(l, 2.2);
}

export function paintedPlaster(seed = 2, tint: [number, number, number] = [0.62, 0.64, 0.58]): TextureSet {
  const n = new TileNoise(seed);
  const n2 = new TileNoise(seed + 3);
  const l = makeLayers();
  each((u, v, i) => {
    const f = n.fbm(u, v, 16, 5);
    const peel = n2.fbm(u, v, 4, 5);
    const peeled = peel > 0.7;
    // v=0 is the top of the canvas = top of the wall; grime rises from the floor.
    const floorGrime = Math.pow(Math.max(0, v - 0.7) / 0.3, 2) * 0.35;
    const drip = n2.sample(u * 1.0, 0.0, 64) > 0.75 ? Math.max(0, v - 0.2) * 0.12 : 0;
    const dirt = Math.max(0, n.fbm(u, v, 4, 4) - 0.5) * 0.4 + floorGrime + drip;
    let r = tint[0], g = tint[1], b = tint[2];
    const peelT = Math.min(1, Math.max(0, (peel - 0.66) * 12));
    r = r * (1 - peelT) + 0.5 * peelT;
    g = g * (1 - peelT) + 0.48 * peelT;
    b = b * (1 - peelT) + 0.44 * peelT;
    const k = (0.92 + f * 0.1) * (1 - dirt);
    l.albedo[i * 3] = r * k;
    l.albedo[i * 3 + 1] = g * k;
    l.albedo[i * 3 + 2] = b * k * 0.97;
    l.height[i] = f * 0.25 + (peeled ? -0.4 : 0);
    l.rough[i] = peeled ? 0.95 : 0.72 + f * 0.1 + dirt * 0.2;
  });
  return finish(l, 1.6);
}

export function woodPlanks(seed = 3): TextureSet {
  const n = new TileNoise(seed);
  const l = makeLayers();
  const planks = 4;
  each((u, v, i) => {
    const plank = Math.floor(v * planks);
    const pv = v * planks - plank;
    const off = (plank * 0.37) % 1;
    const warp = n.fbm((u + off) % 1, v, 4, 3) * 0.25;
    const grain = Math.sin((pv + warp) * 60 + n.sample((u + off) % 1, v, 32) * 6) * 0.5 + 0.5;
    const seam = pv < 0.03 || pv > 0.97 ? 1 : 0;
    const tone = 0.85 + (plank % 3) * 0.07;
    const dirt = n.fbm(u, v, 8, 4);
    const k = tone * (0.75 + grain * 0.25) * (0.8 + dirt * 0.3) * (seam ? 0.35 : 1);
    l.albedo[i * 3] = 0.42 * k;
    l.albedo[i * 3 + 1] = 0.3 * k;
    l.albedo[i * 3 + 2] = 0.19 * k;
    l.height[i] = grain * 0.2 - seam * 1.0;
    l.rough[i] = 0.7 + grain * 0.15;
  });
  return finish(l, 2.5);
}

export function paintedMetal(seed = 4, paint: [number, number, number] = [0.08, 0.12, 0.2]): TextureSet {
  const n = new TileNoise(seed);
  const n2 = new TileNoise(seed + 11);
  const l = makeLayers();
  each((u, v, i) => {
    const rust = n.fbm(u, v, 8, 6);
    const scratch = Math.abs(n2.fbm(u * 1.0, v, 64, 2) - 0.5) < 0.01 ? 1 : 0;
    const isRust = rust > 0.62;
    let r = paint[0], g = paint[1], b = paint[2];
    let rough = 0.45 + n2.fbm(u, v, 16, 3) * 0.2;
    if (isRust) {
      const t = Math.min(1, (rust - 0.62) * 6);
      r = r * (1 - t) + 0.28 * t;
      g = g * (1 - t) + 0.12 * t;
      b = b * (1 - t) + 0.05 * t;
      rough = 0.9;
    }
    if (scratch) {
      r = g = b = 0.5;
      rough = 0.25;
    }
    l.albedo[i * 3] = r;
    l.albedo[i * 3 + 1] = g;
    l.albedo[i * 3 + 2] = b;
    l.height[i] = (isRust ? rust * 0.5 : 0) - scratch * 0.2;
    l.rough[i] = rough;
  });
  return finish(l, 1.5);
}

export function cardboard(seed = 5): TextureSet {
  const n = new TileNoise(seed);
  const l = makeLayers();
  each((u, v, i) => {
    const f = n.fbm(u, v, 16, 5);
    const flute = Math.sin(v * SIZE * 0.35) * 0.5 + 0.5;
    const k = 0.9 + f * 0.15;
    l.albedo[i * 3] = 0.52 * k;
    l.albedo[i * 3 + 1] = 0.38 * k;
    l.albedo[i * 3 + 2] = 0.22 * k;
    l.height[i] = flute * 0.05 + f * 0.1;
    l.rough[i] = 0.9;
  });
  return finish(l, 1.2);
}

/** Radial soft sprite used for muzzle flash, dust and sparks. */
export function softSprite(inner = [255, 230, 180], outer = [255, 120, 20]) {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d')!;
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, `rgba(${inner.join(',')},1)`);
  grad.addColorStop(0.25, `rgba(${outer.join(',')},0.8)`);
  grad.addColorStop(1, `rgba(${outer.join(',')},0)`);
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Muzzle flash: star-shaped burst. */
export function flashSprite() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d')!;
  g.translate(128, 128);
  const r = rng(99);
  for (let k = 0; k < 9; k++) {
    g.rotate((Math.PI * 2) / 9 + r() * 0.3);
    const len = 60 + r() * 60;
    const grad = g.createLinearGradient(0, 0, len, 0);
    grad.addColorStop(0, 'rgba(255,240,200,1)');
    grad.addColorStop(1, 'rgba(255,120,30,0)');
    g.fillStyle = grad;
    g.beginPath();
    g.moveTo(0, -10);
    g.lineTo(len, 0);
    g.lineTo(0, 10);
    g.fill();
  }
  const grad = g.createRadialGradient(0, 0, 0, 0, 0, 50);
  grad.addColorStop(0, 'rgba(255,255,240,1)');
  grad.addColorStop(1, 'rgba(255,160,60,0)');
  g.fillStyle = grad;
  g.fillRect(-60, -60, 120, 120);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Bullet hole decal: dark core, torn ring, alpha falloff. */
export function bulletHole(kind: 'hard' | 'soft') {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d')!;
  const r = rng(kind === 'hard' ? 5 : 6);
  g.translate(64, 64);
  if (kind === 'hard') {
    for (let k = 0; k < 40; k++) {
      const a = r() * Math.PI * 2;
      const len = 20 + r() * 38;
      g.strokeStyle = `rgba(40,38,35,${0.15 + r() * 0.3})`;
      g.lineWidth = 1 + r() * 2;
      g.beginPath();
      g.moveTo(0, 0);
      g.lineTo(Math.cos(a) * len, Math.sin(a) * len);
      g.stroke();
    }
    const grad = g.createRadialGradient(0, 0, 0, 0, 0, 34);
    grad.addColorStop(0, 'rgba(90,86,80,0.9)');
    grad.addColorStop(1, 'rgba(90,86,80,0)');
    g.fillStyle = grad;
    g.fillRect(-64, -64, 128, 128);
  }
  const core = g.createRadialGradient(0, 0, 0, 0, 0, 14);
  core.addColorStop(0, 'rgba(5,5,5,1)');
  core.addColorStop(0.7, 'rgba(15,12,10,0.95)');
  core.addColorStop(1, 'rgba(20,15,10,0)');
  g.fillStyle = core;
  g.beginPath();
  g.arc(0, 0, 14, 0, Math.PI * 2);
  g.fill();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Try /textures/<name>_*.jpg (from the Blender bake step) and fall back to the generator. */
export async function loadOrGenerate(name: string, gen: () => TextureSet): Promise<TextureSet> {
  const base = `${import.meta.env.BASE_URL}textures/${name}`;
  const loader = new THREE.TextureLoader();
  try {
    if (!(await assetManifest()).textures.includes(name)) return gen();
    const [map, normalMap, roughnessMap] = await Promise.all(
      ['albedo', 'normal', 'roughness'].map((k) => loader.loadAsync(`${base}_${k}.jpg`)),
    );
    map.colorSpace = THREE.SRGBColorSpace;
    for (const t of [map, normalMap, roughnessMap]) {
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.anisotropy = 8;
    }
    return { map, normalMap, roughnessMap };
  } catch {
    return gen();
  }
}
