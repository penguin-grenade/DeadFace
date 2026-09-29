import * as THREE from 'three';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import { RectAreaLightUniformsLib } from 'three/examples/jsm/lights/RectAreaLightUniformsLib.js';
import { assetManifest, assetUrl, gltfLoader } from '../engine/assets';
import type { Physics, SurfaceKind } from '../engine/physics';
import type { Level } from './level';
import type { MoonShafts } from '../engine/postfx';
import { applyLevelShading, levelUniforms, MAX_FLICKER } from './levelShading';

/** level/level.json, written by blender/build_level.py. Positions are three.js world space. */
interface LevelJSON {
  version: number;
  lightmap: { file: string; scale: number; size: number } | null;
  mask: { file: string } | null;
  probe: { file: string; pos: V3; boxMin: V3; boxMax: V3 };
  materials: Record<string, { tex: string; tile: number; surface: SurfaceKind; metal: number; tint: V3; rough: number }>;
  glass: Record<string, { tint: V3; opacity: number; rough: number; frosted: boolean }>;
  emissive: Record<string, { color: V3; strength: number; baked: boolean }>;
  colliders: (
    | { type: 'box'; pos: V3; half: V3; quat: [number, number, number, number]; surface: SurfaceKind }
    | { type: 'cyl'; pos: V3; radius: number; half: number; quat: [number, number, number, number]; surface: SurfaceKind }
  )[];
  lights: LightJSON[];
  spots: Record<string, { pos: V3; yaw: number }[]>;
  bounds: { min: V3; max: V3 };
  moon: { dir: V3; color: V3 };
}
type V3 = [number, number, number];
interface LightJSON {
  name: string;
  type: 'spot' | 'point' | 'area' | 'sun';
  mode: 'mixed' | 'baked' | 'realtime';
  pos?: V3;
  target?: V3;
  dir?: V3;
  intensity?: number;
  color: V3;
  angle?: number;
  penumbra?: number;
  shadow?: boolean;
  flicker?: boolean;
  volumetric?: number;
  distance?: number;
  size?: [number, number];
  power?: number;
  normal?: V3;
}

export interface LevelSpotLight {
  light: THREE.SpotLight;
  base: number;
  flicker: boolean;
  flickerIndex: number;
  volumetric: number;
  phase: number;
}

export interface BakedLevel extends Level {
  lamps: LevelSpotLight[];
  /** Registers a moving object: shadows near it refresh when it moves, and its reflections/ambient follow the bake. */
  track(object: THREE.Object3D, radius?: number, castsShadow?: boolean): void;
  /** Local ambient scale at a world position (1 = as bright as the probe spot). */
  ambientAt(p: THREE.Vector3): number;
  /** Moonlight shafts through the windows and skylights, for the volumetric pass. */
  moon: MoonShafts | null;
}

/**
 * Depth map of the static level as seen from the moon, rendered once: the
 * volumetric pass uses it to confine moonlight to the window and skylight shafts.
 */
function renderMoonShadow(renderer: THREE.WebGLRenderer, meshes: THREE.Mesh[], dir: THREE.Vector3, min: THREE.Vector3, max: THREE.Vector3) {
  const size = 2048;
  const rt = new THREE.WebGLRenderTarget(size, size, { depthBuffer: true });
  rt.depthTexture = new THREE.DepthTexture(size, size, THREE.UnsignedIntType);
  rt.depthTexture.compareFunction = THREE.LessEqualCompare;
  rt.depthTexture.minFilter = rt.depthTexture.magFilter = THREE.LinearFilter;
  const center = min.clone().add(max).multiplyScalar(0.5);
  const cam = new THREE.OrthographicCamera();
  cam.position.copy(center).addScaledVector(dir, -60);
  cam.up.set(0, 1, 0);
  cam.lookAt(center);
  cam.updateMatrixWorld();
  const box = new THREE.Box3();
  const c = new THREE.Vector3();
  for (let i = 0; i < 8; i++) {
    c.set(i & 1 ? max.x : min.x, i & 2 ? max.y : min.y, i & 4 ? max.z : min.z).applyMatrix4(cam.matrixWorldInverse);
    box.expandByPoint(c);
  }
  cam.left = box.min.x - 0.5;
  cam.right = box.max.x + 0.5;
  cam.bottom = box.min.y - 0.5;
  cam.top = box.max.y + 0.5;
  cam.near = -box.max.z - 1;
  cam.far = -box.min.z + 1;
  cam.updateProjectionMatrix();
  const scene = new THREE.Scene();
  const depthOnly = new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.DoubleSide });
  for (const m of meshes) {
    const copy = new THREE.Mesh(m.geometry, depthOnly);
    copy.matrixAutoUpdate = false;
    copy.matrix.copy(m.matrixWorld);
    scene.add(copy);
  }
  const prev = renderer.getRenderTarget();
  renderer.setRenderTarget(rt);
  renderer.clear();
  renderer.render(scene, cam);
  renderer.setRenderTarget(prev);
  const matrix = new THREE.Matrix4()
    .set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1)
    .multiply(cam.projectionMatrix)
    .multiply(cam.matrixWorldInverse);
  return { texture: rt.depthTexture, matrix };
}

const lin = (c: V3) => new THREE.Color().setRGB(c[0], c[1], c[2], THREE.LinearSRGBColorSpace);
const v3 = (a: V3) => new THREE.Vector3(a[0], a[1], a[2]);

async function fetchJSON<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url);
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}

/** Radiance HDR, or the same file wrapped as base64 JSON for hosts that refuse unknown file types. */
async function loadHDR(url: string): Promise<THREE.DataTexture> {
  const loader = new HDRLoader().setDataType(THREE.HalfFloatType);
  let buffer: ArrayBuffer;
  if (url.endsWith('.json')) {
    const b64 = (await (await fetch(url)).json()) as string;
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    buffer = bytes.buffer;
  } else {
    buffer = await (await fetch(url)).arrayBuffer();
  }
  const parsed = loader.parse(buffer);
  if (!parsed) throw new Error('probe: unreadable HDR');
  const tex = new THREE.DataTexture(parsed.data, parsed.width, parsed.height, THREE.RGBAFormat, parsed.type);
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.flipY = true;
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.needsUpdate = true;
  return tex;
}

interface TexSet {
  map: THREE.Texture;
  normalMap: THREE.Texture;
  roughnessMap: THREE.Texture;
}

export async function loadBakedLevel(scene: THREE.Scene, physics: Physics, renderer: THREE.WebGLRenderer): Promise<BakedLevel | null> {
  const base = assetUrl('level/');
  const json = await fetchJSON<LevelJSON>(`${base}level.json`);
  if (!json || json.version !== 1) return null;
  const { modelSuffix } = await assetManifest();

  const texLoader = new THREE.TextureLoader();
  const maxAniso = renderer.capabilities.getMaxAnisotropy();
  const loadTex = async (url: string, srgb: boolean, repeat: boolean) => {
    const t = await texLoader.loadAsync(url);
    t.flipY = false; // glTF UV convention
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    if (repeat) {
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.anisotropy = maxAniso;
    } else {
      // Lightmap-style atlases: no mips, so neighbouring charts never bleed together.
      t.generateMipmaps = false;
      t.minFilter = THREE.LinearFilter;
      t.channel = 1;
    }
    t.needsUpdate = true;
    return t;
  };
  const texSets = new Map<string, Promise<TexSet>>();
  const texSet = (name: string) => {
    let p = texSets.get(name);
    if (!p) {
      const t = assetUrl(`textures/${name}`);
      p = Promise.all([loadTex(`${t}_albedo.jpg`, true, true), loadTex(`${t}_normal.jpg`, false, true), loadTex(`${t}_roughness.jpg`, false, true)]).then(
        ([map, normalMap, roughnessMap]) => ({ map, normalMap, roughnessMap }),
      );
      texSets.set(name, p);
    }
    return p;
  };

  const levelFile = modelSuffix === '.glb' ? 'level.glb' : `level${modelSuffix}`;
  const probeFile = modelSuffix === '.glb' ? json.probe.file : `${json.probe.file}.json`;
  const [gltf, lightMap, mask, probeTex] = await Promise.all([
    gltfLoader().loadAsync(`${base}${levelFile}`),
    json.lightmap ? loadTex(`${base}${json.lightmap.file}`, true, false) : Promise.resolve(null),
    json.mask ? loadTex(`${base}${json.mask.file}`, false, false) : Promise.resolve(null),
    loadHDR(`${base}${probeFile}`),
    ...Object.values(json.materials).map((m) => texSet(m.tex)),
  ]);

  // --- Reflection probe --------------------------------------------------------------------------
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envRT = pmrem.fromEquirectangular(probeTex);
  pmrem.dispose();
  scene.environment = envRT.texture;
  scene.environmentIntensity = 1;
  // Same night sky as the Blender world, seen through the windows and skylights.
  scene.background = new THREE.Color().setRGB(0.01, 0.014, 0.026, THREE.LinearSRGBColorSpace);
  scene.fog = null;
  levelUniforms.uProbePos.value.fromArray(json.probe.pos);
  levelUniforms.uBoxMin.value.fromArray(json.probe.boxMin);
  levelUniforms.uBoxMax.value.fromArray(json.probe.boxMax);
  levelUniforms.uLevelMask.value = mask;
  const lmIntensity = json.lightmap ? Math.PI / json.lightmap.scale : 1;
  levelUniforms.uVertexLight.value = lmIntensity;

  // --- Materials ---------------------------------------------------------------------------------
  const materials = new Map<string, THREE.MeshStandardMaterial>();
  const flickerLamps = json.lights.filter((l) => l.type === 'spot' && l.flicker).slice(0, MAX_FLICKER);
  const surfaceMaterial = async (name: string, kind: 'lightmap' | 'vertex') => {
    const key = `${name}|${kind}`;
    const cached = materials.get(key);
    if (cached) return cached;
    const spec = json.materials[name];
    const t = await texSet(spec.tex);
    const m = new THREE.MeshStandardMaterial({
      name: key,
      ...t,
      color: lin(spec.tint),
      roughness: spec.rough,
      metalness: spec.metal,
      normalScale: new THREE.Vector2(1, -1),
      envMapIntensity: 1,
    });
    if (kind === 'lightmap' && lightMap) {
      m.lightMap = lightMap;
      m.lightMapIntensity = lmIntensity;
    }
    applyLevelShading(m, kind, { wet: name === 'floor' || name === 'floor_paint' });
    materials.set(key, m);
    return m;
  };
  const glassMaterial = (name: string) => {
    const g = json.glass[name];
    // Reflection strength is divided by opacity so the blend leaves a physically sized reflection.
    const m = new THREE.MeshStandardMaterial({
      name,
      // Clear glass has no diffuse term; frosted glass scatters a little.
      color: g.frosted ? lin(g.tint).multiplyScalar(0.15) : new THREE.Color(0, 0, 0),
      roughness: g.rough,
      metalness: 0,
      transparent: true,
      opacity: g.opacity,
      depthWrite: false,
      envMapIntensity: 1 / g.opacity,
      emissive: g.frosted ? new THREE.Color(0.012, 0.016, 0.026) : new THREE.Color(0, 0, 0),
      side: THREE.DoubleSide,
    });
    return applyLevelShading(m, 'dynamic');
  };
  const emissiveMaterial = (name: string) => {
    const e = json.emissive[name];
    const m = new THREE.MeshStandardMaterial({
      name,
      color: new THREE.Color(0.02, 0.02, 0.02),
      roughness: 0.35,
      metalness: 0,
      emissive: lin(e.color),
      emissiveIntensity: e.strength,
    });
    return applyLevelShading(m, 'dynamic', { flicker: flickerLamps.length > 0 });
  };

  const root = gltf.scene;
  root.updateMatrixWorld(true);
  const meshes: THREE.Mesh[] = [];
  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) meshes.push(o as THREE.Mesh);
  });

  const trimeshBySurface = new Map<SurfaceKind, { verts: number[]; idx: number[] }>();
  const addTrimesh = (mesh: THREE.Mesh, surface: SurfaceKind) => {
    const g = mesh.geometry;
    const pos = g.getAttribute('position');
    let bucket = trimeshBySurface.get(surface);
    if (!bucket) trimeshBySurface.set(surface, (bucket = { verts: [], idx: [] }));
    const base = bucket.verts.length / 3;
    const p = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      p.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
      bucket.verts.push(p.x, p.y, p.z);
    }
    const index = g.getIndex();
    if (index) for (let i = 0; i < index.count; i++) bucket.idx.push(base + index.getX(i));
    else for (let i = 0; i < pos.count; i++) bucket.idx.push(base + i);
  };

  let floorMesh: THREE.Mesh | null = null;
  const opaque: THREE.Mesh[] = [];
  for (const mesh of meshes) {
    const matName = (mesh.material as THREE.Material).name;
    const g = mesh.geometry;
    if (json.materials[matName]) {
      const vertexLit = g.getAttribute('color') !== undefined && !g.getAttribute('uv1');
      if (vertexLit) {
        g.setAttribute('bakedLight', g.getAttribute('color'));
        g.deleteAttribute('color');
      }
      mesh.material = await surfaceMaterial(matName, vertexLit ? 'vertex' : 'lightmap');
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      addTrimesh(mesh, json.materials[matName].surface);
      opaque.push(mesh);
      if (matName === 'floor' && !vertexLit) floorMesh = mesh;
    } else if (json.glass[matName]) {
      mesh.material = glassMaterial(matName);
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      mesh.renderOrder = 2;
      addTrimesh(mesh, 'glass');
    } else if (json.emissive[matName]) {
      mesh.material = emissiveMaterial(matName);
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      addTrimesh(mesh, 'glass');
    } else {
      console.warn(`level: no material spec for ${matName}`);
    }
  }
  scene.add(root);

  // --- Collision ---------------------------------------------------------------------------------
  const R = physics.R;
  for (const c of json.colliders) {
    if (c.type === 'box') {
      const q = { x: c.quat[0], y: c.quat[1], z: c.quat[2], w: c.quat[3] };
      physics.addFixed(R.ColliderDesc.cuboid(c.half[0], c.half[1], c.half[2]), v3(c.pos), q, c.surface);
    } else {
      physics.addFixed(R.ColliderDesc.cylinder(c.half, c.radius), v3(c.pos), null, c.surface);
    }
  }
  for (const [surface, b] of trimeshBySurface) {
    physics.addTrimesh(new Float32Array(b.verts), new Uint32Array(b.idx), surface);
  }

  // --- Lights ------------------------------------------------------------------------------------
  const lamps: LevelSpotLight[] = [];
  for (const L of json.lights) {
    if (L.mode === 'baked' && L.type === 'area' && L.size && L.power && L.pos) {
      // Baked for the level; a real-time copy lights moving things (static materials skip rect-area lights).
      RectAreaLightUniformsLib.init();
      const radiance = L.power / (Math.PI * L.size[0] * L.size[1]);
      const area = new THREE.RectAreaLight(lin(L.color), radiance, L.size[0], L.size[1]);
      area.position.fromArray(L.pos);
      const n = L.normal ? v3(L.normal) : new THREE.Vector3(0, -1, 0);
      area.lookAt(area.position.clone().add(n));
      scene.add(area);
      continue;
    }
    if (L.type !== 'spot' || L.mode === 'baked' || !L.pos || !L.target) continue;
    const light = new THREE.SpotLight(lin(L.color), L.intensity ?? 10, 0, L.angle ?? 0.9, L.penumbra ?? 0.5, 2);
    light.name = L.name;
    light.position.fromArray(L.pos);
    light.target.position.fromArray(L.target);
    if (L.shadow) {
      light.castShadow = true;
      light.shadow.mapSize.set(1024, 1024);
      light.shadow.camera.near = 0.15;
      light.shadow.camera.far = 22;
      light.shadow.bias = -0.00025;
      light.shadow.normalBias = 0.025;
      light.shadow.radius = 2.5;
      // Static shadow caching: re-rendered only when something moves inside the cone.
      light.shadow.autoUpdate = false;
      light.shadow.needsUpdate = true;
    }
    scene.add(light, light.target);
    const flickerIndex = L.flicker ? flickerLamps.indexOf(L) : -1;
    if (flickerIndex >= 0) levelUniforms.uFlickerPos.value[flickerIndex].fromArray(L.pos);
    lamps.push({ light, base: light.intensity, flicker: !!L.flicker, flickerIndex, volumetric: L.volumetric ?? 0, phase: Math.random() * 100 });
  }

  // Sodium street lamp outside the north windows: only its glow is visible from inside.
  const street = json.lights.find((l) => l.name === 'streetlight');
  if (street?.pos) {
    const glow = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 16, 8),
      new THREE.MeshBasicMaterial({ color: lin(street.color).multiplyScalar(40) }),
    );
    glow.position.fromArray(street.pos);
    scene.add(glow);
  }

  // --- Local ambient for moving objects -----------------------------------------------------------
  // Sample the baked floor light under a point, relative to the floor under the probe.
  let lmPixels: { data: Uint8ClampedArray; w: number; h: number } | null = null;
  if (lightMap && floorMesh) {
    const img = lightMap.image as HTMLImageElement | ImageBitmap;
    const w = 512;
    const h = Math.round((512 * img.height) / img.width);
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    if (ctx) {
      ctx.drawImage(img as CanvasImageSource, 0, 0, w, h);
      lmPixels = { data: ctx.getImageData(0, 0, w, h).data, w, h };
    }
  }
  const ray = new THREE.Raycaster();
  const down = new THREE.Vector3(0, -1, 0);
  const s2l = (v: number) => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  const floorLight = (p: THREE.Vector3) => {
    if (!lmPixels || !floorMesh) return null;
    ray.set(new THREE.Vector3(p.x, p.y + 0.5, p.z), down);
    ray.far = 20;
    const hit = ray.intersectObject(floorMesh, false)[0];
    const uv = hit?.uv1;
    if (!uv) return null;
    const x = Math.min(lmPixels.w - 1, Math.max(0, Math.floor(uv.x * lmPixels.w)));
    const y = Math.min(lmPixels.h - 1, Math.max(0, Math.floor(uv.y * lmPixels.h)));
    const i = (y * lmPixels.w + x) * 4;
    const d = lmPixels.data;
    return 0.2126 * s2l(d[i]) + 0.7152 * s2l(d[i + 1]) + 0.0722 * s2l(d[i + 2]);
  };
  const probeFloor = floorLight(v3(json.probe.pos)) ?? 1;
  const ambientAt = (p: THREE.Vector3) => {
    const l = floorLight(p);
    return l === null ? 1 : THREE.MathUtils.clamp(l / Math.max(probeFloor, 1e-4), 0.08, 1.6);
  };

  interface Tracked {
    object: THREE.Object3D;
    radius: number;
    last: THREE.Matrix4;
    materials: THREE.MeshStandardMaterial[];
    ambientTimer: number;
    castsShadow: boolean;
  }
  const tracked: Tracked[] = [];
  const patched = new WeakSet<THREE.Material>();
  const track = (object: THREE.Object3D, radius = 1, castsShadow = true) => {
    const mats: THREE.MeshStandardMaterial[] = [];
    object.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      const list = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      const own = list.map((m) => {
        if (!(m as THREE.MeshStandardMaterial).isMeshStandardMaterial) return m;
        // Own copy so each object can carry its own ambient level.
        const c = (m as THREE.MeshStandardMaterial).clone();
        if (!patched.has(c)) {
          applyLevelShading(c, 'dynamic');
          patched.add(c);
        }
        mats.push(c);
        return c;
      });
      mesh.material = Array.isArray(mesh.material) ? own : own[0];
    });
    tracked.push({ object, radius, last: new THREE.Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), materials: mats, ambientTimer: 0, castsShadow });
  };

  // --- Moonlight shafts -----------------------------------------------------------------------------
  let moon: MoonShafts | null = null;
  const moonLight = json.lights.find((l) => l.type === 'sun');
  if (moonLight?.dir) {
    const dir = v3(moonLight.dir).normalize();
    const shadow = renderMoonShadow(renderer, opaque, dir, v3(json.bounds.min), v3(json.bounds.max).add(new THREE.Vector3(0, 0.6, 0)));
    moon = { direction: dir, color: lin(moonLight.color).multiplyScalar(moonLight.intensity ?? 0.4), shadow: shadow.texture, matrix: shadow.matrix };
  }

  // --- Gameplay spots ----------------------------------------------------------------------------
  const spots = (k: string) => (json.spots[k] ?? []).map((s) => v3(s.pos));
  const spawnSpot = json.spots.spawn?.[0];
  const spawn = spawnSpot ? v3(spawnSpot.pos) : new THREE.Vector3(0, 0, 6);
  spawn.y += 0.1;

  const tmpP = new THREE.Vector3();
  const update = (dt: number, time: number) => {
    levelUniforms.uTime.value = time;
    for (const l of lamps) {
      if (!l.flicker) continue;
      const t = time + l.phase;
      const n = Math.sin(t * 13.1) * Math.sin(t * 7.7) * Math.sin(t * 2.3);
      const on = n > -0.35 ? 1 : 0.05 + Math.random() * 0.2;
      l.light.intensity = l.base * on;
      if (l.flickerIndex >= 0) levelUniforms.uFlicker.value[l.flickerIndex] = on;
    }
    // Moving objects: refresh the cached shadow maps they can appear in, and follow the local light level.
    for (const t of tracked) {
      t.object.updateWorldMatrix(true, false);
      const moved = !t.object.matrixWorld.equals(t.last);
      if (moved) {
        t.last.copy(t.object.matrixWorld);
        tmpP.setFromMatrixPosition(t.object.matrixWorld);
        if (t.castsShadow) {
          for (const l of lamps) {
            if (l.light.castShadow && l.light.position.distanceTo(tmpP) < 14 + t.radius) l.light.shadow.needsUpdate = true;
          }
        }
      }
      t.ambientTimer -= dt;
      if (t.ambientTimer <= 0) {
        t.ambientTimer = moved ? 0.1 : 0.5;
        const a = ambientAt(tmpP.setFromMatrixPosition(t.object.matrixWorld));
        for (const m of t.materials) m.envMapIntensity = a;
      }
    }
  };

  return {
    spawn,
    spawnYaw: spawnSpot?.yaw ?? 0,
    targetSpots: spots('target'),
    plateSpots: spots('plate'),
    propSpots: [...spots('can'), ...spots('prop')],
    update,
    lamps,
    track,
    ambientAt,
    moon,
  };
}

