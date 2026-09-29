import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

/**
 * public/assets.json lists authored assets produced by the Blender pipeline
 * (blender/build_all.py rewrites it). Anything not listed falls back to the
 * procedural stand-ins, so the game runs with an empty manifest.
 */
export interface AssetManifest {
  models: string[]; // stems of public/models/<name>.glb
  textures: string[]; // stems of public/textures/<name>_{albedo,normal,roughness}.jpg
  /**
   * Where each file under public/ was published, when that differs (tools/artifact.mjs):
   * content-hashed names, and .glb/.hdr files wrapped as base64 JSON for hosts that only
   * serve common web types. Only read from a manifest embedded in the page.
   */
  files?: Record<string, string>;
}

/** A manifest the packager wrote into the page itself, so a stale cached assets.json can't win. */
const embedded = ((): Partial<AssetManifest> | null => {
  const text = document.getElementById('asset-manifest')?.textContent;
  try {
    return text ? (JSON.parse(text) as Partial<AssetManifest>) : null;
  } catch {
    return null;
  }
})();

let manifest: Promise<AssetManifest> | null = null;

export function assetManifest(): Promise<AssetManifest> {
  manifest ??= (
    embedded
      ? Promise.resolve(embedded)
      : fetch(`${import.meta.env.BASE_URL}assets.json`)
          .then((r): Promise<Partial<AssetManifest>> => (r.ok ? r.json() : Promise.resolve({})))
          .catch((): Partial<AssetManifest> => ({}))
  ).then((m) => ({ models: m.models ?? [], textures: m.textures ?? [], files: m.files }));
  return manifest;
}

/** URL of a file under public/, following the packager's renames. */
export const assetUrl = (path: string) => `${import.meta.env.BASE_URL}${embedded?.files?.[path] ?? path}`;

/** Fetch a file published as a base64 JSON string (see tools/artifact.mjs). */
export async function fetchBase64(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const bin = atob((await r.json()) as string);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

/**
 * Load public/<path>, a .glb. When it's published as base64 JSON, the file is decoded here and
 * parsed in memory, with its images decoded through <img> elements: locked-down hosts (a claude.ai
 * artifact) refuse fetch() of anything but their own files, and GLTFLoader would otherwise fetch
 * data: and blob: URLs.
 */
export async function loadGLTF(path: string): Promise<GLTF> {
  const url = assetUrl(path);
  let lost = 0;
  const manager = new THREE.LoadingManager();
  manager.onError = () => lost++;
  // Meshopt: the pipeline compresses the big meshes.
  const loader = new GLTFLoader(manager).setMeshoptDecoder(MeshoptDecoder);
  let gltf: GLTF;
  if (url.endsWith('.json')) {
    loader.register((parser) => {
      parser.textureLoader = new THREE.TextureLoader(manager);
      return { name: 'image_element_textures' };
    });
    gltf = await loader.parseAsync(await fetchBase64(url), '');
  } else {
    gltf = await loader.loadAsync(url);
  }
  // GLTFLoader leaves out an image it can't decode and carries on: say so rather than quietly show an untextured model.
  if (lost) reportAssetProblem(`${path.split('/').pop()} textures`, new Error(`${lost} of its images didn't load`));
  return gltf;
}

/** Asset failures, for the start panel: the game keeps going on stand-ins, but it shouldn't be silent. */
export const assetProblems: string[] = [];
/** What the page's security policy blocked, to explain those failures. */
export const policyBlocks = new Set<string>();

export function reportAssetProblem(what: string, e: unknown) {
  console.warn(`${what} failed to load, using a stand-in`, e);
  let msg = e instanceof Error ? e.message : '';
  if (!msg && e instanceof Event && e.target instanceof HTMLImageElement) msg = `image ${e.target.src.split('/').pop()} didn't load`;
  assetProblems.push(msg ? `${what} (${msg.length > 90 ? `${msg.slice(0, 90)}…` : msg})` : what);
}

document.addEventListener('securitypolicyviolation', (e) => {
  const uri = e.blockedURI;
  policyBlocks.add(`${e.effectiveDirective} ${/^https?:/.test(uri) ? uri.split('/').pop() : uri.split(':')[0]}`);
});
