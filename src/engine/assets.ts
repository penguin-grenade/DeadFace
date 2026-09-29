import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

/** glTF loader that also reads meshopt-compressed files (the pipeline compresses the big meshes). */
export const gltfLoader = () => new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);

/**
 * public/assets.json lists authored assets produced by the Blender pipeline
 * (blender/build_all.py rewrites it). Anything not listed falls back to the
 * procedural stand-ins, so the game runs with an empty manifest.
 */
export interface AssetManifest {
  models: string[]; // stems of public/models/<name>.glb
  textures: string[]; // stems of public/textures/<name>_{albedo,normal,roughness}.jpg
  /** File suffix for models. ".glb" by default; hosts that refuse .glb get embedded glTF as ".gltf.json". */
  modelSuffix: string;
}

let manifest: Promise<AssetManifest> | null = null;

export function assetManifest(): Promise<AssetManifest> {
  manifest ??= fetch(`${import.meta.env.BASE_URL}assets.json`)
    .then((r): Promise<Partial<AssetManifest>> => (r.ok ? r.json() : Promise.resolve({})))
    .catch((): Partial<AssetManifest> => ({}))
    .then((m) => ({ models: m.models ?? [], textures: m.textures ?? [], modelSuffix: m.modelSuffix ?? '.glb' }));
  return manifest;
}

export const assetUrl = (path: string) => `${import.meta.env.BASE_URL}${path}`;

export async function modelUrl(name: string) {
  return assetUrl(`models/${name}${(await assetManifest()).modelSuffix}`);
}
