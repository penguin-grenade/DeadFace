/**
 * public/assets.json lists authored assets produced by the Blender pipeline
 * (blender/build_all.py rewrites it). Anything not listed falls back to the
 * procedural stand-ins, so the game runs with an empty manifest.
 */
export interface AssetManifest {
  models: string[]; // stems of public/models/<name>.glb
  textures: string[]; // stems of public/textures/<name>_{albedo,normal,roughness}.jpg
}

let manifest: Promise<AssetManifest> | null = null;

export function assetManifest(): Promise<AssetManifest> {
  manifest ??= fetch(`${import.meta.env.BASE_URL}assets.json`)
    .then((r) => (r.ok ? r.json() : { models: [], textures: [] }))
    .then((m) => ({ models: m.models ?? [], textures: m.textures ?? [] }))
    .catch(() => ({ models: [], textures: [] }));
  return manifest;
}

export const assetUrl = (path: string) => `${import.meta.env.BASE_URL}${path}`;
