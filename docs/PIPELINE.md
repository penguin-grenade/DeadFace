# Asset pipeline (Blender -> glTF -> Three.js)

All scripts are plain Python against the Blender API and run headless, either
through a Blender install or the `bpy` wheel:

```bash
blender -b -P blender/build_all.py --          # Blender 4.2+ (tested 5.0.1)
python3 blender/build_all.py                   # pip install bpy  (needs Python 3.11)
python3 blender/make_pistol.py --out public/models   # any single step
python3 blender/preview.py --glb public/models/pistol.glb --png /tmp/pistol.png  # Cycles preview render
```

`build_all.py` bakes textures, builds every model, then rewrites
`public/assets.json`. The game only loads what the manifest lists, and falls
back to its procedural stand-ins for anything missing, so a partially built
asset set still runs.

## Scripts

| Script | Output | Notes |
| --- | --- | --- |
| `bake_textures.py` | `public/textures/<name>_{albedo,normal,roughness}.jpg` | Procedural node materials baked with Cycles. UVs are wrapped onto a 4D torus and fed to 4D Noise/Voronoi so every set tiles seamlessly. |
| `make_pistol.py` | `pistol.glb` | Bevelled polymer-frame pistol with boolean-cut serrations, ejection port, bore, sights, weapon light. |
| `make_arms.py` | `arms.glb` | Gloved hands and sleeves in a two-handed grip, sculpted with the Skin modifier. Same origin as the pistol. |
| `make_props.py` | `mannequin.glb`, `steel_target.glb`, `can.glb`, `box_small.glb` | Mannequin is a Skin-modifier body with a plate carrier. The box embeds the baked cardboard texture. |
| `preview.py` | PNG | Three-point-lit Cycles render of any .glb, handy for reviewing output on a headless machine. |
| `common.py` | | Shared helpers: bevelled boxes, cylinders, boolean cuts, joining with modifiers applied, glTF export. |

## Conventions the game relies on

- **Units:** metres.
- **Axes:** model in Blender with **+Y forward, +Z up**. The glTF exporter (`export_yup`) turns that into three.js **-Z forward, +Y up**.
- **Weapon nodes** (found by name in `src/game/weapon.ts`):
  - `Slide`: separate object with its origin at the weapon origin; the game offsets its local Z for cycling.
  - `Muzzle`, `Ejection`, `LightMount`: empties marking the flash/smoke point, casing spawn, and flashlight position.
- **Arms** share the pistol's origin so both parent to one viewmodel pivot.
- **Mannequin** origin at the feet, facing Blender -Y. Colliders in `targets.ts` assume ~1.8 m tall with the head centred near 1.7 m.
- **Steel target** origin at plate centre, face normal along Blender -Y. It hangs 0.45 m below the stand's crossbar.
- **Props** (`can`, `box_small`) are centred on their origin; the box is 0.35 x 0.35 x 0.28 m.
- **Textures:** 1024² JPEG, albedo in sRGB, normal in OpenGL (+Y) tangent space, roughness in all channels (the game reads G).

## Swapping in real assets

- **Scanned materials:** drop CC0 sets (ambientCG, Poly Haven) into `public/textures` using the same file names and make sure the stem is in `assets.json`. Wall textures map one tile per wall height, so vertical grime reads correctly.
- **Hand-made models:** export glTF 2.0 binary with the node names above into `public/models`, add the stem to `assets.json`.
- **Skinned characters:** export with an armature and actions; load with `GLTFLoader` and play clips via `THREE.AnimationMixer` (not wired up yet).
- **Compression:** for larger assets run `npx @gltf-transform/cli optimize in.glb out.glb --texture-compress webp` and enable `DRACOLoader`/`KTX2Loader` in the game.

## Suggested next pipeline steps

1. **Lightmaps:** build the warehouse shell in Blender, bake Cycles lightmaps to a second UV set, load as `lightMap` in three.js. This is the single largest realism upgrade for static interiors.
2. **Reflection probes:** bake a cubemap at a few points and use it as `envMap` per zone instead of the generic `RoomEnvironment`.
3. **Rigged hands:** replace the static arms with an armature and author reload/inspect/draw actions.
