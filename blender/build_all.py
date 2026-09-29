"""Run the whole asset pipeline and refresh public/assets.json.

    blender -b -P blender/build_all.py -- [--size 1024] [--level]
    python3 blender/build_all.py [--size 1024] [--level]   # with `pip install bpy`

--level also rebuilds and bakes the warehouse (public/level), which takes a while.

Order matters: textures are baked first because some models (the cardboard
box) embed baked textures.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PUBLIC = os.path.join(ROOT, "public")


def blender_binary():
    """Path to the Blender executable when running inside Blender, else None (pip `bpy`)."""
    try:
        import bpy

        path = bpy.app.binary_path
    except Exception:
        return None
    return path if path and os.path.basename(path).lower().startswith("blender") else None


def run(script, *args):
    """Each step gets a fresh process so scene state never leaks between scripts."""
    blender = blender_binary()
    if blender:
        cmd = [blender, "-b", "-P", os.path.join(HERE, script), "--", *args]
    else:
        cmd = [sys.executable, os.path.join(HERE, script), *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def meshopt(path):
    """Quantize + meshopt-compress a glb in place (the game registers MeshoptDecoder)."""
    cmd = ["npx", "--yes", "@gltf-transform/cli@4.5.1", "meshopt", path, path]
    print("+", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"meshopt step skipped ({e}); the uncompressed glb still loads", flush=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    size = argv[argv.index("--size") + 1] if "--size" in argv else "1024"
    tex_dir = os.path.join(PUBLIC, "textures")
    model_dir = os.path.join(PUBLIC, "models")
    run("bake_textures.py", "--out", tex_dir, "--size", size)
    run("make_pistol.py", "--out", model_dir)
    run("make_hands.py", "--out", model_dir)
    run("make_props.py", "--out", model_dir, "--textures", tex_dir)
    run("make_mannequin.py", "--out", model_dir)
    meshopt(os.path.join(model_dir, "mannequin.glb"))
    if "--level" in argv:
        # Lightmap + probe bake of the warehouse: slow (tens of minutes on a CPU).
        run("build_level.py", "--out", os.path.join(PUBLIC, "level"), "--tex", tex_dir)
        meshopt(os.path.join(PUBLIC, "level", "level.glb"))

    textures = sorted({f.rsplit("_", 1)[0] for f in os.listdir(tex_dir) if f.endswith("_albedo.jpg")})
    models = sorted(f[:-4] for f in os.listdir(model_dir) if f.endswith(".glb"))
    with open(os.path.join(PUBLIC, "assets.json"), "w") as fh:
        json.dump({"models": models, "textures": textures}, fh, indent=2)
        fh.write("\n")
    print("assets.json:", models, textures)


if __name__ == "__main__":
    main()
