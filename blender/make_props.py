"""Range props: hanging steel target, soda can, small cardboard box.
(The training mannequin has its own script, make_mannequin.py.)

Outputs <out>/{steel_target,can,box_small}.glb.
Pivot conventions match src/game/targets.ts:
  steel_target  origin at plate centre, face normal along Blender -Y
  can, box      origin at the centre of the object
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import bmesh  # noqa: E402
import common as c  # noqa: E402


def steel_target():
    paint = c.material("TargetPaint", (0.75, 0.74, 0.7), roughness=0.55, metallic=0.2)
    plate = c.cylinder("plate", 0.22, 0.012, (0, 0, 0), (math.radians(90), 0, 0), paint, verts=48, bevel=0.002)
    tab_l = c.box("tab_l", (0.03, 0.012, 0.06), (-0.12, 0, 0.23), paint, bevel=0.002)
    tab_r = c.box("tab_r", (0.03, 0.012, 0.06), (0.12, 0, 0.23), paint, bevel=0.002)
    return c.join("SteelTarget", [plate, tab_l, tab_r])


def can():
    alu = c.material("CanBody", (0.55, 0.08, 0.06), roughness=0.3, metallic=0.9)
    lid = c.material("CanLid", (0.8, 0.8, 0.82), roughness=0.25, metallic=1.0)
    body = c.cylinder("can_body", 0.033, 0.106, (0, 0, 0), (0, 0, 0), alu, verts=32)
    top = c.cylinder("can_top", 0.029, 0.008, (0, 0, 0.057), (0, 0, 0), lid, verts=32, bevel=0.002)
    bot = c.cylinder("can_bot", 0.029, 0.008, (0, 0, -0.057), (0, 0, 0), lid, verts=32, bevel=0.002)
    return c.join("Can", [body, top, bot])


def box_small(textures_dir):
    card = c.image_material(
        "Cardboard",
        os.path.join(textures_dir, "cardboard_albedo.jpg"),
        roughness=0.9,
        normal_path=os.path.join(textures_dir, "cardboard_normal.jpg"),
    )
    if not card.node_tree.nodes.get("Image Texture"):
        card.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.36, 0.24, 0.12, 1)
    tape = c.material("Tape", (0.55, 0.45, 0.3), roughness=0.35)
    b = c.box("box", (0.35, 0.35, 0.28), (0, 0, 0), card, bevel=0.004)
    # UV unwrap for the texture
    bm = bmesh.new()
    bm.from_mesh(b.data)
    uv = bm.loops.layers.uv.verify()
    for f in bm.faces:
        n = f.normal
        for loop in f.loops:
            co = loop.vert.co
            if abs(n.z) > 0.5:
                loop[uv].uv = (co.x + 0.5, co.y + 0.5)
            elif abs(n.x) > 0.5:
                loop[uv].uv = (co.y + 0.5, co.z + 0.5)
            else:
                loop[uv].uv = (co.x + 0.5, co.z + 0.5)
    bm.to_mesh(b.data)
    bm.free()
    t1 = c.box("tape_top", (0.06, 0.352, 0.002), (0, 0, 0.14), tape, bevel=0)
    return c.join("BoxSmall", [b, t1])


def main():
    def extra(p):
        p.add_argument("--textures", default=os.path.join(os.path.dirname(__file__), "..", "public", "textures"))

    args = c.parse_args(extra)
    for name, fn in [
        ("steel_target", steel_target),
        ("can", can),
        ("box_small", lambda: box_small(os.path.abspath(args.textures))),
    ]:
        c.reset_scene()
        obj = fn()
        c.export_glb(os.path.join(args.out, f"{name}.glb"), [obj])


if __name__ == "__main__":
    main()
