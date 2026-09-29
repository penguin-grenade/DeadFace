"""Range props: training mannequin, hanging steel target, soda can, small cardboard box.

Outputs <out>/{mannequin,steel_target,can,box_small}.glb.
Pivot conventions match src/game/targets.ts:
  mannequin     origin at the feet, faces Blender -Y (three.js +Z)
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


def mannequin():
    """Skin-modifier body: a vertex skeleton with radii, subdivided into smooth forms."""
    # (name, position, skin radius x/y)
    joints = {
        "pelvis": ((0, 0, 0.98), (0.17, 0.12)),
        "waist": ((0, 0, 1.12), (0.155, 0.11)),
        "chest": ((0, 0, 1.32), (0.2, 0.13)),
        "neck": ((0, 0, 1.53), (0.06, 0.06)),
        "head": ((0, -0.01, 1.66), (0.1, 0.11)),
        "crown": ((0, 0, 1.76), (0.085, 0.095)),
        "sh_l": ((-0.21, 0, 1.45), (0.075, 0.075)),
        "sh_r": ((0.21, 0, 1.45), (0.075, 0.075)),
        "el_l": ((-0.27, 0.02, 1.17), (0.055, 0.055)),
        "el_r": ((0.27, 0.02, 1.17), (0.055, 0.055)),
        "wr_l": ((-0.3, -0.02, 0.92), (0.04, 0.035)),
        "wr_r": ((0.3, -0.02, 0.92), (0.04, 0.035)),
        "hand_l": ((-0.3, -0.02, 0.84), (0.045, 0.025)),
        "hand_r": ((0.3, -0.02, 0.84), (0.045, 0.025)),
        "hip_l": ((-0.1, 0, 0.9), (0.095, 0.095)),
        "hip_r": ((0.1, 0, 0.9), (0.095, 0.095)),
        "kn_l": ((-0.11, -0.01, 0.5), (0.068, 0.068)),
        "kn_r": ((0.11, -0.01, 0.5), (0.068, 0.068)),
        "an_l": ((-0.11, 0.02, 0.09), (0.05, 0.05)),
        "an_r": ((0.11, 0.02, 0.09), (0.05, 0.05)),
        "toe_l": ((-0.12, -0.14, 0.04), (0.045, 0.03)),
        "toe_r": ((0.12, -0.14, 0.04), (0.045, 0.03)),
    }
    bones = [
        ("pelvis", "waist"), ("waist", "chest"), ("chest", "neck"), ("neck", "head"), ("head", "crown"),
        ("chest", "sh_l"), ("chest", "sh_r"), ("sh_l", "el_l"), ("sh_r", "el_r"),
        ("el_l", "wr_l"), ("el_r", "wr_r"), ("wr_l", "hand_l"), ("wr_r", "hand_r"),
        ("pelvis", "hip_l"), ("pelvis", "hip_r"), ("hip_l", "kn_l"), ("hip_r", "kn_r"),
        ("kn_l", "an_l"), ("kn_r", "an_r"), ("an_l", "toe_l"), ("an_r", "toe_r"),
    ]
    names = list(joints)
    mesh = bpy.data.meshes.new("Mannequin")
    mesh.from_pydata([joints[n][0] for n in names], [(names.index(a), names.index(b)) for a, b in bones], [])
    obj = bpy.data.objects.new("Mannequin", mesh)
    bpy.context.scene.collection.objects.link(obj)
    skin = obj.modifiers.new("Skin", "SKIN")
    skin.use_smooth_shade = True
    for i, n in enumerate(names):
        obj.data.skin_vertices[0].data[i].radius = joints[n][1]
    obj.data.skin_vertices[0].data[names.index("pelvis")].use_root = True
    sub = obj.modifiers.new("Subsurf", "SUBSURF")
    sub.levels = sub.render_levels = 2
    cloth = c.material("Coverall", (0.09, 0.095, 0.075), roughness=0.95)
    obj.data.materials.append(cloth)

    # Plate carrier on the chest so hits read against a contrasting surface.
    vest = c.box("vest_front", (0.34, 0.035, 0.3), (0, -0.135, 1.3), c.material("Vest", (0.03, 0.035, 0.04), roughness=0.85), bevel=0.02)
    vest_b = c.box("vest_back", (0.34, 0.035, 0.3), (0, 0.135, 1.3), bpy.data.materials["Vest"], bevel=0.02)
    face = c.box("visor", (0.13, 0.04, 0.05), (0, -0.095, 1.69), c.material("Visor", (0.02, 0.02, 0.02), roughness=0.15, metallic=0.4), bevel=0.012)
    return c.join("Mannequin", [obj, vest, vest_b, face])


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
        ("mannequin", mannequin),
        ("steel_target", steel_target),
        ("can", can),
        ("box_small", lambda: box_small(os.path.abspath(args.textures))),
    ]:
        c.reset_scene()
        obj = fn()
        c.export_glb(os.path.join(args.out, f"{name}.glb"), [obj])


if __name__ == "__main__":
    main()
