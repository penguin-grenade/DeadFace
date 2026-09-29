"""First-person arms in a two-handed pistol grip (skin-modifier sculpt).

Output: <out>/arms.glb, in the same space as pistol.glb (origin = pistol origin,
+Y forward, +Z up) so the game can parent both to one viewmodel pivot.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import common as c  # noqa: E402


def skin_chain(name, joints, bones, mat):
    names = list(joints)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([joints[n][0] for n in names], [(names.index(a), names.index(b)) for a, b in bones], [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new("Skin", "SKIN").use_smooth_shade = True
    for i, n in enumerate(names):
        obj.data.skin_vertices[0].data[i].radius = joints[n][1]
    obj.data.skin_vertices[0].data[0].use_root = True
    sub = obj.modifiers.new("Subsurf", "SUBSURF")
    sub.levels = sub.render_levels = 2
    obj.data.materials.append(mat)
    return obj


def build():
    glove = c.material("Glove", (0.022, 0.021, 0.02), roughness=0.75)
    sleeve = c.material("Sleeve", (0.035, 0.042, 0.05), roughness=0.95)

    # Right (firing) hand: palm behind the grip, fingers wrapping the front strap.
    right_hand = skin_chain("hand_r", {
        "wrist": ((0.018, -0.075, -0.085), (0.026, 0.02)),
        "palm": ((0.02, -0.03, -0.062), (0.022, 0.03)),
        "knuckles": ((0.018, 0.012, -0.06), (0.02, 0.028)),
        "fingers": ((-0.004, 0.024, -0.062), (0.014, 0.026)),
        "fingertips": ((-0.018, 0.012, -0.064), (0.011, 0.022)),
        "thumb_base": ((0.012, -0.03, -0.035), (0.012, 0.012)),
        "thumb_tip": ((0.017, 0.03, -0.01), (0.009, 0.009)),
    }, [("wrist", "palm"), ("palm", "knuckles"), ("knuckles", "fingers"), ("fingers", "fingertips"),
        ("palm", "thumb_base"), ("thumb_base", "thumb_tip")], glove)

    # Left (support) hand: palm on the left grip panel, fingers over the right hand.
    left_hand = skin_chain("hand_l", {
        "wrist": ((-0.05, -0.05, -0.1), (0.026, 0.02)),
        "palm": ((-0.03, -0.012, -0.07), (0.02, 0.03)),
        "knuckles": ((-0.01, 0.02, -0.078), (0.018, 0.026)),
        "fingers": ((0.018, 0.024, -0.085), (0.013, 0.024)),
        "fingertips": ((0.03, 0.0, -0.088), (0.01, 0.02)),
        "thumb_base": ((-0.024, 0.0, -0.04), (0.012, 0.012)),
        "thumb_tip": ((-0.02, 0.06, -0.012), (0.009, 0.009)),
    }, [("wrist", "palm"), ("palm", "knuckles"), ("knuckles", "fingers"), ("fingers", "fingertips"),
        ("palm", "thumb_base"), ("thumb_base", "thumb_tip")], glove)

    right_arm = skin_chain("arm_r", {
        "cuff": ((0.02, -0.085, -0.09), (0.036, 0.032)),
        "fore": ((0.07, -0.25, -0.16), (0.046, 0.042)),
        "elbow": ((0.13, -0.42, -0.24), (0.055, 0.05)),
    }, [("cuff", "fore"), ("fore", "elbow")], sleeve)
    left_arm = skin_chain("arm_l", {
        "cuff": ((-0.055, -0.06, -0.105), (0.036, 0.032)),
        "fore": ((-0.13, -0.22, -0.17), (0.046, 0.042)),
        "elbow": ((-0.2, -0.38, -0.25), (0.055, 0.05)),
    }, [("cuff", "fore"), ("fore", "elbow")], sleeve)
    return c.join("Arms", [right_hand, left_hand, right_arm, left_arm])


def main():
    args = c.parse_args()
    c.reset_scene()
    obj = build()
    c.export_glb(os.path.join(args.out, "arms.glb"), [obj])


if __name__ == "__main__":
    main()
