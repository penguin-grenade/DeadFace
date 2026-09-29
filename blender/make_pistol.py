"""Polymer-frame 9mm service pistol with a weapon-mounted light.

Output: <out>/pistol.glb with nodes Frame, Slide, Muzzle, Ejection, LightMount, LightLens.
Blender axes: +Y = muzzle direction, +Z = up. Origin = centre of the bore axis
at the rear of the frame rail, which is where the game's viewmodel pivots.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402


def build():
    polymer = c.material("Polymer", (0.018, 0.018, 0.017), roughness=0.62)
    grip_tex = c.material("PolymerStipple", (0.016, 0.016, 0.015), roughness=0.85)
    steel = c.material("SlideSteel", (0.03, 0.031, 0.033), roughness=0.32, metallic=1.0)
    barrel = c.material("BarrelSteel", (0.18, 0.17, 0.15), roughness=0.25, metallic=1.0)
    sight = c.material("SightDot", (0.9, 0.9, 0.85), roughness=0.4, emission=(0.8, 1.0, 0.6), emission_strength=0.3)
    lens = c.material("LightLens", (0.05, 0.05, 0.05), roughness=0.05, emission=(1.0, 0.96, 0.9), emission_strength=4.0)

    # ---- Frame -------------------------------------------------------------
    frame_parts = [
        c.box("dust_cover", (0.029, 0.125, 0.02), (0, 0.083, -0.001), polymer),
        c.box("frame_rear", (0.029, 0.05, 0.02), (0, -0.003, -0.001), polymer),
        # Raked grip: rotate so the butt sweeps rearward.
        c.box("grip", (0.03, 0.05, 0.112), (0, -0.012, -0.058), grip_tex, bevel=0.004, rotation=(math.radians(-18), 0, 0)),
        c.box("beavertail", (0.027, 0.02, 0.012), (0, -0.036, -0.004), polymer, bevel=0.004),
        c.box("mag_base", (0.031, 0.052, 0.009), (0, -0.03, -0.115), polymer, bevel=0.003, rotation=(math.radians(-18), 0, 0)),
        # Trigger guard: front post + bottom bar, squared-off like modern polymer frames.
        c.box("guard_bottom", (0.022, 0.058, 0.006), (0, 0.037, -0.036), polymer, bevel=0.002),
        c.box("guard_front", (0.022, 0.006, 0.03), (0, 0.064, -0.022), polymer, bevel=0.002),
        c.box("trigger", (0.006, 0.006, 0.022), (0, 0.035, -0.022), polymer, bevel=0.002, rotation=(math.radians(12), 0, 0)),
        c.box("rail", (0.024, 0.04, 0.006), (0, 0.122, -0.014), polymer, bevel=0.001),
    ]
    frame = c.join("Frame", frame_parts)

    # ---- Slide ---------------------------------------------------------------
    slide_body = c.box("slide_body", (0.027, 0.19, 0.031), (0, 0.075, 0.026), steel, bevel=0.0025)
    # Ejection port on the right.
    port = c.box("port_cut", (0.02, 0.04, 0.02), (0.012, 0.075, 0.04), None, bevel=0)
    c.boolean_cut(slide_body, port)
    # Rear serrations.
    for i in range(7):
        s = c.box(f"serr_{i}", (0.004, 0.0022, 0.028), (0.0145, -0.002 + i * 0.0045, 0.026), None, bevel=0)
        c.boolean_cut(slide_body, s)
        s2 = c.box(f"serrL_{i}", (0.004, 0.0022, 0.028), (-0.0145, -0.002 + i * 0.0045, 0.026), None, bevel=0)
        c.boolean_cut(slide_body, s2)
    slide_parts = [
        slide_body,
        c.box("front_sight", (0.0035, 0.004, 0.006), (0, 0.162, 0.044), steel, bevel=0.0006),
        c.box("front_dot", (0.0024, 0.0006, 0.0024), (0, 0.16, 0.0445), sight, bevel=0),
        c.box("rear_sight", (0.02, 0.006, 0.006), (0, -0.012, 0.044), steel, bevel=0.0008),
        c.box("rear_notch_l", (0.0025, 0.0006, 0.0024), (-0.005, -0.0151, 0.0445), sight, bevel=0),
        c.box("rear_notch_r", (0.0025, 0.0006, 0.0024), (0.005, -0.0151, 0.0445), sight, bevel=0),
        c.box("barrel_hood", (0.014, 0.034, 0.012), (0, 0.075, 0.037), barrel, bevel=0.001),
        c.cylinder("barrel_crown", 0.0065, 0.004, (0, 0.1705, 0.026), (math.radians(90), 0, 0), barrel, verts=24),
    ]
    slide = c.join("Slide", slide_parts)
    bore = c.cylinder("bore", 0.0045, 0.02, (0, 0.17, 0.026), (math.radians(90), 0, 0), None, verts=16)
    slide.modifiers.new("Bore", "BOOLEAN").object = bore
    bore.hide_render = bore.hide_viewport = True

    # ---- Weapon light ------------------------------------------------------------
    light_parts = [
        c.box("light_body", (0.03, 0.062, 0.03), (0, 0.13, -0.03), polymer, bevel=0.004),
        c.box("light_clamp", (0.026, 0.03, 0.01), (0, 0.125, -0.012), polymer, bevel=0.001),
        c.cylinder("light_bezel", 0.0145, 0.008, (0, 0.163, -0.03), (math.radians(90), 0, 0), barrel, verts=32, bevel=0.001),
        c.box("light_switch", (0.034, 0.012, 0.008), (0, 0.1, -0.03), polymer, bevel=0.002),
    ]
    light = c.join("WeaponLight", light_parts)
    lens_obj = c.cylinder("LightLens", 0.012, 0.001, (0, 0.1675, -0.03), (math.radians(90), 0, 0), lens, verts=32)

    root = c.empty("Pistol", (0, 0, 0))
    for o in (frame, slide, light, lens_obj):
        o.parent = root
    c.empty("Muzzle", (0, 0.178, 0.026), root)
    c.empty("Ejection", (0.016, 0.075, 0.042), root)
    c.empty("LightMount", (0, 0.17, -0.03), root)
    return root


def main():
    args = c.parse_args()
    c.reset_scene()
    root = build()
    objs = [root] + list(root.children)
    c.export_glb(os.path.join(args.out, "pistol.glb"), objs)


if __name__ == "__main__":
    main()
