"""Duty pistol viewmodel: polymer-frame 9 mm with night sights and a weapon light.

Output: <out>/pistol.glb with a baked 2K PBR atlas and the nodes the game uses:
Frame, Slide (animated), WeaponLight, LightLens and the empties Muzzle,
Ejection and LightMount.

Blender axes: +Y = muzzle direction, +Z = up, metres. The origin sits 26 mm
below the bore axis under the rear of the slide, where the viewmodel pivots.

Modelling: each part is a profile (side, section or plan view) extruded or
lofted with bmesh, shaped with exact booleans (ejection port, V-groove
serrations, sight notch, trigger-guard opening) and finished with 0.4-0.6 mm
machined edge bevels with hardened normals. Texturing is baked (texbake.py):
nitride slide with holster wear on the edges, stippled polymer grip, anodised
light body with bare aluminium showing on its corners, dust packed into the
crevices and fingerprint smudges in the roughness.

    python3 blender/make_pistol.py --out public/models [--size 2048] [--preview DIR]
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import common as c  # noqa: E402
import hardsurface as hs  # noqa: E402
import texbake as tb  # noqa: E402

BORE = 0.026                    # bore axis height
SY0, SY1 = -0.0185, 0.1675      # slide rear / front
SZ0, SZ1 = 0.0105, 0.0425       # slide bottom / top
SHW = 0.01275                   # slide half width
FHW = 0.0115                    # frame half width
LIGHT_Z = -0.0285               # weapon-light axis height
GRIP = math.radians(22)         # grip rake

# Surface types. Colours are linear; the baked atlas is built from these (see composite()).
SURF = {
    #            id  base colour               metal  rough
    "Nitride":  (1, (0.030, 0.031, 0.033), 0.85, 0.40),
    "Barrel":   (2, (0.085, 0.083, 0.080), 1.00, 0.30),
    "Polymer":  (3, (0.030, 0.030, 0.029), 0.00, 0.52),
    "Stipple":  (4, (0.027, 0.027, 0.026), 0.00, 0.78),
    "Anodized": (5, (0.024, 0.024, 0.026), 0.30, 0.42),
    "Steel":    (6, (0.300, 0.300, 0.295), 1.00, 0.34),
    "Paint":    (7, (0.720, 0.720, 0.690), 0.00, 0.50),
}


# ----------------------------------------------------------------------------- materials
def _bump(mat, kind):
    """Procedural micro-detail that only the tangent-space normal bake sees."""
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    bump = nt.nodes.new("ShaderNodeBump")
    if kind == "stipple":
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.inputs["Scale"].default_value = 1150.0      # ~0.9 mm raised stipple cells
        vor.inputs["Randomness"].default_value = 0.85
        nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeMapRange")
        ramp.inputs["From Min"].default_value = 0.05
        ramp.inputs["From Max"].default_value = 0.45
        ramp.inputs["To Min"].default_value = 1.0
        ramp.inputs["To Max"].default_value = 0.0
        nt.links.new(vor.outputs["Distance"], ramp.inputs["Value"])
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 5200.0
        add = nt.nodes.new("ShaderNodeMath")
        add.operation = "MULTIPLY_ADD"
        add.inputs[2].default_value = 0.0
        nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], add.inputs[1])
        nt.links.new(ramp.outputs["Result"], add.inputs[0])
        nt.links.new(add.outputs["Value"], bump.inputs["Height"])
        bump.inputs["Strength"].default_value = 0.9
        bump.inputs["Distance"].default_value = 0.00025
    else:
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = {"peel": 2600.0, "blast": 6000.0}[kind]
        noise.inputs["Detail"].default_value = 3.0
        nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        bump.inputs["Strength"].default_value = {"peel": 0.22, "blast": 0.12}[kind]
        bump.inputs["Distance"].default_value = 0.0001
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def materials():
    m = {}
    for name, (_id, col, metal, rough) in SURF.items():
        m[name] = c.material(name, col, roughness=rough, metallic=metal)
    _bump(m["Stipple"], "stipple")
    _bump(m["Polymer"], "peel")
    _bump(m["Anodized"], "blast")
    _bump(m["Nitride"], "blast")
    m["Tritium"] = c.material("Tritium", (0.55, 0.62, 0.5), roughness=0.15, emission=(0.35, 1.0, 0.3), emission_strength=6.0)
    m["LightLens"] = c.material("LightLens", (0.02, 0.02, 0.02), roughness=0.05, emission=(1.0, 0.97, 0.92), emission_strength=8.0)
    return m


# ----------------------------------------------------------------------------- helpers
def y_axis(y0, z=BORE, x=0.0):
    """Lathe frame whose local +Z runs along +Y starting at (x, y0, z)."""
    return Matrix.Translation((x, y0, z)) @ Matrix.Rotation(-math.pi / 2, 4, "X")


def x_axis(x0, y, z):
    return Matrix.Translation((x0, y, z)) @ Matrix.Rotation(math.pi / 2, 4, "Y")


def xs(s, a, b):
    """X interval [a, b] on the right side (s = 1) or mirrored to the left (s = -1)."""
    return min(s * a, s * b), max(s * a, s * b)


def join_cutters(name, objs):
    return hs.merge(name, objs) if len(objs) > 1 else objs[0]


def groove_prisms(name, ys, x_face, depth, half_w, z0, z1, side):
    """V-grooves cut into a side face at x = x_face (side = +1 right, -1 left)."""
    parts = []
    for i, yc in enumerate(ys):
        xo, xi = x_face + side * 0.002, x_face - side * depth
        pts = [(xo, yc - half_w), (xo, yc + half_w), (xi, yc + half_w * 0.2), (xi, yc - half_w * 0.2)]
        parts.append(hs.extrude_xy(f"{name}{i}", pts, z0, z1))
    return join_cutters(name, parts)


def sight_dot(name, y_face, x, z, ring_r, vial_r, m, facing=-1):
    """White ring insert with a tritium vial, set into a sight face that looks along `facing` Y."""
    ring = hs.cyl(f"{name}_ring", ring_r, (x, y_face - facing * 0.0004, z), (x, y_face + facing * 0.00012, z), 20, m["Paint"])
    vial = hs.cyl(f"{name}_vial", vial_r, (x, y_face - facing * 0.0004, z), (x, y_face + facing * 0.00022, z), 16, m["Tritium"])
    hs.bevel(ring, 0.00012, 1, 30)
    return ring, vial


# ----------------------------------------------------------------------------- slide
def build_slide(m):
    ch = 0.0031
    sec = [(SHW, SZ0), (SHW, SZ1 - ch), (SHW - ch, SZ1), (-SHW + ch, SZ1), (-SHW, SZ1 - ch), (-SHW, SZ0)]
    body = hs.extrude_xz("slide_body", sec, SY0, SY1, m["Nitride"])
    # Side view: bull nose under the muzzle, small chamfer on the top front edge.
    side = [(SY0 - 0.001, SZ0 - 0.001), (SY1 - 0.0095, SZ0 - 0.001), (SY1 + 0.0005, SZ0 + 0.0082),
            (SY1 + 0.0005, SZ1 - 0.0016), (SY1 - 0.0021, SZ1 + 0.001), (SY0 - 0.001, SZ1 + 0.001)]
    hs.cut(body, hs.extrude_yz("slide_side", side, -0.02, 0.02), "INTERSECT")
    # Plan view: the front of the slide tapers in (the "nose" bevel).
    plan = [(-SHW - 0.001, SY0 - 0.001), (SHW + 0.001, SY0 - 0.001), (SHW + 0.001, SY1 - 0.0135),
            (SHW - 0.0026, SY1 + 0.001), (-SHW + 0.0026, SY1 + 0.001), (-SHW - 0.001, SY1 - 0.0135)]
    hs.cut(body, hs.extrude_xy("slide_plan", plan, SZ0 - 0.002, SZ1 + 0.002), "INTERSECT")
    # Hollow it: the barrel hood shows through the ejection port, barrel and guide rod through the front.
    hs.cut(body, hs.box("slide_cavity", (-0.0091, 0.0015, SZ0 - 0.003), (0.0091, SY1 - 0.0045, 0.0372)))
    hs.cut(body, hs.cyl("barrel_hole", 0.0070, (0, SY1 - 0.006, BORE), (0, SY1 + 0.003, BORE), 40))
    hs.cut(body, hs.cyl("rod_hole", 0.0034, (0, SY1 - 0.006, 0.0146), (0, SY1 + 0.003, 0.0146), 28))
    # Ejection port: open on top and down the right flank.
    hs.cut(body, hs.extrude_xy("port", hs.rounded_rect(-0.0046, 0.0470, 0.0200, 0.1045, 0.0016, 4), 0.0302, 0.05))
    # V-groove serrations, rear and front, both flanks.
    rear = [-0.0150 + i * 0.0043 for i in range(8)]
    front = [0.1165 + i * 0.0043 for i in range(6)]
    for side_ in (1, -1):
        x_face = SHW * side_
        hs.cut(body, groove_prisms(f"serr_r{side_}", rear, x_face, 0.0011, 0.00105, SZ0 + 0.0026, SZ1 - ch - 0.0005, side_))
        hs.cut(body, groove_prisms(f"serr_f{side_}", front, x_face, 0.0010, 0.00100, SZ0 + 0.0030, SZ1 - ch - 0.0006, side_))
    hs.bevel(body, 0.00045, 3, 30)

    # Extractor plate behind the port (right side) and the polymer slide cover plate at the back.
    extractor = hs.extrude_yz("extractor", hs.rounded_rect(0.0352, 0.0266, 0.0505, 0.0338, 0.0018, 4), SHW - 0.0004, SHW + 0.00055, m["Nitride"])
    hs.bevel(extractor, 0.0002, 2, 30)
    plate = hs.extrude_xz("cover_plate", hs.rounded_rect(-0.0071, 0.0172, 0.0071, 0.0386, 0.0022, 4), SY0 - 0.0007, SY0 + 0.001, m["Polymer"])
    hs.bevel(plate, 0.00025, 2, 30)

    # Rear sight: ledge-style with a U notch, dovetailed into the slide top.
    rs_side = [(-0.0149, SZ1 - 0.0008), (-0.0034, SZ1 - 0.0008), (-0.0034, SZ1 + 0.0064),
               (-0.0098, SZ1 + 0.0064), (-0.0149, SZ1 + 0.0050)]
    rsight = hs.extrude_yz("rear_sight", rs_side, -0.0086, 0.0086, m["Nitride"])
    taper = [(-0.0096, SZ1 - 0.001), (0.0096, SZ1 - 0.001), (0.0078, SZ1 + 0.0072), (-0.0078, SZ1 + 0.0072)]
    hs.cut(rsight, hs.extrude_xz("rs_taper", taper, -0.017, 0.0), "INTERSECT")
    notch = [(0.0021, SZ1 + 0.01)] + hs.arc(0, SZ1 + 0.0055, 0.0021, 0, -math.pi, 10) + [(-0.0021, SZ1 + 0.01)]
    hs.cut(rsight, hs.extrude_xz("rs_notch", notch, -0.017, 0.0))
    hs.bevel(rsight, 0.0003, 2, 30)
    # Front sight post.
    fs_side = [(0.1500, SZ1 - 0.0008), (0.1572, SZ1 - 0.0008), (0.1550, SZ1 + 0.0062), (0.1500, SZ1 + 0.0062)]
    fsight = hs.extrude_yz("front_sight", fs_side, -0.0018, 0.0018, m["Nitride"])
    hs.bevel(fsight, 0.00028, 2, 30)
    dots = []
    for x in (-0.0050, 0.0050):
        dots += sight_dot(f"rdot{x > 0:d}", -0.0149, x, SZ1 + 0.0027, 0.00165, 0.00092, m)
    dots += sight_dot("fdot", 0.1500, 0.0, SZ1 + 0.0034, 0.00150, 0.00085, m)

    # Barrel: chamber hood (seen through the port), tube and crown; recoil guide rod below it.
    hood = hs.box("hood", (-0.0071, 0.0400, 0.0212), (0.0071, 0.1080, 0.0417), m["Barrel"])
    hs.cut(hood, hs.extrude_yz("hood_ramp", [(0.0395, 0.0360), (0.0440, 0.0425), (0.0380, 0.0425)], -0.01, 0.01))
    hs.bevel(hood, 0.0004, 2, 30)
    prof = [(0.0045, 0.1000), (0.0068, 0.1000), (0.0068, SY1 - 0.0016), (0.0063, SY1 - 0.0009),
            (0.0052, SY1 - 0.0009), (0.0046, SY1 - 0.0015)]
    barrel = hs.lathe("barrel", [(r, h) for r, h in prof], y_axis(0.0), 48, m["Barrel"], closed_profile=True)
    rod = hs.lathe("guide_rod", [(0, SY1 - 0.0016), (0.0027, SY1 - 0.0016), (0.0030, SY1 - 0.0019), (0.0030, 0.12), (0, 0.12)],
                   y_axis(0.0, 0.0146), 24, m["Steel"])
    for o in (barrel, rod):
        o.data.shade_smooth()

    parts = [body, extractor, plate, rsight, fsight, hood, barrel, rod] + dots
    return hs.merge("Slide", parts)


# ----------------------------------------------------------------------------- frame
def grip_rings(n_rings=26, n_pts=64):
    """Rings perpendicular to the raked grip axis: asymmetric superellipses (fuller
    backstrap, rounder front strap), a palm swell and a flared magazine well."""
    O = Vector((0.0, -0.0040, -0.0040))
    D = Vector((0.0, -math.sin(GRIP), -math.cos(GRIP)))
    F = Vector((0.0, math.cos(GRIP), -math.sin(GRIP)))
    X = Vector((1.0, 0.0, 0.0))
    L = 0.0890
    t_top = -0.022
    rings, ts = [], []
    for k in range(n_rings):
        u = k / (n_rings - 1)
        t = t_top + (L - t_top) * (u ** 0.92)
        s = max(0.0, t) / L
        a = 0.0148 + 0.0005 * math.sin(math.pi * min(1, s * 1.2)) + 0.0006 * max(0.0, (s - 0.93) / 0.07)
        # Flare out of the narrower frame instead of stepping out from under it.
        f = min(1.0, max(0.0, (t + 0.004) / 0.030))
        a = 0.0113 + (a - 0.0113) * f * f * (3 - 2 * f)
        bf = 0.0198 + 0.0006 * math.sin(math.pi * s) - 0.0012 * math.exp(-((t - 0.010) / 0.006) ** 2)
        bb = 0.0262 + 0.0022 * math.exp(-((s - 0.5) / 0.28) ** 2) + 0.0004 * max(0.0, (s - 0.93) / 0.07)
        ctr = O + D * t
        ring = []
        for i in range(n_pts):
            th = 2 * math.pi * i / n_pts
            cx, sy = math.cos(th), math.sin(th)
            n = 2.9 if sy > 0 else 3.4
            px = a * math.copysign(abs(cx) ** (2 / n), cx)
            pv = (bf if sy > 0 else bb) * math.copysign(abs(sy) ** (2 / n), sy)
            ring.append(tuple(ctr + X * px + F * pv))
        rings.append(ring)
        ts.append(t)
    return rings, ts, (O, D, F, X, L)


def build_frame(m):
    # Upper frame: receiver, dust cover and the beavertail tang, from the side.
    upper_pts = [(-0.0205, SZ0), (0.1530, SZ0), (0.1545, SZ0 - 0.0025), (0.1545, -0.0050), (0.1520, -0.0090),
                 (0.0640, -0.0090), (0.0600, -0.0040), (0.0040, -0.0040), (-0.0200, -0.0060),
                 (-0.0280, -0.0040), (-0.0302, 0.0012), (-0.0298, 0.0045), (-0.0255, 0.0090)]
    radii = [0.0005, 0.0005, 0.0012, 0.0015, 0.0030, 0.0020, 0.0020, 0.0, 0.0, 0.0040, 0.0025, 0.0020, 0.0040]
    upper = hs.extrude_yz("frame_upper", hs.fillet(upper_pts, radii, 4), -FHW, FHW, m["Polymer"])
    # Plan view: the tang narrows behind the slide.
    plan = [(-FHW - 0.001, 0.16), (-FHW - 0.001, -0.0195), (-0.0098, -0.0320), (0.0098, -0.0320), (FHW + 0.001, -0.0195), (FHW + 0.001, 0.16)]
    hs.cut(upper, hs.extrude_xy("frame_plan", plan, -0.05, 0.02), "INTERSECT")
    # Rail: narrower than the frame, with clamp grooves along both sides and one cross slot.
    for s in (1, -1):
        hs.cut(upper, hs.box(f"rail_side{s}", (s * 0.0106, 0.0655, -0.012), (s * 0.014, 0.160, -0.0038)))
        hs.cut(upper, hs.box(f"rail_groove{s}", (s * 0.0094, 0.0655, -0.0062), (s * 0.014, 0.160, -0.0046)))
    hs.cut(upper, hs.box("rail_slot", (-0.02, 0.1085, -0.0100), (0.02, 0.1135, -0.0064)))

    # Trigger guard (square-fronted), unioned into the frame, opening cut after the union.
    guard_pts = [(0.0000, -0.0030), (0.0630, -0.0030), (0.0656, -0.0085), (0.0656, -0.0330), (0.0630, -0.0386),
                 (0.0575, -0.0402), (0.0100, -0.0402), (0.0030, -0.0330)]
    guard = hs.extrude_yz("guard", hs.fillet(guard_pts, [0, 0.002, 0.003, 0.004, 0.004, 0.003, 0.006, 0], 4), -0.0088, 0.0088, m["Polymer"])
    hs.cut(upper, guard, "UNION", transfer=True)

    # Grip: lofted, trimmed flat where it disappears into the receiver.
    rings, ts, (O, D, F, X, L) = grip_rings()
    grip = hs.loft("grip", rings, m["Polymer"])
    grip.data.materials.append(m["Stipple"])
    n_pts = len(rings[0])
    for f, poly in enumerate(grip.data.polygons):
        k = f // n_pts
        if k < len(ts) - 1 and 0.008 < ts[k] < L - 0.004:
            poly.material_index = 1
    grip.data.shade_smooth()
    hs.cut(grip, hs.box("grip_trim", (-0.03, -0.06, -0.2), (0.03, 0.05, 0.0035)), "INTERSECT")
    hs.cut(upper, grip, "UNION", transfer=True)

    opening = [(0.0105, -0.0035), (0.0598, -0.0035), (0.0598, -0.0358), (0.0105, -0.0358)]
    hs.cut(upper, hs.extrude_yz("guard_opening", hs.fillet(opening, [0.002, 0.0070, 0.0060, 0.0040], 6), -0.02, 0.02))
    # Magazine well mouth: a shallow bevelled pocket around the baseplate.
    hs.bevel(upper, 0.0006, 3, 32)
    frame = upper

    # Magazine baseplate, with a finger lip at the front.
    ring = []
    for i in range(64):
        th = 2 * math.pi * i / 64
        cx, sy = math.cos(th), math.sin(th)
        n = 2.9 if sy > 0 else 3.4
        px = 0.0147 * math.copysign(abs(cx) ** (2 / n), cx)
        pv = (0.0212 if sy > 0 else 0.0262) * math.copysign(abs(sy) ** (2 / n), sy)
        ring.append((px, pv))
    base_rings = []
    for t in (L - 0.0005, L + 0.0080):
        ctr = O + D * t
        base_rings.append([tuple(ctr + X * px + F * pv) for px, pv in ring])
    base = hs.loft("baseplate", base_rings, m["Polymer"])
    hs.bevel(base, 0.0012, 3, 30)

    # Trigger blade with the blade safety.
    blade = [(0.0306, -0.0030), (0.0346, -0.0030), (0.0372, -0.0090), (0.0392, -0.0170), (0.0396, -0.0235),
             (0.0386, -0.0282), (0.0360, -0.0290), (0.0350, -0.0240), (0.0344, -0.0170), (0.0330, -0.0100)]
    trig = hs.extrude_yz("trigger", hs.fillet(blade, [0, 0, 0.004, 0.006, 0.004, 0.0015, 0.0012, 0.004, 0.006, 0.004], 4),
                         -0.0034, 0.0034, m["Polymer"])
    hs.bevel(trig, 0.0004, 2, 30)
    safety = [(0.0376, -0.0085), (0.0402, -0.0165), (0.0405, -0.0232), (0.0390, -0.0236), (0.0386, -0.0170), (0.0366, -0.0095)]
    tsafe = hs.extrude_yz("trigger_safety", hs.fillet(safety, 0.0008, 3), -0.0008, 0.0008, m["Polymer"])
    hs.bevel(tsafe, 0.0002, 1, 30)

    small = [trig, tsafe, base]
    # Slide stop levers (ambidextrous) and takedown tabs.
    stop = [(0.0330, 0.0048), (0.0558, 0.0048), (0.0574, 0.0066), (0.0558, 0.0090), (0.0405, 0.0090),
            (0.0368, 0.0099), (0.0332, 0.0084)]
    for s in (1, -1):
        lever = hs.extrude_yz(f"slide_stop{s}", hs.fillet(stop, [0.0008, 0.0008, 0.0012, 0.0008, 0.002, 0.0015, 0.0012], 3),
                              *xs(s, FHW - 0.0004, FHW + 0.0011), m["Nitride"])
        for i in range(3):
            y = 0.0342 + i * 0.0011
            x0, x1 = xs(s, FHW + 0.0007, FHW + 0.002)
            hs.cut(lever, hs.box(f"stop_rib{s}{i}", (x0, y, 0.0080), (x1, y + 0.0005, 0.0102)))
        tab = hs.extrude_yz(f"takedown{s}", hs.rounded_rect(0.0615, 0.0060, 0.0690, 0.0099, 0.0008, 3), *xs(s, FHW - 0.0004, FHW + 0.0008), m["Nitride"])
        for i in range(3):
            y = 0.0628 + i * 0.0018
            x0, x1 = xs(s, FHW + 0.0004, FHW + 0.002)
            hs.cut(tab, hs.box(f"tab_rib{s}{i}", (x0, y, 0.005), (x1, y + 0.0006, 0.011)))
        for o in (lever, tab):
            hs.bevel(o, 0.00018, 1, 30)
        small += [lever, tab]
    # Magazine release (left side), ribbed.
    mag = hs.extrude_yz("mag_release", hs.rounded_rect(0.0012, -0.0120, 0.0098, -0.0046, 0.0020, 4), *xs(-1, FHW - 0.001, FHW + 0.0014), m["Polymer"])
    for i in range(3):
        z = -0.0108 + i * 0.0022
        x0, x1 = xs(-1, FHW + 0.0009, FHW + 0.003)
        hs.cut(mag, hs.box(f"mag_rib{i}", (x0, 0.0, z), (x1, 0.012, z + 0.0007)))
    hs.bevel(mag, 0.00025, 2, 30)
    small.append(mag)
    # Cross pins.
    for i, (y, z, r) in enumerate(((0.0355, -0.0010, 0.0012), (0.0512, 0.0042, 0.0014), (-0.0100, 0.0015, 0.0012))):
        pin = hs.lathe(f"pin{i}", [(0, 0), (r, 0), (r, 2 * FHW + 0.0002), (0, 2 * FHW + 0.0002)], x_axis(-FHW - 0.0001, y, z), 20, m["Steel"])
        hs.bevel(pin, 0.0002, 1, 30)
        small.append(pin)
    return hs.merge("Frame", [frame] + small)


# ----------------------------------------------------------------------------- weapon light
def build_light(m):
    sec = [(-0.0152, -0.0105), (0.0152, -0.0105), (0.0152, -0.0440), (-0.0152, -0.0440)]
    body = hs.extrude_xz("light_body", hs.fillet(sec, [0.003, 0.003, 0.007, 0.007], 5), 0.0930, 0.1500, m["Anodized"])
    side = [(0.0920, -0.0095), (0.1510, -0.0095), (0.1510, -0.0450), (0.1140, -0.0450), (0.0920, -0.0330)]
    hs.cut(body, hs.extrude_yz("light_side", hs.fillet(side, [0.002, 0, 0, 0.008, 0.003], 5), -0.02, 0.02), "INTERSECT")
    # Battery-cap seam.
    for s in (1, -1):
        hs.cut(body, hs.box(f"seam{s}", (s * 0.0148, 0.1290, -0.047), (s * 0.02, 0.1297, -0.009)))
    hs.cut(body, hs.box("seam_b", (-0.02, 0.1290, -0.05), (0.02, 0.1297, -0.0436)))
    hs.bevel(body, 0.0005, 3, 30)

    # Head: knurled bezel ring around the lens.
    prof = [(0.0, 0.1440), (0.0128, 0.1440), (0.0170, 0.1478), (0.0172, 0.1510)]
    y = 0.1520
    for _ in range(5):
        prof += [(0.0172, y), (0.0167, y + 0.0003), (0.0167, y + 0.0011), (0.0172, y + 0.0014)]
        y += 0.0021
    prof += [(0.0172, 0.1655), (0.0165, 0.1682), (0.0151, 0.1687), (0.0146, 0.1680), (0.0146, 0.1668), (0.0, 0.1668)]
    head = hs.lathe("light_head", prof, y_axis(0.0, LIGHT_Z), 56, m["Anodized"])
    head.data.shade_smooth()
    hs.bevel(head, 0.0002, 1, 40, weighted=False)

    # Rail clamp: jaws along the rail, cross bar in the slot, knob on the right, bolt head on the left.
    parts = [body, head]
    for s in (1, -1):
        jaw = hs.extrude_xz(f"jaw{s}", [(s * 0.0096, -0.0110), (s * 0.0128, -0.0110), (s * 0.0128, -0.0042), (s * 0.0100, -0.0042),
                                        (s * 0.0092, -0.0054), (s * 0.0100, -0.0062)], 0.0990, 0.1410, m["Anodized"])
        hs.bevel(jaw, 0.0003, 2, 30)
        parts.append(jaw)
    bar = hs.box("clamp_bar", (-0.0096, 0.1090, -0.0106), (0.0096, 0.1130, -0.0066), m["Anodized"])
    hs.bevel(bar, 0.0002, 1, 30)
    knob = hs.lathe("clamp_knob", [(0, 0), (0.0046, 0), (0.0048, 0.0004), (0.0048, 0.0030), (0.0044, 0.0036), (0, 0.0036)],
                    x_axis(0.0126, 0.1110, -0.0085), 18, m["Anodized"])
    hs.bevel(knob, 0.0002, 1, 25)
    bolt = hs.lathe("clamp_bolt", [(0, 0), (0.0036, 0), (0.0036, 0.0022), (0.0030, 0.0027), (0, 0.0027)],
                    Matrix.Translation((-0.0126, 0.1110, -0.0085)) @ Matrix.Rotation(-math.pi / 2, 4, "Y"), 6, m["Steel"])
    hs.bevel(bolt, 0.0002, 1, 25)
    parts += [bar, knob, bolt]
    # Ambidextrous switch paddles at the rear.
    pad = [(0.0870, -0.0185), (0.1000, -0.0185), (0.1000, -0.0360), (0.0880, -0.0335), (0.0858, -0.0260)]
    for s in (1, -1):
        paddle = hs.extrude_yz(f"paddle{s}", hs.fillet(pad, [0.002, 0.001, 0.001, 0.003, 0.003], 4), *xs(s, 0.0146, 0.0170), m["Polymer"])
        for i in range(3):
            z = -0.0215 - i * 0.0036
            x0, x1 = xs(s, 0.0162, 0.02)
            hs.cut(paddle, hs.box(f"pad_rib{s}{i}", (x0, 0.084, z), (x1, 0.0995, z + 0.0012)))
        hs.bevel(paddle, 0.0003, 2, 30)
        parts.append(paddle)
    light = hs.merge("WeaponLight", parts)
    lens = hs.lathe("LightLens", [(0, 0.1668), (0.0146, 0.1668), (0.0146, 0.16715), (0, 0.16715)], y_axis(0.0, LIGHT_Z), 48, m["LightLens"])
    return light, lens


# ----------------------------------------------------------------------------- textures
def composite(ids, cover, aoe, pos, nrm, tnrm, bmin, bmax):
    """Mix the baked utility maps into baseColor / ORM / normal (all (h, w, 3) in [0, 1])."""
    h, w = ids.shape
    ao, edge, cav = aoe[..., 0], aoe[..., 1], aoe[..., 2]
    p01 = (pos - bmin) / (bmax - bmin).max()          # uniform scale keeps the noise isotropic
    y01 = (pos[..., 1] - bmin[1]) / (bmax[1] - bmin[1])

    base = np.zeros((h, w, 3), np.float32)
    metal = np.zeros((h, w), np.float32)
    rough = np.zeros((h, w), np.float32)
    for name, (i, col, mt, rg) in SURF.items():
        sel = ids == i
        base[sel] = col
        metal[sel] = mt
        rough[sel] = rg

    n_big = tb.fbm(p01, 6.0, 4, seed=1)
    n_mid = tb.fbm(p01, 28.0, 4, seed=2)
    n_fine = tb.fbm(p01, 140.0, 3, seed=3)
    crevice = tb.smoothstep(0.95, 0.55, cav)                      # 1 deep in grooves and corners
    convex = np.clip(edge * 3.0, 0, 1) * tb.smoothstep(0.75, 0.97, cav)

    coated = (ids == SURF["Nitride"][0]) | (ids == SURF["Anodized"][0]) | (ids == SURF["Barrel"][0])
    # Holster and handling wear: heavier towards the muzzle and on the sights, broken up by noise.
    wear_bias = 0.55 + 0.45 * tb.smoothstep(0.55, 0.95, y01)
    wear = tb.smoothstep(0.42, 0.75, convex * wear_bias * (0.55 + 0.9 * n_mid) + 0.12 * n_fine)
    wear *= coated
    bare = np.where((ids == SURF["Anodized"][0])[..., None], np.array((0.62, 0.62, 0.63), np.float32), np.array((0.50, 0.49, 0.47), np.float32))
    base = base * (1 - wear[..., None]) + bare * wear[..., None]
    metal = metal * (1 - wear) + wear
    rough = rough * (1 - wear) + 0.26 * wear

    # Finish variation and fingerprint smudges (oily = smoother) on the flat faces.
    smudge = tb.smoothstep(0.58, 0.75, tb.fbm(p01 * np.array((1.0, 0.35, 1.0), np.float32), 22.0, 4, seed=7))
    rough += (n_big - 0.5) * 0.10 + (n_fine - 0.5) * 0.06 - smudge * 0.12 * (1 - crevice)
    base *= (0.92 + 0.16 * n_big)[..., None]

    # Dust and pocket lint packed into crevices (light grey on polymer, dark grime on steel).
    dust = crevice * tb.smoothstep(0.35, 0.65, n_mid * 0.6 + n_fine * 0.4 + crevice * 0.3)
    polymer = (ids == SURF["Polymer"][0]) | (ids == SURF["Stipple"][0])
    dust_col = np.where(polymer[..., None], np.array((0.085, 0.080, 0.072), np.float32), np.array((0.035, 0.032, 0.028), np.float32))
    k = (dust * 0.8)[..., None]
    base = base * (1 - k) + dust_col * k
    rough = rough + dust * 0.25
    metal = metal * (1 - dust * 0.8)

    # Occlusion: long-range AO times the cavity term; a little of it is baked into the albedo.
    occ = np.clip(ao * (0.55 + 0.45 * cav), 0, 1)
    base *= (0.75 + 0.25 * cav)[..., None]

    base = tb.dilate(base, cover, 10)
    orm = np.stack([occ, np.clip(rough, 0.05, 1), np.clip(metal, 0, 1)], -1)
    orm = tb.dilate(orm, cover, 10)
    tn = tnrm.copy()
    tn[~cover] = (0.5, 0.5, 1.0)
    tn = tb.dilate(tn, cover, 10)
    return tb.linear_to_srgb(base), orm, tn


def bake_textures(objs, out, size, prefix="pistol"):
    t0 = time.time()
    tb.setup(samples=16)
    info = tb.atlas_uvs(objs, resolution=size, padding=max(4, size // 256))
    print(f"atlas: {info} in {time.time() - t0:.1f}s", flush=True)
    corners = np.array([o.matrix_world @ Vector(v) for o in objs for v in o.bound_box])
    bmin, bmax = corners.min(0).astype(np.float32) - 0.001, corners.max(0).astype(np.float32) + 0.001
    ids, cover = tb.bake_ids(objs, size, {n: v[0] for n, v in SURF.items()})
    aoe = tb.bake_ao_edge(objs, size // 2, samples=24, ao_dist=0.03, cavity_dist=0.0035, bevel_radius=0.0008)
    aoe = tb.upsample(tb.dilate(aoe, tb.upsample(cover.astype(np.float32), size // 2) > 0.5, 4), size)
    pos, nrm = tb.bake_position_normal(objs, size, bmin, bmax)
    tnrm = tb.bake_tangent_normal(objs, size, samples=6)
    base, orm, tn = composite(ids, cover, aoe, pos, nrm, tnrm, bmin, bmax)
    paths = {k: os.path.join(out, f"{prefix}_{k}.jpg") for k in ("basecolor", "orm", "normal")}
    tb.save(base, paths["basecolor"], 90)
    tb.save(orm, paths["orm"], 92)
    tb.save(tn, paths["normal"], 95)
    print(f"textures in {time.time() - t0:.1f}s", flush=True)
    return paths


def finalize_materials(objs, paths, name="PistolPBR"):
    pbr = tb.pbr_material(name, paths["basecolor"], paths["orm"], paths["normal"])
    keep = {"Tritium", "LightLens"}
    for o in objs:
        me = o.data
        old = [s.material for s in o.material_slots]
        new = []
        remap = []
        for m in old:
            target = m if m is not None and m.name in keep else pbr
            if target not in new:
                new.append(target)
            remap.append(new.index(target))
        idx = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("material_index", idx)
        idx = np.array(remap, np.int32)[idx] if remap else idx
        me.materials.clear()
        for m in new:
            me.materials.append(m)
        me.polygons.foreach_set("material_index", idx)
        me.update()


# ----------------------------------------------------------------------------- preview
def preview(out_dir, root):
    """Cycles stills: a 3/4 product shot and the in-game hip view."""
    sc = bpy.context.scene
    tb.setup(samples=48)
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = 960, 540
    sc.view_settings.view_transform = "AgX"
    world = sc.world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    probe = os.path.join(os.path.dirname(__file__), "..", "public", "level", "probe.hdr")
    if os.path.exists(probe):
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(os.path.abspath(probe))
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 1.0
    else:
        bg.inputs["Color"].default_value = (0.05, 0.05, 0.06, 1)
    key = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA"))
    key.data.energy = 6.0
    key.data.size = 0.3
    key.location = (0.25, -0.15, 0.35)
    key.rotation_euler = (Vector((0, 0.06, 0)) - key.location).to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(key)
    for i, (loc, e) in enumerate((((-0.3, 0.05, 0.25), 3.0), ((0.0, 0.1, 0.45), 2.0))):
        fill = bpy.data.objects.new(f"fill{i}", bpy.data.lights.new(f"fill{i}", "AREA"))
        fill.data.energy = e
        fill.data.size = 0.4
        fill.location = loc
        fill.rotation_euler = (Vector((0, 0.06, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(fill)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    shots = {
        "three_quarter": ((0.30, -0.16, 0.12), (0.0, 0.07, -0.02), 50),
        "left": ((-0.34, 0.06, 0.0), (0.0, 0.06, -0.03), 50),
        "hip": ((-0.045, -0.33, 0.125), None, 0),
    }
    for name, (loc, tgt, lens) in shots.items():
        cam.location = loc
        if tgt is None:
            # The game's hip pose: camera at the eye, 78 deg vertical FOV, weapon canted.
            cam.data.sensor_fit = "VERTICAL"
            cam.data.angle = math.radians(78)
            cam.rotation_euler = (math.radians(90), 0.0, 0.0)
            root.rotation_euler = (0.03, 0.1, 0.1)
        else:
            cam.data.sensor_fit = "AUTO"
            cam.data.lens = lens
            cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
            root.rotation_euler = (0, 0, 0)
        cam.data.clip_start = 0.005
        sc.render.filepath = os.path.join(out_dir, f"pistol_{name}.png")
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"  preview {name} {time.time() - t:.1f}s", flush=True)
    root.rotation_euler = (0, 0, 0)


# ----------------------------------------------------------------------------- main
def main():
    args = c.parse_args(lambda p: (p.add_argument("--size", type=int, default=2048),
                                   p.add_argument("--preview", default=""),
                                   p.add_argument("--no-bake", action="store_true")))
    c.reset_scene()
    t = time.time()
    m = materials()
    slide = build_slide(m)
    frame = build_frame(m)
    light, lens = build_light(m)
    print(f"modelled in {time.time() - t:.1f}s: " + ", ".join(f"{o.name} {sum(len(p.vertices) - 2 for p in o.data.polygons)} tris" for o in (slide, frame, light)), flush=True)

    baked = [slide, frame, light]
    if not args.no_bake:
        paths = bake_textures(baked, args.out, args.size)
        finalize_materials(baked, paths)

    root = c.empty("Pistol", (0, 0, 0))
    for o in (frame, slide, light, lens):
        o.parent = root
    c.empty("Muzzle", (0, 0.178, BORE), root)
    c.empty("Ejection", (0.016, 0.075, 0.042), root)
    c.empty("LightMount", (0, 0.1700, LIGHT_Z), root)
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        preview(args.preview, root)
    objs = [root] + list(root.children)
    c.export_glb(os.path.join(args.out, "pistol.glb"), objs)


if __name__ == "__main__":
    main()
