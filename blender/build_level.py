"""Builds "Range 4": a lightmapped night-time warehouse used as a shooting range.

    python3 blender/build_level.py --out public/level                # full build + bake
    python3 blender/build_level.py --preview /tmp/view.png           # Cycles render from spawn, no bake
    python3 blender/build_level.py --lm-size 1024 --samples 32       # quick bake

Outputs (in --out):
  level.glb     static meshes, one per material. TEXCOORD_0 = tiling UVs, TEXCOORD_1 = lightmap UVs
  level.json    materials, colliders, lights, gameplay spots, probe, lightmap scale
  lightmap.jpg  baked GI: indirect light of the real-time lamps + direct and indirect light of
                the baked-only lights (moon, street light, office troffer, exit sign, sky)
  mask.jpg      R wetness, G grime, B specular occlusion (lightmap UVs)
  probe.hdr     equirectangular reflection probe rendered from the middle of the hall

Lamps are "mixed": the game draws their direct light and shadows in real time (so normal maps,
specular and moving objects work) and adds their bounced light from the lightmap.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Euler, Matrix, Vector  # noqa: E402

import levelkit as lk  # noqa: E402
import level_props as P  # noqa: E402
from level_props import LM_BIG, LM_DETAIL, LM_PROP, LM_SMALL, LM_TINY, Frame, jitter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------------------------
# Layout constants (Blender space: +X east, +Y north, +Z up, metres)
# ------------------------------------------------------------------------------------
X0, X1, Y0, Y1 = -12.0, 12.0, -8.0, 18.0
WALL_T = 0.2
CMU_H = 2.4
EAVE = 6.7
DECK_Z = 6.6
GIRDER_TOP = 6.5
GIRDER_Y = (-2.0, 5.0, 12.0)
JOIST_X = [-11.25 + 1.5 * i for i in range(16)]
EAST_WINDOWS = (-5.0, 1.5, 8.5, 15.0)
NORTH_WINDOWS = (-1.5, 6.5)
WIN_W, WIN_Z0, WIN_Z1 = 2.4, 3.4, 4.6
ROLLER = (3.5, 7.5, 4.2)
MAN_DOOR = (-3.0, 0.92, 2.12)
SKYLIGHTS = ((4.5, -4.5), (4.5, 8.5), (-3.0, 1.5))
SKY_W, SKY_L = 1.2, 2.4
OFFICE = (-12.0, -4.5, 8.0, 18.0)
OFFICE_H = 3.2
OFFICE_CEIL = 2.75
OFFICE_DOOR = (-6.5, -5.6)
OFFICE_WIN = (11.4, 14.6, 0.95, 2.1)
MOON_DIR = Vector((-0.797, 0.29, -0.53)).normalized()
PROBE_POS = (0.0, 4.5, 1.6)


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(ROOT, "public", "level"))
    p.add_argument("--tex", default=os.path.join(ROOT, "public", "textures"))
    p.add_argument("--lm-size", type=int, default=2048)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--probe-samples", type=int, default=128)
    p.add_argument("--probe-size", type=int, default=1024)
    p.add_argument("--preview", default="", help="render a Cycles preview from the spawn to this PNG and exit")
    p.add_argument("--preview-cam", default="spawn")
    p.add_argument("--preview-samples", type=int, default=48)
    p.add_argument("--preview-width", type=int, default=960)
    p.add_argument("--save-blend", default="")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--no-bake", action="store_true")
    return p.parse_args(argv)


# ------------------------------------------------------------------------------------
# Walls
# ------------------------------------------------------------------------------------

def wall_quads(b, axis, fixed, s0, s1, z0, z1, normal, openings=(), reveal=0.0, reveal_dir=1):
    """Planar wall face with rectangular holes. axis 'x': the face spans X at y=fixed;
    'y': spans Y at x=fixed. `normal` is +1/-1 along the other axis. Openings are (s0, s1, z0, z1).
    `reveal` adds the jamb/head/sill faces of each opening, going `reveal_dir * reveal` behind the face."""
    cuts = sorted({s0, s1, *[o[0] for o in openings], *[o[1] for o in openings]})
    cuts = [c for c in cuts if s0 <= c <= s1]

    def P3(s, z, depth=0.0):
        off = fixed - normal * depth
        return (s, off, z) if axis == "x" else (off, s, z)

    def face(sa, sb, za, zb, depth=0.0):
        a, b_, c, d = P3(sa, za, depth), P3(sb, za, depth), P3(sb, zb, depth), P3(sa, zb, depth)
        # (a,b,c,d) faces -Y for axis x, +X for axis y.
        facing = -1 if axis == "x" else 1
        if facing == normal:
            b.quad(a, b_, c, d, uv="world")
        else:
            b.quad(b_, a, d, c, uv="world")

    for sa, sb in zip(cuts, cuts[1:]):
        if sb - sa < 1e-5:
            continue
        mid = (sa + sb) / 2
        holes = [(o[2], o[3]) for o in openings if o[0] < mid < o[1]]
        for (za, zb) in P._subtract([(z0, z1)], holes):
            face(sa, sb, za, zb)
    if reveal > 0:
        d = reveal * reveal_dir
        for (os0, os1, oz0, oz1) in openings:
            # Jambs (face into the opening), head (faces down), sill (faces up).
            for s, sgn in ((os0, 1), (os1, -1)):
                pa, pb = P3(s, oz0), P3(s, oz1)
                qa, qb = P3(s, oz0, d), P3(s, oz1, d)
                verts = [pa, qa, qb, pb]
                n = Vector(qa) - Vector(pa)
                n = n.cross(Vector(pb) - Vector(pa))
                want = Vector((sgn, 0, 0)) if axis == "x" else Vector((0, sgn, 0))
                if n.dot(want) < 0:
                    verts = [pa, pb, qb, qa]
                b.quad(*verts, uv="world")
            for z, sgn in ((oz1, -1), (oz0, 1)):
                if z <= z0 + 1e-4:
                    continue
                pa, pb = P3(os0, z), P3(os1, z)
                qa, qb = P3(os0, z, d), P3(os1, z, d)
                verts = [pa, pb, qb, qa]
                n = (Vector(pb) - Vector(pa)).cross(Vector(qa) - Vector(pa))
                if n.z * sgn < 0:
                    verts = [pa, qa, qb, pb]
                b.quad(*verts, uv="world")


def build_shell(k, rng):
    # Floor slab.
    k.B("floor", lm=LM_BIG, weld=True).quad((X0 - 0.2, Y0 - 0.2, 0), (X1 + 0.2, Y0 - 0.2, 0), (X1 + 0.2, Y1 + 0.2, 0), (X0 - 0.2, Y1 + 0.2, 0), uv="world")
    k.collide_box((0, 5, -0.25), (X1 - X0 + 2, Y1 - Y0 + 2, 0.5), (0, 0, 0), "concrete")

    cmu = k.B("cmu", lm=LM_BIG, weld=True)
    md = MAN_DOOR
    south_cmu_open = [(md[0] - md[1] / 2, md[0] + md[1] / 2, 0.0, md[2]), (ROLLER[0], ROLLER[1], 0.0, CMU_H + 0.01)]
    # Inner faces (CMU wainscot), top ledges.
    wall_quads(cmu, "x", Y0, X0, X1, 0.0, CMU_H, +1, south_cmu_open, reveal=WALL_T, reveal_dir=1)
    wall_quads(cmu, "x", Y1, X0, X1, 0.0, CMU_H, -1)
    wall_quads(cmu, "y", X0, Y0, Y1, 0.0, CMU_H, +1)
    wall_quads(cmu, "y", X1, Y0, Y1, 0.0, CMU_H, -1)
    for (a, b_) in P._subtract([(X0 - WALL_T, X1 + WALL_T)], [(ROLLER[0], ROLLER[1])]):
        cmu.quad((a, Y0 - WALL_T, CMU_H), (b_, Y0 - WALL_T, CMU_H), (b_, Y0, CMU_H), (a, Y0, CMU_H), uv="world")
    cmu.quad((X0 - WALL_T, Y1, CMU_H), (X1 + WALL_T, Y1, CMU_H), (X1 + WALL_T, Y1 + WALL_T, CMU_H), (X0 - WALL_T, Y1 + WALL_T, CMU_H), uv="world")
    cmu.quad((X0 - WALL_T, Y0, CMU_H), (X0, Y0, CMU_H), (X0, Y1, CMU_H), (X0 - WALL_T, Y1, CMU_H), uv="world")
    cmu.quad((X1, Y0, CMU_H), (X1 + WALL_T, Y0, CMU_H), (X1 + WALL_T, Y1, CMU_H), (X1, Y1, CMU_H), uv="world")
    # Outer faces of the CMU are never seen but close the shell for the bake (no light leaks).
    shell = k.B("concrete", tag="outer", kind="occluder", weld=True)
    for (axis, fixed, s0, s1, n) in (("x", Y0 - WALL_T, X0 - WALL_T, X1 + WALL_T, -1), ("x", Y1 + WALL_T, X0 - WALL_T, X1 + WALL_T, 1),
                                     ("y", X0 - WALL_T, Y0 - WALL_T, Y1 + WALL_T, -1), ("y", X1 + WALL_T, Y0 - WALL_T, Y1 + WALL_T, 1)):
        ops = [(ROLLER[0], ROLLER[1], 0.0, ROLLER[2]), (md[0] - md[1] / 2, md[0] + md[1] / 2, 0.0, md[2])] if fixed == Y0 - WALL_T else []
        wall_quads(shell, axis, fixed, s0, s1, 0.0, CMU_H, n, ops)
    # Colliders for the outer walls (thick, full height).
    for (c, s) in (((0, Y0 - 0.5, 3.4), (X1 - X0 + 2, 1.0, 6.8)), ((0, Y1 + 0.5, 3.4), (X1 - X0 + 2, 1.0, 6.8)),
                   ((X0 - 0.5, 5, 3.4), (1.0, Y1 - Y0 + 2, 6.8)), ((X1 + 0.5, 5, 3.4), (1.0, Y1 - Y0 + 2, 6.8)),
                   ((0, 5, DECK_Z + 0.5), (X1 - X0 + 2, Y1 - Y0 + 2, 1.0))):
        k.collide_box(c, s, (0, 0, 0), "concrete" if c[2] < 5 else "metal")

    # Metal cladding above the CMU with window and door openings.
    east_open = [(yc - WIN_W / 2, yc + WIN_W / 2, WIN_Z0, WIN_Z1) for yc in EAST_WINDOWS]
    north_open = [(xc - WIN_W / 2, xc + WIN_W / 2, WIN_Z0, WIN_Z1) for xc in NORTH_WINDOWS]
    south_open = [(ROLLER[0], ROLLER[1], CMU_H - 0.01, ROLLER[2])]
    clad = WALL_T + 0.02
    P.cladding(k, "x", Y0 - clad, X0 - clad, X1 + clad, CMU_H, EAVE, +1, south_open)
    P.cladding(k, "x", Y1 + clad, X0 - clad, X1 + clad, CMU_H, EAVE, -1, north_open)
    P.cladding(k, "y", X0 - clad, Y0 - clad, Y1 + clad, CMU_H, EAVE, +1)
    P.cladding(k, "y", X1 + clad, Y0 - clad, Y1 + clad, CMU_H, EAVE, -1, east_open)
    # Base flashing where the cladding meets the block.
    galv = k.B("galv", lm=LM_DETAIL)
    for (p0, p1) in (((X0, Y0 - 0.1, CMU_H), (X1, Y0 - 0.1, CMU_H)), ((X0, Y1 + 0.1, CMU_H), (X1, Y1 + 0.1, CMU_H)),
                     ((X0 - 0.1, Y0, CMU_H), (X0 - 0.1, Y1, CMU_H)), ((X1 + 0.1, Y0, CMU_H), (X1 + 0.1, Y1, CMU_H))):
        segs = [(p0, p1)]
        if p0[1] == Y0 - 0.1:
            segs = [((p0[0], p0[1], p0[2]), (ROLLER[0], p0[1], p0[2])), ((ROLLER[1], p0[1], p0[2]), (p1[0], p1[1], p1[2]))]
        for (a, b_) in segs:
            galv.box_between(Vector(a) + Vector((0, 0, 0.012)), Vector(b_) + Vector((0, 0, 0.012)), 0.22, 0.024)

    # Girts (C-channels, web horizontal) at 3.2 m and 4.8 m on every wall, broken at openings.
    for z in (3.2, 4.8):
        for (axis, fixed, s0, s1, n, ops) in (("x", Y0 - 0.12, X0, X1, 1, south_open), ("x", Y1 + 0.12, X0, X1, -1, north_open),
                                              ("y", X0 - 0.12, Y0, Y1, 1, []), ("y", X1 + 0.12, Y0, Y1, -1, east_open)):
            cuts = [(o[0] - 0.05, o[1] + 0.05) for o in ops if o[2] - 0.1 < z < o[3] + 0.1]
            for (a, b_) in P._subtract([(s0, s1)], cuts):
                if axis == "x":
                    pa, pb = Vector((a, fixed, z)), Vector((b_, fixed, z))
                    side = Vector((0, 1, 0))
                else:
                    pa, pb = Vector((fixed, a, z)), Vector((fixed, b_, z))
                    side = Vector((1, 0, 0))
                st = k.B("steel", tag="girt", lm=LM_DETAIL)
                st.box_between(pa, pb, 0.2, 0.004)
                for sgn in (-1, 1):
                    st.box_between(pa + side * (0.098 * sgn) + Vector((0, 0, 0.032)), pb + side * (0.098 * sgn) + Vector((0, 0, 0.032)), 0.004, 0.064)


def build_structure(k, rng):
    # Interior columns and girders.
    for y in GIRDER_Y:
        for x in (-6.0, 6.0):
            P.w_column(k, x, y, 0.0, GIRDER_TOP - 0.6, rng)
        P.girder(k, y, X0 + 0.15, X1 - 0.15, GIRDER_TOP)
    # Perimeter columns.
    for y in (Y0 + 0.16, -2.0, 5.0, 12.0, Y1 - 0.16):
        for x in (X0 + 0.16, X1 - 0.16):
            P.w_column(k, x, y, 0.0, GIRDER_TOP, rng, yaw=math.pi / 2)
    for x in (-6.0, 0.0):
        P.w_column(k, x, Y0 + 0.16, 0.0, GIRDER_TOP, rng)
    for x in (-6.0, 0.0, 6.0):
        P.w_column(k, x, Y1 - 0.16, 0.0, GIRDER_TOP, rng)
    # Ledger beams along the south/north walls and eave struts along east/west.
    st = k.B("steel")
    for y in (Y0 + 0.16, Y1 - 0.16):
        st.prism(lk.i_profile(0.4, 0.18, 0.009, 0.014), (X0 + 0.15, y, GIRDER_TOP - 0.2), (X1 - 0.15, y, GIRDER_TOP - 0.2), lm=LM_SMALL)
    for x in (X0 + 0.16, X1 - 0.16):
        st.prism(lk.i_profile(0.3, 0.15, 0.008, 0.012), (x, Y0 + 0.15, DECK_Z - 0.15), (x, Y1 - 0.15, DECK_Z - 0.15), lm=LM_SMALL)
    # Open-web joists.
    bays = [(Y0 + 0.16, GIRDER_Y[0]), (GIRDER_Y[0], GIRDER_Y[1]), (GIRDER_Y[1], GIRDER_Y[2]), (GIRDER_Y[2], Y1 - 0.16)]
    for x in JOIST_X:
        for (ya, yb) in bays:
            P.bar_joist(k, x, ya + 0.1, yb - 0.1, DECK_Z)
    # Horizontal bridging (angles along X) at mid-span of every bay, top and bottom chord.
    br = k.B("steel", tag="joist")
    for (ya, yb) in bays:
        ym = (ya + yb) / 2
        for z in (DECK_Z - 0.07, DECK_Z - 0.42):
            br.prism(lk.angle_profile(0.032, 0.004), (X0 + 0.3, ym, z), (X1 - 0.3, ym, z), lm=LM_TINY)
    # X-bracing rods in one bay of each long wall, with turnbuckles.
    for x in (X0 + 0.3, X1 - 0.3):
        for (a, b_) in (((x, -2.0, 0.4), (x, 5.0, 5.8)), ((x, 5.0, 0.4), (x, -2.0, 5.8))):
            st.rod(a, b_, 0.012, sides=6, lm=LM_TINY)
            mid = (Vector(a) + Vector(b_)) / 2
            d = (Vector(b_) - Vector(a)).normalized()
            st.rod(mid - d * 0.12, mid + d * 0.12, 0.02, sides=6, caps=True, lm=LM_TINY)
    # Roof deck with skylight openings.
    holes = [(x - SKY_W / 2, x + SKY_W / 2, y - SKY_L / 2, y + SKY_L / 2) for (x, y) in SKYLIGHTS]
    P.deck(k, X0 - 0.25, X1 + 0.25, Y0 - 0.25, Y1 + 0.25, DECK_Z, holes)
    # Skylight curbs and panels (frosted, lets moonlight in).
    glass_sky = k.glass_mat("skylight", tint=(0.75, 0.8, 0.82), transparency=0.8, rough=0.35, frosted=True)
    for (x, y) in SKYLIGHTS:
        g = k.B("galv", tag="curb", lm=LM_SMALL)
        for (p0, p1) in (((x - SKY_W / 2, y - SKY_L / 2), (x + SKY_W / 2, y - SKY_L / 2)), ((x - SKY_W / 2, y + SKY_L / 2), (x + SKY_W / 2, y + SKY_L / 2)),
                         ((x - SKY_W / 2, y - SKY_L / 2), (x - SKY_W / 2, y + SKY_L / 2)), ((x + SKY_W / 2, y - SKY_L / 2), (x + SKY_W / 2, y + SKY_L / 2))):
            g.box_between((p0[0], p0[1], DECK_Z + 0.15), (p1[0], p1[1], DECK_Z + 0.15), 0.03, 0.3)
        k.B("skylight", kind="glass").quad((x - SKY_W / 2, y - SKY_L / 2, DECK_Z + 0.3), (x - SKY_W / 2, y + SKY_L / 2, DECK_Z + 0.3),
                                           (x + SKY_W / 2, y + SKY_L / 2, DECK_Z + 0.3), (x + SKY_W / 2, y - SKY_L / 2, DECK_Z + 0.3))


def window(k, axis, fixed, sc, inward, rng, broken=()):
    """Steel-framed window in the cladding: 3 x 2 panes. `broken` lists pane indices without glass."""
    fr = k.B("paint_dark", tag="win", lm=LM_DETAIL)
    glass = k.B("glass", kind="glass")
    s0, s1 = sc - WIN_W / 2, sc + WIN_W / 2

    def PP(s, z, d):
        off = fixed + inward * d
        return Vector((s, off, z)) if axis == "x" else Vector((off, s, z))

    depth = 0.07
    # Outer frame.
    for (a, b_) in ((PP(s0, WIN_Z0, 0), PP(s1, WIN_Z0, 0)), (PP(s0, WIN_Z1, 0), PP(s1, WIN_Z1, 0))):
        fr.box_between(a + (b_ - a).normalized() * -0.04 + PP(0, 0, depth / 2) - PP(0, 0, 0), b_ + (b_ - a).normalized() * 0.04 + PP(0, 0, depth / 2) - PP(0, 0, 0), depth, 0.06)
    for s in (s0, s1):
        a, b_ = PP(s, WIN_Z0, depth / 2), PP(s, WIN_Z1, depth / 2)
        fr.box_between(a, b_, depth, 0.06, up=tuple(PP(1, 0, 0) - PP(0, 0, 0)))
    # Mullions and transom.
    for i in (1, 2):
        s = s0 + WIN_W * i / 3
        fr.box_between(PP(s, WIN_Z0, depth / 2), PP(s, WIN_Z1, depth / 2), 0.035, 0.04, up=tuple(PP(1, 0, 0) - PP(0, 0, 0)))
    zm = (WIN_Z0 + WIN_Z1) / 2
    fr.box_between(PP(s0, zm, depth / 2), PP(s1, zm, depth / 2), 0.035, 0.04)
    # Sill angle inside.
    fr.box_between(PP(s0 - 0.05, WIN_Z0 - 0.02, 0.08), PP(s1 + 0.05, WIN_Z0 - 0.02, 0.08), 0.16, 0.012)
    # Glass panes (single-sided quads facing inward; double-sided in the game).
    pw = WIN_W / 3
    ph = (WIN_Z1 - WIN_Z0) / 2
    idx = 0
    for r in range(2):
        for c in range(3):
            a0, a1 = s0 + c * pw + 0.02, s0 + (c + 1) * pw - 0.02
            z0, z1 = WIN_Z0 + r * ph + 0.02, WIN_Z0 + (r + 1) * ph - 0.02
            if idx in broken:
                # A few shards left in the frame.
                for (ss, zz) in ((a0, z0), (a1, z1)):
                    t = [PP(ss, zz, 0.035), PP(ss + (0.12 if ss == a0 else -0.12), zz, 0.035), PP(ss, zz + (0.18 if zz == z0 else -0.18), 0.035)]
                    n = (t[1] - t[0]).cross(t[2] - t[0])
                    want = PP(0, 0, 1) - PP(0, 0, 0)
                    if n.dot(want) < 0:
                        t = [t[0], t[2], t[1]]
                    glass.add_mesh([tuple(v) for v in t], [(0, 1, 2)], fuv=[[(0, 0), (1, 0), (0, 1)]])
            else:
                q = [PP(a0, z0, 0.035), PP(a1, z0, 0.035), PP(a1, z1, 0.035), PP(a0, z1, 0.035)]
                n = (q[1] - q[0]).cross(q[3] - q[0])
                want = PP(0, 0, 1) - PP(0, 0, 0)
                if n.dot(want) < 0:
                    q = [q[1], q[0], q[3], q[2]]
                glass.quad(*q)
            idx += 1


def build_openings(k, rng):
    k.glass_mat("glass", tint=(0.72, 0.78, 0.76), transparency=0.86, rough=0.06)
    brokens = {-5.0: (1, 4), 1.5: (), 8.5: (0,), 15.0: (2, 3, 5)}
    for yc in EAST_WINDOWS:
        window(k, "y", X1 + WALL_T + 0.02, yc, -1, rng, broken=brokens[yc])
    for xc in NORTH_WINDOWS:
        window(k, "x", Y1 + WALL_T + 0.02, xc, -1, rng, broken=(1,) if xc > 0 else ())

    # --- Roller door (closed) -------------------------------------------------------
    x0, x1, h = ROLLER
    door = k.B("paint_white", tag="roller", lm=0.8, sharp=80, weld=True)
    yd = Y0 - 0.1
    z = 0.06
    pitch = 0.08
    while z < h + 0.1:
        zb = z + pitch
        prof = [(z, 0.0), (z + 0.02, 0.012), (z + 0.055, 0.012), (z + 0.07, 0.0), (zb, 0.0)]
        for (pa, pb) in zip(prof, prof[1:]):
            a = (x0 - 0.05, yd + pa[1], pa[0])
            b_ = (x1 + 0.05, yd + pa[1], pa[0])
            c = (x1 + 0.05, yd + pb[1], pb[0])
            d = (x0 - 0.05, yd + pb[1], pb[0])
            door.quad(a, b_, c, d, uv="world", flat=((a[0], yd, a[2]), (b_[0], yd, b_[2]), (c[0], yd, c[2]), (d[0], yd, d[2])))
        z = zb
    dk = k.B("paint_dark", lm=LM_DETAIL)
    dk.box(((x0 + x1) / 2, yd + 0.02, 0.035), (x1 - x0 + 0.08, 0.05, 0.07))
    k.B("rubber", lm=LM_TINY).box(((x0 + x1) / 2, yd + 0.02, 0.006), (x1 - x0, 0.03, 0.012))
    for x in (x0 - 0.06, x1 + 0.06):
        dk.box((x, yd + 0.05, h / 2 + 0.1), (0.1, 0.12, h + 0.2))
        # Jamb guards.
        k.B("paint_yellow", tag="guard", lm=LM_SMALL).box((x + (-0.1 if x < x0 else 0.1), Y0 + 0.1, 0.6), (0.14, 0.2, 1.2))
    # Hood with the coil inside.
    k.B("galv", tag="hood", lm=LM_SMALL, sharp=60).lathe([(0.0, x0 - 0.2), (0.32, x0 - 0.2), (0.32, x1 + 0.2), (0.0, x1 + 0.2)],
                                                      Matrix.Translation((0, Y0 + 0.28, h + 0.3)) @ Matrix.Rotation(math.pi / 2, 4, "Y") @ Matrix.Diagonal((-1, 1, 1, 1)), segments=24)
    # Hand chain loop on the right side.
    P.chain(k, (x1 + 0.25, Y0 + 0.5, h + 0.2), (x1 + 0.25, Y0 + 0.5, 1.1))
    P.chain(k, (x1 + 0.3, Y0 + 0.5, h + 0.2), (x1 + 0.3, Y0 + 0.5, 1.1))
    k.collide_box(((x0 + x1) / 2, Y0 - 0.1, h / 2), (x1 - x0 + 0.3, 0.2, h), (0, 0, 0), "metal")

    # --- Man door + exit sign -------------------------------------------------------
    xc, w, hh = MAN_DOOR
    fr = k.B("paint_grey", tag="door", lm=LM_SMALL)
    for x in (xc - w / 2 - 0.025, xc + w / 2 + 0.025):
        fr.box((x, Y0 - 0.1, hh / 2), (0.05, 0.22, hh))
    fr.box((xc, Y0 - 0.1, hh + 0.025), (w + 0.1, 0.22, 0.05))
    leaf = k.B("paint_blue", tag="door", lm=LM_SMALL)
    leaf.box((xc, Y0 - 0.06, (hh - 0.02) / 2 + 0.01), (w - 0.01, 0.045, hh - 0.02))
    k.B("galv", lm=LM_TINY).box_between((xc - w / 2 + 0.1, Y0 - 0.0, 1.0), (xc + w / 2 - 0.06, Y0 - 0.0, 1.0), 0.05, 0.06)
    k.B("paint_dark", lm=LM_TINY).box((xc - 0.2, Y0 - 0.02, hh - 0.08), (0.35, 0.05, 0.06))  # closer
    k.B("paint_white", tag="sign", lm=LM_TINY).box((xc, Y0 + 0.04, hh + 0.28), (0.36, 0.08, 0.2))
    k.emit_mat("emit_exit", (1.0, 0.04, 0.02), 9.0, baked=True)
    k.B("emit_exit", kind="emissive").box((xc, Y0 + 0.081, hh + 0.28), (0.3, 0.002, 0.14))
    k.point_light("exit_glow", (xc, Y0 + 0.2, hh + 0.28), 0.35, (1.0, 0.05, 0.03), radius=0.1, mode="baked")
    k.collide_box((xc, Y0 - 0.08, hh / 2), (w, 0.1, hh), (0, 0, 0), "metal")


def build_office(k, rng):
    ox0, ox1, oy0, oy1 = OFFICE
    t = 0.12
    dw = k.B("drywall", lm=LM_BIG, weld=True)
    door = (OFFICE_DOOR[0], OFFICE_DOOR[1], 0.0, 2.1)
    win = OFFICE_WIN
    # South partition (y = oy0), both faces.
    wall_quads(dw, "x", oy0, ox0, ox1, 0.0, OFFICE_H, -1, [door], reveal=t, reveal_dir=-1)
    wall_quads(dw, "x", oy0 + t, ox0, ox1 - t, 0.0, OFFICE_H, +1, [door])
    # East partition (x = ox1), both faces.
    wall_quads(dw, "y", ox1, oy0, oy1, 0.0, OFFICE_H, +1, [win], reveal=t, reveal_dir=1)
    wall_quads(dw, "y", ox1 - t, oy0 + t, oy1, 0.0, OFFICE_H, -1, [win])
    # Tops of the partitions and the corner.
    dw.quad((ox0, oy0, OFFICE_H), (ox1, oy0, OFFICE_H), (ox1, oy0 + t, OFFICE_H), (ox0, oy0 + t, OFFICE_H), uv="world")
    dw.quad((ox1 - t, oy0 + t, OFFICE_H), (ox1, oy0 + t, OFFICE_H), (ox1, oy1, OFFICE_H), (ox1 - t, oy1, OFFICE_H), uv="world")
    # Office roof lid (seen from the hall only at a glance) and the plenum underside.
    k.B("plywood", tag="lid", lm=0.25).quad((ox0, oy0 + t, OFFICE_H), (ox1 - t, oy0 + t, OFFICE_H), (ox1 - t, oy1, OFFICE_H), (ox0, oy1, OFFICE_H), uv="world")
    k.B("paint_dark", tag="plenum", lm=0.25).quad((ox0, oy0 + t, OFFICE_H - 0.01), (ox0, oy1, OFFICE_H - 0.01), (ox1 - t, oy1, OFFICE_H - 0.01), (ox1 - t, oy0 + t, OFFICE_H - 0.01), uv="world")
    # Rubber base on both sides.
    base = k.B("plastic_black", tag="base", lm=LM_TINY)
    for (p0, p1) in (((ox0, oy0 - 0.004), (door[0], oy0 - 0.004)), ((door[1], oy0 - 0.004), (ox1 + 0.004, oy0 - 0.004)),
                     ((ox1 + 0.004, oy0), (ox1 + 0.004, oy1)), ((ox0, oy0 + t + 0.004), (door[0], oy0 + t + 0.004)),
                     ((door[1], oy0 + t + 0.004), (ox1 - t, oy0 + t + 0.004)), ((ox1 - t - 0.004, oy0 + t), (ox1 - t - 0.004, oy1))):
        base.box_between((p0[0], p0[1], 0.05), (p1[0], p1[1], 0.05), 0.008, 0.1)
    # Colliders (door gap left open, window blocked).
    k.collide_box(((ox0 + door[0]) / 2, oy0 + t / 2, OFFICE_H / 2), (door[0] - ox0, t, OFFICE_H), (0, 0, 0), "plaster")
    k.collide_box(((door[1] + ox1) / 2, oy0 + t / 2, OFFICE_H / 2), (ox1 - door[1], t, OFFICE_H), (0, 0, 0), "plaster")
    k.collide_box(((door[0] + door[1]) / 2, oy0 + t / 2, (2.1 + OFFICE_H) / 2), (door[1] - door[0], t, OFFICE_H - 2.1), (0, 0, 0), "plaster")
    k.collide_box((ox1 - t / 2, (oy0 + oy1) / 2, OFFICE_H / 2), (t, oy1 - oy0, OFFICE_H), (0, 0, 0), "plaster")

    # Door frame and open door leaf (swung out into the hall).
    fr = k.B("paint_grey", tag="door", lm=LM_SMALL)
    for x in (door[0] - 0.02, door[1] + 0.02):
        fr.box((x, oy0 + t / 2, 1.06), (0.04, t + 0.03, 2.12))
    fr.box(((door[0] + door[1]) / 2, oy0 + t / 2, 2.12), (door[1] - door[0] + 0.08, t + 0.03, 0.04))
    hinge = Vector((door[0], oy0 - 0.02, 0))
    ang = -1.75
    lf = Frame(hinge, ang)
    k.B("plywood", tag="door", lm=LM_SMALL).box(lf.p(0.45, -0.02, 1.035), (0.88, 0.04, 2.05), lf.r())
    k.B("galv", lm=LM_TINY).rod(lf.p(0.8, -0.06, 1.0), lf.p(0.8, -0.11, 1.0), 0.01, sides=8, caps=True)
    k.collide_box(lf.p(0.45, -0.02, 1.035), (0.88, 0.04, 2.05), lf.r(), "wood")

    # Interior window frame + glass (one pane cracked through).
    wf = k.B("paint_grey", tag="win", lm=LM_DETAIL)
    y0w, y1w, z0w, z1w = win
    for (a, b_) in (((ox1 - t / 2, y0w, z0w), (ox1 - t / 2, y1w, z0w)), ((ox1 - t / 2, y0w, z1w), (ox1 - t / 2, y1w, z1w))):
        wf.box_between(a, b_, t + 0.02, 0.04)
    for y in (y0w, (y0w + y1w) / 2, y1w):
        wf.box_between((ox1 - t / 2, y, z0w), (ox1 - t / 2, y, z1w), t + 0.02, 0.04, up=(1, 0, 0))
    gl = k.B("glass", kind="glass")
    ym = (y0w + y1w) / 2
    gl.quad((ox1 - t / 2, y0w + 0.02, z0w + 0.02), (ox1 - t / 2, ym - 0.02, z0w + 0.02), (ox1 - t / 2, ym - 0.02, z1w - 0.02), (ox1 - t / 2, y0w + 0.02, z1w - 0.02))
    # Right pane broken out: jagged shards only.
    for (yy, zz, dy, dz) in ((ym + 0.02, z0w + 0.02, 0.35, 0.22), (y1w - 0.02, z1w - 0.02, -0.28, -0.3), (y1w - 0.02, z0w + 0.02, -0.2, 0.16)):
        gl.add_mesh([(ox1 - t / 2, yy, zz), (ox1 - t / 2, yy + dy, zz), (ox1 - t / 2, yy, zz + dz)], [(0, 1, 2)], fuv=[[(0, 0), (1, 0), (0, 1)]])

    # Drop ceiling: grid + tiles, a few missing or pushed up, two troffers.
    grid = k.B("paint_white", tag="grid", lm=LM_TINY)
    tiles = k.B("ceiling_tile", lm=LM_SMALL)
    cx0, cx1, cy0, cy1 = ox0 + 0.02, ox1 - t, oy0 + t, oy1 - 0.02
    zc = OFFICE_CEIL
    x = cx0
    while x <= cx1 + 1e-6:
        grid.box_between((x, cy0, zc), (x, cy1, zc), 0.024, 0.03)
        x += 1.2
    y = cy0
    while y <= cy1 + 1e-6:
        grid.box_between((cx0, y, zc), (cx1, y, zc), 0.024, 0.03)
        y += 0.6
    troffers = [(-8.22, 13.02), (-8.22, 16.02)]
    missing = {(1, 3), (4, 12), (5, 11)}
    pushed = {(3, 7)}
    ix = 0
    x = cx0
    while x + 1.2 <= cx1 + 1e-6:
        iy = 0
        y = cy0
        while y + 0.6 <= cy1 + 1e-6:
            c = (x + 0.6, y + 0.3)
            is_troffer = any(abs(c[0] - tx) < 0.61 and abs(c[1] - ty) < 0.31 for (tx, ty) in troffers)
            if (ix, iy) in missing or is_troffer:
                pass
            elif (ix, iy) in pushed:
                tiles.box((c[0], c[1], zc + 0.12), (1.17, 0.57, 0.015), (0.25, 0.0, 0.1))
            else:
                tiles.box((c[0], c[1], zc + 0.0075), (1.17, 0.57, 0.015))
            y += 0.6
            iy += 1
        x += 1.2
        ix += 1
    k.emit_mat("emit_troffer", (0.86, 0.93, 1.0), 14.0, baked=False)
    k.emit_mat("emit_troffer_off", (0.6, 0.65, 0.7), 0.0, baked=False)
    for i, (tx, ty) in enumerate(troffers):
        k.B("paint_white", tag="troffer", lm=LM_DETAIL).box((tx, ty, zc + 0.06), (1.19, 0.59, 0.1))
        k.B("emit_troffer" if i == 0 else "emit_troffer_off", kind="emissive").box((tx, ty, zc - 0.004), (1.12, 0.52, 0.006))
    k.area_light("office_troffer", (troffers[0][0], troffers[0][1], zc - 0.02), (1.1, 0.5), 42.0, (0.86, 0.93, 1.0), mode="baked")

    # Vinyl floor.
    k.B("plastic_grey", tag="vinyl", lm=LM_BIG).quad((ox0, oy0 + t, 0.004), (ox1 - t, oy0 + t, 0.004), (ox1 - t, oy1, 0.004), (ox0, oy1, 0.004), uv="world")

    # Furniture.
    desk(k, (-9.0, 17.3, 0), math.pi, rng)
    desk(k, (-5.4, 12.9, 0), math.pi / 2, rng, monitor=False)
    chair(k, (-9.1, 16.5, 0), 0.4, rng)
    chair(k, (-6.2, 12.6, 0), -1.9, rng, tipped=True)
    filing_cabinet(k, (-11.65, 12.2, 0), -math.pi / 2, rng)
    filing_cabinet(k, (-11.65, 11.65, 0), -math.pi / 2, rng, open_drawer=2)
    wb = Frame((-11.97, 14.9, 0), -math.pi / 2)
    k.B("paint_white", tag="board", lm=LM_SMALL).box(wb.p(0, -0.01, 1.45), (1.8, 0.015, 1.0), wb.r())
    k.B("galv", lm=LM_TINY).box(wb.p(0, -0.02, 0.94), (1.8, 0.05, 0.02), wb.r())
    for i in range(6):
        P.paper_sheet(k, (-11.94, 14.2 + i * 0.25, 1.3 + (i % 2) * 0.3), 0.0, rng) if False else None
    # Notices pinned on the board (vertical sheets).
    for i in range(5):
        k.B("paper", lm=LM_TINY).box(wb.p(-0.6 + i * 0.28, -0.02, 1.5 + jitter(rng, 0.2)), (0.21, 0.001, 0.29), wb.r(ry=jitter(rng, 0.06)))
    # Boxes of files and papers on the floor.
    for (x, y) in ((-11.4, 17.4), (-10.95, 17.45), (-11.2, 17.35)):
        P.carton(k, (x, y, 0.15 if y < 17.4 else 0.45), (0.42, 0.32, 0.3), jitter(rng, 0.2), rng, tape=False)
    for i in range(14):
        P.paper_sheet(k, (-7.5 + jitter(rng, 2.5), 11.5 + jitter(rng, 2.5), 0.004), rng.random() * math.tau, rng)
    k.spot("target", (-8.5, 13.6, 0))


def desk(k, pos, yaw, rng, monitor=True):
    f = Frame(pos, yaw)
    top = k.B("plywood", tag="desk", lm=LM_PROP)
    top.box(f.p(0, 0, 0.74), (1.5, 0.75, 0.03), f.r())
    leg = k.B("paint_dark", tag="desk", lm=LM_DETAIL)
    for sx in (-0.7, 0.7):
        leg.box(f.p(sx, 0, 0.36), (0.05, 0.65, 0.72), f.r())
    leg.box(f.p(0, 0.3, 0.45), (1.35, 0.02, 0.5), f.r())  # modesty panel
    # Drawer pedestal.
    k.B("paint_grey", tag="desk", lm=LM_DETAIL).box(f.p(0.45, 0, 0.36), (0.4, 0.6, 0.66), f.r())
    if monitor:
        bl = k.B("plastic_black", lm=LM_DETAIL)
        bl.box(f.p(0, 0.15, 0.77), (0.22, 0.18, 0.02), f.r())
        bl.box(f.p(0, 0.17, 0.9), (0.04, 0.03, 0.26), f.r())
        bl.box(f.p(0, 0.14, 1.07), (0.56, 0.035, 0.34), f.r(rx=-0.08))
        bl.box(f.p(-0.05, -0.12, 0.765), (0.44, 0.14, 0.02), f.r(rz=0.05))
        k.B("paint_white", tag="mug", lm=LM_TINY, sharp=60).lathe([(0.0, 0.0), (0.04, 0.0), (0.043, 0.1), (0.038, 0.1), (0.036, 0.02), (0.0, 0.02)],
                                                                  f.m4(0.5, -0.1, 0.755), segments=14)
    for i in range(4):
        P.paper_sheet(k, f.p(jitter(rng, 0.5), jitter(rng, 0.2), 0.756), yaw + jitter(rng, 0.5), rng)
    k.collide_box(f.p(0, 0, 0.38), (1.5, 0.75, 0.76), f.r(), "wood")


def chair(k, pos, yaw, rng, tipped=False):
    if tipped:
        f = Frame(Vector(pos) + Vector((0, 0, 0.3)), yaw, rot=(1.45, 0, 0))
    else:
        f = Frame(pos, yaw)
    bl = k.B("plastic_black", tag="chair", lm=LM_DETAIL, sharp=60)
    for i in range(5):
        t = math.tau * i / 5
        bl.box_between(f.p(0, 0, 0.1), f.p(math.cos(t) * 0.3, math.sin(t) * 0.3, 0.06), 0.04, 0.03)
        bl.rod(f.p(math.cos(t) * 0.3, math.sin(t) * 0.3, 0.035), f.p(math.cos(t) * 0.3 + 0.001, math.sin(t) * 0.3, 0.035), 0.03, sides=8, caps=True) if False else None
        c = f.p(math.cos(t) * 0.3, math.sin(t) * 0.3, 0.03)
        bl.rod(c - f.R @ Vector((0, 0.02, 0)), c + f.R @ Vector((0, 0.02, 0)), 0.028, sides=10, caps=True)
    k.B("galv", lm=LM_TINY).rod(f.p(0, 0, 0.1), f.p(0, 0, 0.42), 0.025, sides=10)
    fab = k.B("plastic_grey", tag="chair", lm=LM_DETAIL)
    fab.box(f.p(0, 0, 0.46), (0.48, 0.46, 0.08), f.r())
    fab.box(f.p(0, 0.25, 0.8), (0.46, 0.07, 0.52), f.r(rx=-0.12))
    bl.box(f.p(0, 0.24, 0.52), (0.06, 0.04, 0.16), f.r())


def filing_cabinet(k, pos, yaw, rng, open_drawer=-1):
    f = Frame(pos, yaw)
    g = k.B("paint_grey", tag="cab", lm=LM_SMALL)
    g.box(f.p(0, 0, 0.66), (0.46, 0.62, 1.32), f.r())
    for i in range(4):
        z = 0.17 + i * 0.32
        out = 0.35 if i == open_drawer else 0.0
        g.box(f.p(0, -0.31 - 0.012 - out / 2, z), (0.43, 0.024 + out, 0.3), f.r())
        k.B("galv", lm=LM_TINY).box(f.p(0, -0.335 - out, z + 0.08), (0.14, 0.02, 0.025), f.r())
        if out:
            for j in range(3):
                k.B("paper", lm=LM_TINY).box(f.p(jitter(rng, 0.1), -0.31 - out * (0.3 + j * 0.25), z + 0.15), (0.36, 0.003, 0.24), f.r(rx=jitter(rng, 0.2)))
    k.collide_box(f.p(0, 0, 0.66), (0.46, 0.62, 1.32), f.r(), "metal")


def build_racks(k, rng):
    fx0, fx1 = 10.4, 11.5
    frames = [-1.0, 1.7, 4.4, 7.1]
    for y in frames:
        P.rack_upright_frame(k, fx0, fx1, y, 3.6, rng)
    for (ya, yb) in zip(frames, frames[1:]):
        for z in (1.55, 3.05):
            P.rack_beam_pair(k, fx0, fx1, ya, yb, z)
        k.collide_box(((fx0 + fx1) / 2, (ya + yb) / 2, 1.8), (fx1 - fx0 + 0.1, yb - ya + 0.1, 3.6), (0, 0, 0), "metal")
    # Column guards on the aisle side.
    for y in frames:
        k.B("paint_yellow", tag="guard", lm=LM_DETAIL).box((fx0 - 0.08, y, 0.2), (0.08, 0.16, 0.4))
    # Loads.
    xm = (fx0 + fx1) / 2
    yaw = math.pi / 2  # pallet length across the rack depth
    bays = list(zip(frames, frames[1:]))
    # Bay 1
    ya, yb = bays[0]
    for i, yy in enumerate((ya + 0.72, yb - 0.72)):
        top = P.pallet(k, (xm, yy, 0.0), yaw, rng)
        P.carton_load(k, (xm, yy, top), yaw, rng, layers=3 if i == 0 else 2)
        top = P.pallet(k, (xm, yy, 1.55), yaw, rng)
        if i == 0:
            P.drum(k, (xm - 0.25, yy - 0.2, top), "paint_blue", rng)
            P.drum(k, (xm + 0.25, yy + 0.2, top), "paint_blue", rng)
        else:
            P.carton_load(k, (xm, yy, top), yaw, rng, layers=2, wrap=False)
        if i == 1:
            top = P.pallet(k, (xm, yy, 3.05), yaw, rng)
            P.carton_load(k, (xm, yy, top), yaw, rng, layers=1, wrap=False)
    # Bay 2
    ya, yb = bays[1]
    P.pallet_stack(k, (xm, ya + 0.72, 0.0), yaw, 6, rng)
    P.tyre_stack(k, (xm - 0.1, yb - 0.75, 0.0), 5, rng)
    top = P.pallet(k, (xm, ya + 0.72, 1.55), yaw, rng)
    P.carton_load(k, (xm, ya + 0.72, top), yaw, rng, layers=2)
    k.spot("prop", (xm, yb - 0.72, 1.55 + 0.14 + 0.01))
    # Bay 3
    ya, yb = bays[2]
    top = P.pallet(k, (xm, ya + 0.72, 0.0), yaw, rng)
    for i in range(4):
        for j in range(2):
            P.sandbag(k, (xm - 0.28 + 0.56 * j, ya + 0.72 - 0.33 + i * 0.22, top + 0.07), math.pi / 2 + jitter(rng, 0.1), rng, size=(0.55, 0.36, 0.12))
    top = P.pallet(k, (xm, yb - 0.72, 1.55), yaw, rng)
    P.carton_load(k, (xm, yb - 0.72, top), yaw, rng, layers=2, box=(0.6, 0.5, 0.45))
    top = P.pallet(k, (xm, ya + 0.72, 3.05), yaw, rng, broken=True)


def build_range(k, rng):
    # Sandbag berm against the north wall.
    xs0, xs1 = -4.2, 4.2
    rows_by_layer = [3] * 4 + [2] * 4 + [1] * 3
    z = 0.075
    for layer, rows in enumerate(rows_by_layer):
        for r in range(rows):
            y = Y1 - 0.2 - r * 0.33
            x = xs0 + (0.3 if layer % 2 else 0.0) + 0.3
            while x < xs1 - 0.2:
                P.sandbag(k, (x + jitter(rng, 0.02), y + jitter(rng, 0.02), z), jitter(rng, 0.06), rng)
                x += 0.6
        z += 0.14
    k.collide_box((0, Y1 - 0.55, 0.6), (xs1 - xs0, 1.1, 1.2), (0, 0, 0), "fabric")
    k.collide_box((0, Y1 - 0.4, 1.35), (xs1 - xs0, 0.8, 0.3), (0, 0, 0), "fabric")
    # Rubber curtain on a pipe above the berm.
    pipe_z = 3.25
    P.pipe_run(k, [(xs0 - 0.3, Y1 - 0.25, pipe_z), (xs1 + 0.3, Y1 - 0.25, pipe_z)], 0.03, "paint_dark")
    for x in (xs0 - 0.2, xs1 + 0.2):
        k.B("paint_dark", lm=LM_TINY).box_between((x, Y1, pipe_z), (x, Y1 - 0.25, pipe_z), 0.04, 0.04)
    x = xs0
    rub = k.B("rubber", tag="curtain", lm=LM_SMALL)
    while x < xs1 - 0.05:
        w = 0.3
        rot = (jitter(rng, 0.03), jitter(rng, 0.02), jitter(rng, 0.08))
        rub.box((x + w / 2, Y1 - 0.25, (pipe_z + 1.3) / 2 + jitter(rng, 0.03)), (w - 0.012, 0.012, pipe_z - 1.3), rot)
        x += w
    # Tyre walls flanking the berm.
    for (x, y) in ((-5.0, 17.4), (-5.0, 16.7), (5.0, 17.4), (5.0, 16.7)):
        P.tyre_stack(k, (x, y, 0), 5 if y > 17 else 3, rng)
    P.tyre(k, (-4.5, 15.9, 0), rng, lying=False, yaw=0.3, tilt=0.15)
    # Target plate stands (dynamic in the game).
    for x in (-3.0, 0.0, 3.0):
        k.spot("plate", (x, 16.0, 0))
    # Range furniture: a folding table with ammo cans near the firing line.
    tf = Frame((-6.5, 0.8, 0), 0.15)
    tb = k.B("plastic_grey", tag="table", lm=LM_PROP)
    tb.box(tf.p(0, 0, 0.73), (1.8, 0.75, 0.04), tf.r())
    for sx in (-0.8, 0.8):
        k.B("galv", lm=LM_TINY).box_between(tf.p(sx, -0.3, 0.02), tf.p(sx, 0.3, 0.71), 0.025, 0.025, up=tuple(tf.R @ Vector((1, 0, 0))))
        k.B("galv", lm=LM_TINY).box_between(tf.p(sx, 0.3, 0.02), tf.p(sx, -0.3, 0.71), 0.025, 0.025, up=tuple(tf.R @ Vector((1, 0, 0))))
    for i in range(3):
        P.ammo_can(k, tf.p(-0.5 + i * 0.35, jitter(rng, 0.1), 0.75), 0.15 + jitter(rng, 0.2), rng)
    k.collide_box(tf.p(0, 0, 0.375), (1.8, 0.75, 0.75), tf.r(), "rubber")


def build_clutter(k, rng):
    # Workbench near the spawn (plinking cans on top).
    wb = P.workbench(k, (3.0, -3.9, 0), 0.0, rng)
    for i, x in enumerate((2.25, 2.65, 3.05, 3.45)):
        k.spot("can", (x, -4.0, wb + 0.062))
    k.spot("prop", (3.8, -3.9, 0.21 + 0.14))
    # Barricades.
    P.barricade(k, (-1.4, -2.3, 0), 0.0, rng, port=(0.0, 1.35, 0.35, 0.3))
    P.barricade(k, (5.2, 5.6, 0), 0.55, rng, width=1.22, height=2.0)
    P.barricade(k, (-3.6, 6.4, 0), -0.35, rng, port=(0.25, 0.75, 0.3, 0.25))
    # Pallet stacks and loads.
    top = P.pallet_stack(k, (-3.2, 1.8, 0), 0.12, 6, rng)
    k.spot("prop", (-3.2, 1.8, top + 0.141))
    P.pallet_stack(k, (-8.6, -6.1, 0), -0.05, 4, rng)
    top = P.pallet(k, (-7.1, -6.3, 0), 0.03, rng)
    P.carton_load(k, (-7.1, -6.3, top), 0.03, rng, layers=3)
    top = P.pallet_stack(k, (1.6, 9.4, 0), 0.9, 3, rng)
    P.carton_load(k, (1.6, 9.4, top), 0.9, rng, layers=2, wrap=False, box=(0.6, 0.5, 0.4))
    top = P.pallet(k, (7.2, 13.4, 0), -0.4, rng)
    for i in range(3):
        for j in range(2):
            for l in range(2):
                P.sandbag(k, (7.2 + (i - 1) * 0.36 * math.cos(-0.4) - (j - 0.5) * 0.55 * math.sin(-0.4),
                              13.4 + (i - 1) * 0.36 * math.sin(-0.4) + (j - 0.5) * 0.55 * math.cos(-0.4), top + 0.06 + l * 0.11),
                          -0.4 + math.pi / 2 + jitter(rng, 0.1), rng, size=(0.55, 0.34, 0.11))
    k.collide_box((7.2, 13.4, 0.3), (1.22, 1.1, 0.6), (0, 0, -0.4), "fabric")
    # Drums.
    P.drum(k, (7.9, -5.8, 0), "paint_blue", rng)
    P.drum(k, (8.55, -5.55, 0), "paint_red", rng)
    P.drum(k, (8.2, -5.0, 0), "rack_orange", rng)
    P.drum(k, (6.9, -4.7, 0), "paint_blue", rng, lying=True, yaw=0.5)
    P.drum(k, (-10.9, 5.9, 0), "paint_green", rng)
    P.drum(k, (-10.95, 6.55, 0), "paint_dark", rng)
    k.spot("prop", (7.3, -6.6, 0.14))
    # Cones and bollards at the roller door.
    P.traffic_cone(k, (4.4, -6.6, 0), rng)
    P.traffic_cone(k, (5.6, -6.9, 0), rng, yaw=0.4)
    P.traffic_cone(k, (6.5, -6.3, 0), rng, knocked=True, yaw=2.2)
    P.traffic_cone(k, (-4.8, 15.3, 0), rng)
    for x in (ROLLER[0] - 0.45, ROLLER[1] + 0.45):
        P.bollard(k, (x, Y0 + 0.5, 0), rng)
    # Pallet jack.
    pallet_jack(k, (8.9, -1.9, 0), 2.2, rng)
    # Electrical panels + conduit on the west wall.
    P.electrical_panel(k, (X0 + 0.1, -4.4, 1.5), -math.pi / 2, rng)
    P.electrical_panel(k, (X0 + 0.08, -3.55, 1.65), -math.pi / 2, rng, size=(0.4, 0.15, 0.5))
    P.electrical_panel(k, (X0 + 0.08, -3.0, 1.5), -math.pi / 2, rng, size=(0.25, 0.14, 0.35))
    for (y, z) in ((-4.55, 1.95), (-4.25, 1.95), (-3.55, 1.9), (-3.0, 1.68)):
        P.pipe_run(k, [(X0 + 0.06, y, z), (X0 + 0.06, y, 5.8)], 0.012, "galv", sides=8, lm=LM_TINY)
    P.pipe_run(k, [(X0 + 0.06, -4.55, 5.8), (X0 + 0.06, -2.0, 5.8), (X1 - 0.4, -2.0, 5.8)], 0.012, "galv", sides=8, lm=LM_TINY)
    P.pipe_run(k, [(X0 + 0.06, -4.25, 5.82), (X0 + 0.06, 5.0, 5.82), (X1 - 0.4, 5.0, 5.82)], 0.012, "galv", sides=8, lm=LM_TINY)
    P.pipe_run(k, [(X0 + 0.06, -3.55, 5.84), (X0 + 0.06, 12.0, 5.84), (X1 - 0.4, 12.0, 5.84)], 0.012, "galv", sides=8, lm=LM_TINY)
    # Fire extinguisher on an interior column.
    P.fire_extinguisher(k, (-6.0, 4.78, 0.95), math.pi, rng)
    # Floor markings: rack aisle, walkway, firing line, hatched door zone.
    for (a, b_) in (((9.95, -1.3), (9.95, 7.4)), ((-4.25, -7.8), (-4.25, 7.8)), ((-4.0, 0.0), (4.0, 0.0)), ((-4.3, 15.1), (4.3, 15.1))):
        P.floor_line(k, (*a, 0), (*b_, 0))
    for i in range(9):
        x = ROLLER[0] + 0.1 + i * 0.45
        P.floor_line(k, (x, Y0 + 0.1, 0), (x + 0.5, Y0 + 1.0, 0), w=0.08)
    # Scattered paper and a floor drain.
    for i in range(10):
        P.paper_sheet(k, (jitter(rng, 8), 4 + jitter(rng, 10), 0.0), rng.random() * math.tau, rng)
    k.B("paint_dark", tag="drain", lm=LM_TINY).box((0.4, 6.2, 0.002), (0.45, 0.45, 0.004))
    # Hanging chain hoist.
    P.chain(k, (2.25, 9.6, DECK_Z - 0.45), (2.25, 9.6, 2.7), pitch=0.04)
    hb = k.B("paint_yellow", tag="hoist", lm=LM_DETAIL)
    hb.box((2.25, 9.6, DECK_Z - 0.62), (0.28, 0.22, 0.3))
    hb.box((2.25, 9.6, 2.62), (0.07, 0.05, 0.14))
    k.B("steel", lm=LM_TINY).lathe([(0.03, 0.0), (0.045, -0.04), (0.03, -0.1), (0.0, -0.11)], Matrix.Translation((2.25, 9.6, 2.56)), segments=10)
    # Loose cables sagging between joists.
    for (p0, p1, sag) in (((-3.75, 7.3, DECK_Z - 0.45), (-0.75, 7.9, DECK_Z - 0.45), 0.7), ((6.75, -5.5, DECK_Z - 0.45), (9.75, -4.8, DECK_Z - 0.45), 0.5)):
        pts = []
        for i in range(13):
            t = i / 12
            pts.append(Vector(p0).lerp(Vector(p1), t) - Vector((0, 0, sag * 4 * t * (1 - t))))
        P.pipe_run(k, pts, 0.008, "plastic_black", sides=6, lm=LM_TINY, elbow=False)


def pallet_jack(k, pos, yaw, rng):
    f = Frame(pos, yaw)
    r = k.B("paint_red", tag="jack", lm=LM_DETAIL)
    for sy in (-0.2, 0.2):
        r.box(f.p(0.55, sy, 0.05), (1.15, 0.16, 0.06), f.r())
        k.B("plastic_black", tag="jack", lm=LM_TINY).rod(f.p(1.05, sy - 0.03, 0.035), f.p(1.05, sy + 0.03, 0.035), 0.035, sides=10, caps=True)
    r.box(f.p(-0.08, 0, 0.12), (0.2, 0.55, 0.2), f.r())
    r.rod(f.p(-0.12, 0, 0.2), f.p(-0.12, 0, 0.45), 0.05, sides=12, caps=True)
    r.rod(f.p(-0.12, 0, 0.45), f.p(-0.55, 0, 1.15), 0.016, sides=8)
    r.box_between(f.p(-0.55, -0.12, 1.15), f.p(-0.55, 0.12, 1.15), 0.03, 0.03)
    k.B("plastic_black", tag="jack", lm=LM_TINY).rod(f.p(-0.2, -0.08, 0.06), f.p(-0.2, 0.08, 0.06), 0.06, sides=12, caps=True)
    k.collide_box(f.p(0.4, 0, 0.1), (1.4, 0.55, 0.2), f.r(), "metal")


def build_services(k, rng):
    # Sprinkler mains and branch lines (red), riser with valve on the west wall.
    red = "paint_red"
    zb = 5.75
    P.pipe_run(k, [(X0 + 0.25, 5.6, 0.1), (X0 + 0.25, 5.6, zb), (X1 - 0.4, 5.6, zb)], 0.055, red, sides=14, lm=LM_DETAIL)
    for x in (-9.0, -3.0, 3.0, 9.0):
        P.pipe_run(k, [(x, Y0 + 0.4, zb - 0.02), (x, Y1 - 0.4, zb - 0.02)], 0.028, red, sides=10, lm=LM_TINY, elbow=False)
        y = Y0 + 1.5
        while y < Y1 - 0.5:
            if not (OFFICE[0] < x < OFFICE[1] and y > OFFICE[2]):
                h = k.B("galv", tag="spr", lm=LM_TINY, sharp=60)
                h.rod((x, y, zb - 0.05), (x, y, zb - 0.12), 0.008, sides=6)
                h.lathe([(0.0, 0.0), (0.022, 0.0), (0.022, 0.004), (0.0, 0.004)], Matrix.Translation((x, y, zb - 0.125)), segments=10)
            # Hanger rod up to the joists.
            k.B("steel", tag="joist").rod((x, y + 0.5, zb + 0.03), (x, y + 0.5, DECK_Z - 0.45), 0.005, sides=5, lm=LM_TINY)
            y += 3.0
    # Riser valve + gauge.
    valve = Vector((X0 + 0.25, 5.6, 1.2))
    k.B(red, lm=LM_TINY, sharp=60).lathe([(0.0, -0.1), (0.075, -0.08), (0.075, 0.08), (0.0, 0.1)], Matrix.Translation(valve), segments=14)
    k.B("paint_dark", lm=LM_TINY, sharp=60).lathe([(0.1, -0.012), (0.12, 0.0), (0.1, 0.012), (0.085, 0.0)],
                                                  Matrix.Translation(valve + Vector((0.18, 0, 0))) @ Matrix.Rotation(math.pi / 2, 4, "Y"), segments=16, closed=True)
    k.B("galv", lm=LM_TINY).rod(valve + Vector((0.07, 0, 0)), valve + Vector((0.18, 0, 0)), 0.012, sides=8)
    k.B("paint_white", tag="gauge", lm=LM_TINY, sharp=60).lathe([(0.0, 0.0), (0.04, 0.0), (0.04, 0.02), (0.0, 0.02)],
                                                               Matrix.Translation(valve + Vector((0.0, -0.1, 0.35))) @ Matrix.Rotation(math.pi / 2, 4, "X"), segments=14)


def build_lights(k, rng):
    """Hanging lamps (mixed), work light (mixed), moon, street light, sky (baked)."""
    # Mixed lamp stock, like a real building that has been relamped piecemeal: warm metal halide,
    # one tired cool-white (the flickering one) and one sodium retrofit.
    warm, cool, sodium = (1.0, 0.8, 0.58), (0.86, 0.94, 1.0), (1.0, 0.66, 0.36)
    k.emit_mat("emit_lamp", (1.0, 0.82, 0.6), 30.0, baked=False)
    k.emit_mat("emit_lamp_cool", (0.88, 0.95, 1.0), 30.0, baked=False)
    k.emit_mat("emit_lamp_sodium", (1.0, 0.68, 0.38), 30.0, baked=False)
    k.emit_mat("emit_worklight", (1.0, 0.9, 0.75), 45.0, baked=False)
    lamps = [
        ("lamp_1", (0.75, -3.2, 4.85), 58.0, False, warm, "emit_lamp"),
        ("lamp_2", (-2.25, 4.0, 4.85), 54.0, True, cool, "emit_lamp_cool"),
        ("lamp_3", (0.75, 11.0, 4.85), 62.0, False, warm, "emit_lamp"),
        ("lamp_4", (-6.75, 0.0, 4.85), 50.0, False, sodium, "emit_lamp_sodium"),
    ]
    for (name, pos, cd, flick, color, bulb) in lamps:
        P.lamp_dome(k, pos, DECK_Z - 0.45, rng, bulb=bulb)
        x, y, z = pos
        k.spot_light(name, (x, y, z - 0.03), (x, y, 0.0), cd, color=color, angle_deg=62, penumbra=0.62,
                     radius=0.06, mode="mixed", shadow=True, flicker=flick, volumetric=1.0)
    # Dead fixture over the rack aisle.
    P.lamp_dome(k, (8.25, 3.0, 4.85), DECK_Z - 0.45, rng)
    # Work light on the range.
    head, aim = P.work_light(k, (3.4, 12.6, 0), (0.0, 17.2, 0.9), rng)
    k.spot_light("worklight", tuple(head), tuple(Vector(head) + aim * 5.0), 170.0, color=(1.0, 0.86, 0.68), angle_deg=34,
                 penumbra=0.45, radius=0.1, mode="mixed", shadow=True, volumetric=0.6)
    # Moonlight through the east windows and skylights.
    k.sun("moon", tuple(MOON_DIR), 0.42, (0.58, 0.72, 1.0), angle_deg=0.55, mode="baked")
    # Sodium street light outside the north windows.
    k.point_light("streetlight", (2.5, 23.0, 8.0), 140.0, (1.0, 0.52, 0.16), radius=0.25, mode="baked")
    k.spot("spawn", (0.0, -6.5, 0.0), 0.0)
    for (x, y) in ((-2.5, 11.0), (3.5, 13.3), (8.2, 3.6), (-6.2, -1.6), (4.3, 7.0)):
        k.spot("target", (x, y, 0.0))


def build(tex_dir):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    k = lk.Kit(tex_dir)
    rng = random.Random(4)
    build_shell(k, rng)
    build_structure(k, rng)
    build_openings(k, rng)
    build_office(k, rng)
    build_racks(k, rng)
    build_range(k, rng)
    build_clutter(k, rng)
    build_services(k, rng)
    build_lights(k, rng)
    k.build_batches()
    return k, list(k.static)


# ------------------------------------------------------------------------------------
# Rendering / baking
# ------------------------------------------------------------------------------------

def setup_world(sc, strength=1.0):
    world = bpy.data.worlds.new("night")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.010, 0.014, 0.026, 1)
    bg.inputs["Strength"].default_value = strength
    return bg


def setup_cycles(sc, samples, threads=0):
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.max_bounces = 8
    sc.cycles.diffuse_bounces = 4
    sc.cycles.glossy_bounces = 2
    sc.cycles.transmission_bounces = 4
    sc.cycles.transparent_max_bounces = 8
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.sample_clamp_indirect = 8.0
    sc.cycles.blur_glossy = 1.0
    if threads:
        sc.render.threads_mode = "FIXED"
        sc.render.threads = threads


def set_light_mode(k, mode):
    """mode: 'all', 'mixed' (only real-time lamps), 'baked' (only baked lights/emitters/sky)."""
    for L in k.lights:
        on = mode == "all" or L["mode"] == mode
        L["obj"].hide_render = not on
    for name, m in k.mats.items():
        if name.startswith("emit_"):
            em = m.node_tree.nodes["Emission"]
            base = m["strength"]
            if mode == "all":
                em.inputs["Strength"].default_value = base
            elif mode == "baked":
                em.inputs["Strength"].default_value = base if m["baked"] else 0.0
            else:
                em.inputs["Strength"].default_value = 0.0
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.0 if mode == "mixed" else 1.0


def add_camera(name, loc, rot, lens=18.0):
    cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = rot
    cam.data.lens = lens
    cam.data.clip_start = 0.05
    return cam


PREVIEW_CAMS = {
    "spawn": ((0.0, -6.5, 1.55), (math.radians(88), 0, 0)),
    "hall": ((-9.5, -6.8, 2.2), (math.radians(84), 0, math.radians(-38))),
    "range": ((0.5, 5.0, 1.6), (math.radians(88), 0, math.radians(-8))),
    "office": ((-4.0, 6.5, 1.6), (math.radians(86), 0, math.radians(35))),
    "racks": ((7.0, -4.5, 1.6), (math.radians(90), 0, math.radians(-18))),
    "up": ((0.0, 2.0, 1.6), (math.radians(150), 0, 0)),
}


def render_preview(k, path, cam_name, samples, width, threads):
    sc = bpy.context.scene
    setup_cycles(sc, samples, threads)
    sc.cycles.use_denoising = True
    loc, rot = PREVIEW_CAMS[cam_name]
    cam = add_camera("preview", loc, rot, lens=16.0)
    sc.camera = cam
    sc.render.resolution_x = width
    sc.render.resolution_y = width * 9 // 16
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Base Contrast"
    sc.view_settings.exposure = 1.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = path
    set_light_mode(k, "all")
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"preview {path} in {time.time() - t:.1f}s")


def prepare_bake_materials(k, on):
    """For baking: flatten normal maps and make metals diffuse so their lightmap texels are valid."""
    for name, m in k.mats.items():
        if name not in lk.MATERIALS:
            continue
        nt = m.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        nm = nt.nodes.get("NormalMap")
        if on:
            m["_metal"] = bsdf.inputs["Metallic"].default_value
            bsdf.inputs["Metallic"].default_value = 0.0
            if nm:
                nm.inputs["Strength"].default_value = 0.0
        else:
            bsdf.inputs["Metallic"].default_value = m.get("_metal", 0.0)
            if nm:
                nm.inputs["Strength"].default_value = 1.0


def attach_bake_image(objs, img):
    nodes = []
    for o in objs:
        for slot in o.material_slots:
            m = slot.material
            nt = m.node_tree
            n = nt.nodes.get("BakeTarget") or nt.nodes.new("ShaderNodeTexImage")
            n.name = "BakeTarget"
            n.image = img
            n.interpolation = "Linear"
            nt.nodes.active = n
            nodes.append(n)
    return nodes


def select_only(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


def new_image(name, size, float_buffer=True):
    img = bpy.data.images.new(name, size, size, alpha=True, float_buffer=float_buffer)
    img.colorspace_settings.name = "Non-Color" if not float_buffer else "Linear Rec.709"
    img.generated_color = (0, 0, 0, 0)
    return img


def image_array(img):
    w, h = img.size
    arr = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    return arr.reshape(h, w, 4)


def bake(objs, img, kind, pass_filter=None, margin=0):
    attach_bake_image(objs, img)
    select_only(objs)
    kw = dict(type=kind, uv_layer="LM", margin=margin, margin_type="EXTEND", use_clear=True, target="IMAGE_TEXTURES", save_mode="INTERNAL")
    if pass_filter is not None:
        kw["pass_filter"] = pass_filter
    if kind == "NORMAL":
        kw["normal_space"] = "OBJECT"
    t = time.time()
    bpy.ops.object.bake(**kw)
    print(f"  bake {kind} {pass_filter or ''} {img.size[0]}px in {time.time() - t:.1f}s", flush=True)
    return image_array(img)


def denoise(rgb, normal=None):
    """OIDN through the compositor on an empty helper scene."""
    h, w, _ = rgb.shape
    src = bpy.data.images.new("dn_src", w, h, alpha=True, float_buffer=True)
    px = np.concatenate([rgb, np.ones((h, w, 1), np.float32)], -1)
    src.pixels.foreach_set(px.ravel())
    main = bpy.context.window.scene if bpy.context.window else bpy.context.scene
    sc = bpy.data.scenes.new("denoise")
    ng = bpy.data.node_groups.new("dn", "CompositorNodeTree")
    sc.compositing_node_group = ng
    im = ng.nodes.new("CompositorNodeImage")
    im.image = src
    den = ng.nodes.new("CompositorNodeDenoise")
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    out = ng.nodes.new("NodeGroupOutput")
    ng.links.new(im.outputs["Image"], den.inputs["Image"])
    if normal is not None:
        nimg = bpy.data.images.new("dn_nrm", w, h, alpha=True, float_buffer=True)
        npx = np.concatenate([normal, np.ones((h, w, 1), np.float32)], -1)
        nimg.pixels.foreach_set(npx.ravel())
        nn = ng.nodes.new("CompositorNodeImage")
        nn.image = nimg
        ng.links.new(nn.outputs["Image"], den.inputs["Normal"])
    ng.links.new(den.outputs["Image"], out.inputs[0])
    cam = bpy.data.objects.new("dn_cam", bpy.data.cameras.new("dn_cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = w
    sc.render.resolution_y = h
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.view_settings.view_transform = "Standard"
    path = os.path.join(bpy.app.tempdir or "/tmp", f"dn_{w}.exr")
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True, scene=sc.name)
    res = bpy.data.images.load(path)
    arr = image_array(res)[..., :3].copy()
    bpy.data.images.remove(res)
    bpy.data.images.remove(src)
    bpy.data.scenes.remove(sc)
    return arr


def dilate(rgb, mask, iterations=12):
    """Fill texels outside `mask` from covered neighbours (keeps bilinear filtering clean)."""
    out = rgb.copy()
    m = mask.astype(np.float32)
    for _ in range(iterations):
        acc = np.zeros_like(out)
        cnt = np.zeros(m.shape, np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
            sm = np.roll(np.roll(m, dy, 0), dx, 1)
            acc += np.roll(np.roll(out * m[..., None], dy, 0), dx, 1)
            cnt += sm
        fill = (m == 0) & (cnt > 0)
        out[fill] = acc[fill] / cnt[fill][:, None]
        m = np.where(fill, 1.0, m)
    return out


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save_jpeg(arr01, path, quality=95):
    from PIL import Image
    img = Image.fromarray((np.clip(arr01, 0, 1) * 255 + 0.5).astype(np.uint8)[::-1], "RGB")
    img.save(path, quality=quality, subsampling=0, optimize=True)


def mask_material():
    """Emission shader writing R wetness (puddles), G grime (cavity dirt), B specular occlusion."""
    m = bpy.data.materials.new("bake_mask")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    N = nt.nodes.new
    L = nt.links.new
    out = N("ShaderNodeOutputMaterial")
    em = N("ShaderNodeEmission")
    L(em.outputs[0], out.inputs["Surface"])
    geo = N("ShaderNodeNewGeometry")
    sep = N("ShaderNodeSeparateXYZ")
    L(geo.outputs["Position"], sep.inputs[0])
    nsep = N("ShaderNodeSeparateXYZ")
    L(geo.outputs["Normal"], nsep.inputs[0])

    def math_(op, a, b=None, clamp=False):
        n = N("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L(v, n.inputs[i])
        return n.outputs[0]

    def smooth(v, a, b):
        n = N("ShaderNodeMapRange")
        n.interpolation_type = "SMOOTHSTEP"
        n.inputs["From Min"].default_value = a
        n.inputs["From Max"].default_value = b
        L(v, n.inputs["Value"])
        return n.outputs[0]

    def noise(scale, detail=4.0, rough=0.55, w=0.0):
        n = N("ShaderNodeTexNoise")
        n.noise_dimensions = "4D"
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["W"].default_value = w
        L(geo.outputs["Position"], n.inputs["Vector"])
        return n.outputs["Fac"]

    # --- wetness: explicit puddles with noisy edges, only on upward floor surfaces --------
    edge = noise(0.9, 5.0, 0.6, 1.3)
    puddles = [(-3.0, 1.2, 1.6), (5.6, -6.2, 1.3), (-0.8, 8.6, 1.1), (7.6, 8.6, 1.4), (1.8, 14.6, 1.0),
               (-7.2, -3.8, 1.2), (4.3, 3.2, 0.7), (-9.0, 6.5, 0.9), (9.2, -3.0, 0.8)]
    wet = None
    for (px, py, r) in puddles:
        dx = math_("SUBTRACT", sep.outputs["X"], px)
        dy = math_("SUBTRACT", sep.outputs["Y"], py)
        d = math_("SQRT", math_("ADD", math_("MULTIPLY", dx, dx), math_("MULTIPLY", dy, dy)))
        d = math_("ADD", math_("DIVIDE", d, r), math_("MULTIPLY", math_("SUBTRACT", edge, 0.5), 1.4))
        w = math_("SUBTRACT", 1.0, smooth(d, 0.72, 0.9))
        wet = w if wet is None else math_("MAXIMUM", wet, w)
    # Sparse damp patches everywhere.
    damp = smooth(noise(0.45, 3.0, 0.5, 7.0), 0.6, 0.68)
    wet = math_("MAXIMUM", wet, math_("MULTIPLY", damp, 0.6))
    floor_only = math_("MULTIPLY", smooth(nsep.outputs["Z"], 0.85, 0.95), math_("SUBTRACT", 1.0, smooth(sep.outputs["Z"], 0.02, 0.05)))
    wet = math_("MULTIPLY", wet, floor_only)

    # --- grime: cavity from short-range AO + low-height dirt + noise -------------------------
    ao_s = N("ShaderNodeAmbientOcclusion")
    ao_s.samples = 12
    ao_s.inputs["Distance"].default_value = 0.35
    cav = math_("SUBTRACT", 1.0, ao_s.outputs["AO"])
    low = math_("SUBTRACT", 1.0, smooth(sep.outputs["Z"], 0.0, 0.9))
    dirt_n = smooth(noise(1.3, 5.0, 0.6, 3.1), 0.35, 0.75)
    grime = math_("ADD", math_("MULTIPLY", math_("POWER", cav, 0.8), 1.2), math_("MULTIPLY", math_("MULTIPLY", low, dirt_n), 0.5), clamp=True)
    grime = math_("MAXIMUM", grime, math_("MULTIPLY", dirt_n, 0.25))
    # Rust/leak streaks below the east windows on the block wall.
    streak = smooth(N("ShaderNodeTexNoise").outputs["Fac"], 0.0, 1.0)
    streak_n = N("ShaderNodeTexNoise")
    streak_n.noise_dimensions = "3D"
    streak_n.inputs["Scale"].default_value = 3.0
    streak_n.inputs["Detail"].default_value = 3.0
    mapn = N("ShaderNodeMapping")
    mapn.inputs["Scale"].default_value = (1.0, 5.0, 0.08)
    L(geo.outputs["Position"], mapn.inputs["Vector"])
    L(mapn.outputs[0], streak_n.inputs["Vector"])
    streaks = smooth(streak_n.outputs["Fac"], 0.55, 0.68)
    east = smooth(sep.outputs["X"], 11.9, 11.99)
    streaks = math_("MULTIPLY", math_("MULTIPLY", streaks, east), math_("SUBTRACT", 1.0, smooth(sep.outputs["Z"], 2.0, 2.4)))
    grime = math_("MAXIMUM", grime, math_("MULTIPLY", streaks, 0.8))

    # --- specular occlusion: long-range AO ---------------------------------------------------
    ao_l = N("ShaderNodeAmbientOcclusion")
    ao_l.samples = 12
    ao_l.inputs["Distance"].default_value = 1.6
    spec = math_("POWER", ao_l.outputs["AO"], 0.7)

    comb = N("ShaderNodeCombineXYZ")
    L(wet, comb.inputs[0])
    L(grime, comb.inputs[1])
    L(spec, comb.inputs[2])
    L(comb.outputs[0], em.inputs["Color"])
    img_node = N("ShaderNodeTexImage")
    img_node.name = "BakeTarget"
    return m


def bake_vertex(vobjs, pass_filter):
    """Bake diffuse light into the 'baked' corner colour attribute of every vertex-lit mesh."""
    select_only(vobjs)
    t = time.time()
    bpy.ops.object.bake(type="DIFFUSE", pass_filter=pass_filter, target="VERTEX_COLORS", use_clear=True)
    res = []
    for o in vobjs:
        a = o.data.color_attributes["baked"]
        arr = np.empty(len(a.data) * 4, np.float32)
        a.data.foreach_get("color", arr)
        res.append(arr.reshape(-1, 4)[:, :3].copy())
    print(f"  bake vertex {pass_filter} {sum(len(r) for r in res)} corners in {time.time() - t:.1f}s", flush=True)
    return res


def bake_all(k, objs, vobjs, args, out_dir):
    sc = bpy.context.scene
    setup_cycles(sc, args.samples, args.threads)
    size = args.lm_size
    t0 = time.time()
    info = lk.lightmap_uvs(objs, resolution=size, padding=3)
    print("lightmap atlas", info, f"{time.time() - t0:.1f}s", flush=True)
    for o in objs:
        o.data.uv_layers.active = o.data.uv_layers["LM"]
    for o in vobjs:
        me = o.data
        attr = me.color_attributes.new("baked", "FLOAT_COLOR", "CORNER")
        me.color_attributes.active_color = attr
        me.color_attributes.render_color_index = me.color_attributes.active_color_index

    # Coverage + guide normals (no margin so uncovered texels stay empty).
    sc.cycles.samples = 1
    nimg = new_image("lm_normal", size)
    nrm = bake(objs, nimg, "NORMAL", margin=0)
    covered = nrm[..., 3] > 0.5
    guide = np.where(covered[..., None], nrm[..., :3] * 2.0 - 1.0, 0.0).astype(np.float32)

    prepare_bake_materials(k, True)
    # A: indirect light of the real-time lamps.
    set_light_mode(k, "mixed")
    sc.cycles.samples = args.samples
    img_a = new_image("lm_a", size)
    a = bake(objs, img_a, "DIFFUSE", {"INDIRECT"}, margin=0)[..., :3]
    sc.cycles.samples = args.samples * 2
    va = bake_vertex(vobjs, {"INDIRECT"})
    # B: everything from the baked-only lights.
    set_light_mode(k, "baked")
    sc.cycles.samples = args.samples
    img_b = new_image("lm_b", size)
    b = bake(objs, img_b, "DIFFUSE", {"DIRECT", "INDIRECT"}, margin=0)[..., :3]
    sc.cycles.samples = args.samples * 2
    vb = bake_vertex(vobjs, {"DIRECT", "INDIRECT"})
    prepare_bake_materials(k, False)

    lm = a + b
    t = time.time()
    lm = denoise(lm * covered[..., None], guide)
    print(f"  denoise {time.time() - t:.1f}s", flush=True)
    lm = dilate(np.maximum(lm, 0.0), covered, iterations=16)
    vals = lm[covered].max(axis=1)
    peak = float(np.percentile(vals, 99.8))
    scale = 0.96 / max(peak, 1e-6)
    save_jpeg(linear_to_srgb(lm * scale), os.path.join(out_dir, "lightmap.jpg"), quality=94)
    print(f"  lightmap peak {peak:.4f} -> scale {scale:.4f}", flush=True)
    # Vertex light uses the same scale, stored as sqrt() in COLOR_0 so 8-bit quantisation keeps
    # dark values (the game squares it back).
    for o, ca, cb in zip(vobjs, va, vb):
        rgb = np.sqrt(np.clip((ca + cb) * scale, 0.0, 1.0))
        rgba = np.concatenate([rgb, np.ones((len(rgb), 1), np.float32)], 1)
        o.data.color_attributes["baked"].data.foreach_set("color", rgba.astype(np.float32).ravel())

    # Mask (wetness/grime/specular occlusion) through an emission bake.
    mm = mask_material()
    saved = [[s.material for s in o.material_slots] for o in objs]
    for o in objs:
        for s in o.material_slots:
            s.material = mm
    sc.cycles.samples = max(16, args.samples // 8)
    img_m = new_image("lm_mask", size)
    mask = bake(objs, img_m, "EMIT", margin=0)[..., :3]
    for o, mats in zip(objs, saved):
        for s, m in zip(o.material_slots, mats):
            s.material = m
    mask = dilate(np.clip(mask, 0, 1), covered, iterations=16)
    save_jpeg(mask, os.path.join(out_dir, "mask.jpg"), quality=92)
    return dict(scale=round(scale, 6), size=size, texels_per_unit=round(float(info["texels_per_unit"]) * size / max(info["width"], 1), 3),
                charts=info["charts"])


def render_probe(k, args, out_dir):
    sc = bpy.context.scene
    setup_cycles(sc, args.probe_samples, args.threads)
    sc.cycles.use_denoising = True
    set_light_mode(k, "all")
    cam = add_camera("probe", PROBE_POS, (math.radians(90), 0, math.radians(-90)))
    cam.data.type = "PANO"
    cam.data.panorama_type = "EQUIRECTANGULAR"
    sc.camera = cam
    sc.render.resolution_x = args.probe_size
    sc.render.resolution_y = args.probe_size // 2
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.render.image_settings.file_format = "HDR"
    path = os.path.join(out_dir, "probe.hdr")
    sc.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"  probe {time.time() - t:.1f}s", flush=True)


def export(k, objs, out_dir, lm_info):
    select_only(objs + k.glass + k.emissive)
    for o in objs + k.glass + k.emissive:
        for name in ("lm_d", "lm_c"):
            if name in o.data.attributes:
                o.data.attributes.remove(o.data.attributes[name])
        if "UVMap" in o.data.uv_layers:
            o.data.uv_layers["UVMap"].active_render = True
    path = os.path.join(out_dir, "level.glb")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_apply=True, export_yup=True,
                              export_texcoords=True, export_normals=True, export_tangents=False, export_materials="EXPORT",
                              export_image_format="NONE", export_cameras=False, export_lights=False, export_extras=False,
                              export_vertex_color="ACTIVE", export_all_vertex_colors=False)
    used = sorted({o.material_slots[0].material.name for o in objs})
    mats = {}
    for name in used:
        spec = lk.MATERIALS[name]
        mats[name] = dict(tex=spec["tex"], tile=spec["tile"], surface=spec["surface"], metal=spec.get("metal", 0.0),
                          tint=list(spec.get("tint", (1.0, 1.0, 1.0))), rough=spec.get("rough", 1.0))
    lights = []
    for L in k.lights:
        d = {kk: v for kk, v in L.items() if kk != "obj"}
        lights.append(d)
    data = dict(
        version=1,
        lightmap=dict(file="lightmap.jpg", **lm_info) if lm_info else None,
        mask=dict(file="mask.jpg") if lm_info else None,
        probe=dict(file="probe.hdr", pos=lk.b2t(PROBE_POS), boxMin=[X0 - 0.2, 0.0, -(Y1 + 0.2)], boxMax=[X1 + 0.2, DECK_Z, -(Y0 - 0.2)]),
        materials=mats,
        glass=lk.GLASS,
        emissive=lk.EMISSIVE,
        colliders=k.colliders,
        lights=lights,
        spots=k.spots,
        bounds=dict(min=[X0, 0.0, -Y1], max=[X1, DECK_Z, -Y0]),
        moon=dict(dir=lk.b2t(MOON_DIR), color=[0.58, 0.72, 1.0]),
        windows=[dict(center=lk.b2t((X1 + 0.2, yc, (WIN_Z0 + WIN_Z1) / 2)), size=[WIN_W, WIN_Z1 - WIN_Z0], normal=[-1, 0, 0]) for yc in EAST_WINDOWS]
        + [dict(center=lk.b2t((xc, Y1 + 0.2, (WIN_Z0 + WIN_Z1) / 2)), size=[WIN_W, WIN_Z1 - WIN_Z0], normal=[0, 0, 1]) for xc in NORTH_WINDOWS],
        skylights=[dict(center=lk.b2t((x, y, DECK_Z + 0.3)), size=[SKY_W, SKY_L]) for (x, y) in SKYLIGHTS],
    )
    with open(os.path.join(out_dir, "level.json"), "w") as f:
        json.dump(data, f, indent=1)
    print("exported", path, os.path.getsize(path) // 1024, "KB")


def stats(objs):
    tris = 0
    for o in objs:
        o.data.calc_loop_triangles()
        tris += len(o.data.loop_triangles)
    return tris


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    k, objs = build(args.tex)
    setup_world(bpy.context.scene)
    print(f"built {len(objs)} static meshes, {len(k.glass)} glass, {len(k.emissive)} emissive in {time.time() - t0:.1f}s", flush=True)
    if args.preview:
        render_preview(k, args.preview, args.preview_cam, args.preview_samples, args.preview_width, args.threads)
        return
    lk.apply_modifiers(objs + k.glass + k.emissive + k.occluders)
    lmobjs = [o for o in objs if o["lit"] == "lm"]
    vobjs = [o for o in objs if o["lit"] == "vtx"]
    print("triangles:", stats(lmobjs), "lightmapped,", stats(vobjs), "vertex-lit,", stats(k.glass + k.emissive), "other", flush=True)
    lm_info = None
    if not args.no_bake:
        lm_info = bake_all(k, lmobjs, vobjs, args, args.out)
        render_probe(k, args, args.out)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
    export(k, lmobjs + vobjs, args.out, lm_info)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
