"""Hard-surface modelling helpers for the weapon/hands/prop scripts.

Everything builds plain meshes with bmesh (no operators), then relies on a
Bevel modifier (angle-limited, hardened normals) + Weighted Normal for the
soft machined edges that sell a close-up first-person model.

Profiles are 2D point lists; `extrude_yz` lays one in the Blender YZ plane
(side view: +Y forward, +Z up) and extrudes it across X, which is how most
parts of a pistol read: slide, frame, trigger guard, trigger, sights.
"""
from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Matrix, Vector

TAU = math.pi * 2


# ----------------------------------------------------------------------------- 2D shapes
def arc(cx, cy, r, a0, a1, n=8):
    """Points on a circular arc from angle a0 to a1 (radians), inclusive."""
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def rounded_rect(x0, y0, x1, y1, r, n=4):
    """Counter-clockwise rounded rectangle."""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = []
    pts += arc(x1 - r, y0 + r, r, -math.pi / 2, 0, n)
    pts += arc(x1 - r, y1 - r, r, 0, math.pi / 2, n)
    pts += arc(x0 + r, y1 - r, r, math.pi / 2, math.pi, n)
    pts += arc(x0 + r, y0 + r, r, math.pi, 1.5 * math.pi, n)
    return dedupe(pts)


def fillet(pts, radius, n=4, closed=True):
    """Round every corner of a polygon (list of (x, y)) with the given radius (or per-vertex list)."""
    out = []
    m = len(pts)
    for i in range(m):
        if not closed and (i == 0 or i == m - 1):
            out.append(pts[i])
            continue
        r = radius[i] if isinstance(radius, (list, tuple)) else radius
        p = Vector(pts[i])
        a = Vector(pts[i - 1]) - p
        b = Vector(pts[(i + 1) % m]) - p
        if r <= 0 or a.length < 1e-9 or b.length < 1e-9:
            out.append(pts[i])
            continue
        a.normalize()
        b.normalize()
        cos_t = max(-0.9999, min(0.9999, a.dot(b)))
        theta = math.acos(cos_t)
        if theta < 1e-3 or abs(theta - math.pi) < 1e-3:
            out.append(pts[i])
            continue
        d = r / math.tan(theta / 2)
        d = min(d, (Vector(pts[i - 1]) - p).length * 0.49, (Vector(pts[(i + 1) % m]) - p).length * 0.49)
        rr = d * math.tan(theta / 2)
        p0 = p + a * d
        p1 = p + b * d
        bis = (a + b).normalized()
        c = p + bis * (rr / math.sin(theta / 2))
        a0 = math.atan2(p0.y - c.y, p0.x - c.x)
        a1 = math.atan2(p1.y - c.y, p1.x - c.x)
        da = (a1 - a0 + math.pi) % TAU - math.pi
        for k in range(n + 1):
            t = a0 + da * k / n
            out.append((c.x + rr * math.cos(t), c.y + rr * math.sin(t)))
    return dedupe(out)


def dedupe(pts, eps=1e-7):
    out = []
    for p in pts:
        if not out or (abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps):
            out.append(p)
    if len(out) > 2 and abs(out[0][0] - out[-1][0]) < eps and abs(out[0][1] - out[-1][1]) < eps:
        out.pop()
    return out


def superellipse(a, b, n=3.0, count=48, phase=0.0):
    """Rounded-rectangle-ish closed curve (|x/a|^n + |y/b|^n = 1), counter-clockwise."""
    pts = []
    for i in range(count):
        t = phase + TAU * i / count
        c, s = math.cos(t), math.sin(t)
        pts.append((a * math.copysign(abs(c) ** (2 / n), c), b * math.copysign(abs(s) ** (2 / n), s)))
    return pts


def signed_area(pts):
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))


# ----------------------------------------------------------------------------- objects
def new_object(name, bm, mat=None):
    mesh = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def _cap_and_bridge(bm, rings, cap_start=True, cap_end=True):
    """rings: list of lists of BMVerts with equal counts. Returns side faces."""
    faces = []
    n = len(rings[0])
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(n):
            j = (i + 1) % n
            faces.append(bm.faces.new((a[i], a[j], b[j], b[i])))
    if cap_start:
        bm.faces.new(list(reversed(rings[0])))
    if cap_end:
        bm.faces.new(rings[-1])
    return faces


def extrude_yz(name, pts, x0, x1, mat=None):
    """Side-view profile (list of (y, z)) extruded across X from x0 to x1."""
    if signed_area(pts) < 0:
        pts = list(reversed(pts))
    bm = bmesh.new()
    ra = [bm.verts.new((x0, y, z)) for (y, z) in pts]
    rb = [bm.verts.new((x1, y, z)) for (y, z) in pts]
    _cap_and_bridge(bm, [ra, rb])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return new_object(name, bm, mat)


def extrude_xz(name, pts, y0, y1, mat=None):
    """Cross-section (list of (x, z)) extruded along Y from y0 to y1."""
    if signed_area(pts) < 0:
        pts = list(reversed(pts))
    bm = bmesh.new()
    ra = [bm.verts.new((x, y0, z)) for (x, z) in pts]
    rb = [bm.verts.new((x, y1, z)) for (x, z) in pts]
    _cap_and_bridge(bm, [ra, rb])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return new_object(name, bm, mat)


def extrude_xy(name, pts, z0, z1, mat=None):
    """Top-view outline (list of (x, y)) extruded along Z."""
    if signed_area(pts) < 0:
        pts = list(reversed(pts))
    bm = bmesh.new()
    ra = [bm.verts.new((x, y, z0)) for (x, y) in pts]
    rb = [bm.verts.new((x, y, z1)) for (x, y) in pts]
    _cap_and_bridge(bm, [ra, rb])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return new_object(name, bm, mat)


def loft(name, rings, mat=None, cap_start=True, cap_end=True):
    """rings: list of lists of 3D points (same count, consistent winding)."""
    bm = bmesh.new()
    vrings = [[bm.verts.new(p) for p in ring] for ring in rings]
    _cap_and_bridge(bm, vrings, cap_start, cap_end)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return new_object(name, bm, mat)


def lathe(name, profile, axis_matrix=Matrix(), segments=32, mat=None, closed_ends=True, closed_profile=False):
    """Revolve (r, h) profile points around local +Z, then transform by axis_matrix.
    closed_profile bridges the last ring back to the first (tubes, rings)."""
    bm = bmesh.new()
    rings = []
    for (r, h) in profile:
        if r < 1e-6:
            rings.append([bm.verts.new(axis_matrix @ Vector((0, 0, h)))])
            continue
        rings.append([bm.verts.new(axis_matrix @ Vector((r * math.cos(TAU * i / segments), r * math.sin(TAU * i / segments), h)))
                      for i in range(segments)])
    pairs = list(zip(rings[:-1], rings[1:]))
    if closed_profile:
        pairs.append((rings[-1], rings[0]))
        closed_ends = False
    for a, b in pairs:
        if len(a) == 1 and len(b) == 1:
            continue
        if len(a) == 1:
            for i in range(segments):
                bm.faces.new((a[0], b[i], b[(i + 1) % segments]))
        elif len(b) == 1:
            for i in range(segments):
                bm.faces.new((a[i], b[0], a[(i + 1) % segments]))
        else:
            for i in range(segments):
                j = (i + 1) % segments
                bm.faces.new((a[i], b[i], b[j], a[j]))
    if closed_ends:
        if len(rings[0]) > 2:
            bm.faces.new(list(reversed(rings[0])))
        if len(rings[-1]) > 2:
            bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return new_object(name, bm, mat)


def cyl(name, r, p0, p1, segments=24, mat=None, r1=None):
    """Cylinder (or cone with r1) between two points."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    rot = d.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    m = Matrix.Translation(p0) @ rot
    return lathe(name, [(0, 0), (r, 0), (r if r1 is None else r1, d.length), (0, d.length)], m, segments, mat)


def box(name, p0, p1, mat=None):
    """Axis-aligned box between two corners."""
    (x0, y0, z0), (x1, y1, z1) = p0, p1
    return extrude_xy(name, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z0, z1, mat)


def transform(obj, matrix):
    obj.data.transform(matrix)
    obj.data.update()
    return obj


# ----------------------------------------------------------------------------- modifiers
def bevel(obj, width=0.0006, segments=2, angle=35.0, harden=True, weighted=True, clamp=True):
    mod = obj.modifiers.new("Bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = math.radians(angle)
    mod.harden_normals = harden
    mod.use_clamp_overlap = clamp
    mod.miter_outer = "MITER_ARC"
    obj.data.shade_smooth()
    if weighted:
        wn = obj.modifiers.new("WeightedNormal", "WEIGHTED_NORMAL")
        wn.keep_sharp = True
    return mod


def cut(target, cutter, op="DIFFERENCE", transfer=False):
    """Boolean (exact) placed before any bevel so cut edges bevel too; the cutter is deleted on apply().
    transfer=True keeps the operand's own materials (unions of differently-textured parts);
    otherwise the new faces take the target's first material."""
    mod = target.modifiers.new("Bool", "BOOLEAN")
    mod.operation = op
    mod.object = cutter
    mod.solver = "EXACT"
    mod.material_mode = "TRANSFER" if transfer else "INDEX"
    idx = len(target.modifiers) - 1
    first_non_bool = next((i for i, m in enumerate(target.modifiers) if m.type != "BOOLEAN"), idx)
    if first_non_bool < idx:
        target.modifiers.move(idx, first_non_bool)
    cutter.hide_render = True
    cutter.hide_viewport = True
    target["_cutters"] = list(target.get("_cutters", [])) + [cutter.name]
    return mod


def apply_all(obj):
    """Evaluate modifiers into the mesh and delete boolean cutters."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    bpy.data.meshes.remove(old)
    for name in obj.get("_cutters", []):
        c = bpy.data.objects.get(name)
        if c is not None:
            bpy.data.objects.remove(c, do_unlink=True)
    if "_cutters" in obj:
        del obj["_cutters"]
    return obj


def merge(name, objs):
    """Apply modifiers and join into one object (materials and custom normals kept)."""
    for o in objs:
        apply_all(o)
    target = objs[0]
    if len(objs) > 1:
        with bpy.context.temp_override(active_object=target, selected_editable_objects=objs, selected_objects=objs):
            bpy.ops.object.join()
    target.name = name
    target.data.name = name
    return target
