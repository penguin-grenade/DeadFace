"""Geometry + material kit for building lightmapped levels headless.

Everything is authored in Blender world space (metres, +Z up, +Y forward).
Parts are appended to a `Batch` per material, so a whole level ends up as a few
dozen meshes (one draw call each in the game). Every part carries:
  * its own tiling UVs in metres (u follows the long axis, so wood grain and
    stripes run the right way), later divided by the material's tile size
  * a lightmap density hint, so bolts and rods get fewer lightmap texels than
    floors and walls

The kit also records what the game needs besides visuals:
  * colliders   - boxes / cylinders with a surface kind, in three.js space
  * spots       - named gameplay positions (spawn, targets, props)
  * lights      - mixed (real-time direct + baked indirect) and baked-only
"""
from __future__ import annotations

import math
import os
import random

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

# --------------------------------------------------------------------------------------
# Materials. `tex` names a texture set from bake_textures.py; `tile` is metres per tile.
# `surface` drives bullet impact effects in the game. `bevel` is the edge bevel (m)
# applied to every part of that material.
# --------------------------------------------------------------------------------------
MATERIALS = {
    "floor": dict(tex="floor_concrete", tile=4.0, surface="concrete"),
    "floor_paint": dict(tex="floor_concrete", tile=4.0, surface="concrete", tint=(2.3, 1.75, 0.25), rough=0.8),
    "concrete": dict(tex="floor_concrete", tile=2.0, surface="concrete", tint=(1.12, 1.12, 1.1), bevel=0.012),
    "cmu": dict(tex="cmu_block", tile=1.6, surface="concrete", bevel=0.01),
    "corrugated": dict(tex="corrugated", tile=2.0, surface="metal", metal=0.8),
    "deck": dict(tex="corrugated", tile=2.0, surface="metal", metal=0.8, tint=(0.8, 0.8, 0.8)),
    "steel": dict(tex="steel", tile=1.5, surface="metal", metal=0.5, bevel=0.004),
    "galv": dict(tex="corrugated", tile=1.0, surface="metal", metal=0.8, bevel=0.002),
    "rack_blue": dict(tex="rack_blue", tile=1.0, surface="metal", metal=0.2, bevel=0.002),
    "rack_orange": dict(tex="rack_orange", tile=1.0, surface="metal", metal=0.2, bevel=0.003),
    "wood": dict(tex="wood", tile=1.2, surface="wood", bevel=0.004),
    "plywood": dict(tex="plywood", tile=1.2, surface="wood", tint=(1.0, 1.0, 1.0), bevel=0.003),
    "drywall": dict(tex="drywall", tile=2.4, surface="plaster", bevel=0.004),
    "ceiling_tile": dict(tex="drywall", tile=0.6, surface="plaster", tint=(1.12, 1.12, 1.1)),
    "sandbag": dict(tex="sandbag", tile=0.5, surface="fabric", tint=(0.74, 0.72, 0.7)),
    "rubber": dict(tex="rubber", tile=0.5, surface="rubber", bevel=0.003),
    "cardboard": dict(tex="cardboard", tile=0.6, surface="cardboard", bevel=0.003),
    "paper": dict(tex="paper", tile=0.5, surface="cardboard"),
    "paint_grey": dict(tex="painted", tile=1.0, surface="metal", metal=0.2, tint=(0.5, 0.51, 0.52), bevel=0.003),
    "paint_dark": dict(tex="painted", tile=1.0, surface="metal", metal=0.2, tint=(0.14, 0.145, 0.15), bevel=0.003),
    "paint_red": dict(tex="painted", tile=1.0, surface="metal", metal=0.1, tint=(0.72, 0.07, 0.05), bevel=0.003),
    "paint_blue": dict(tex="painted", tile=1.0, surface="metal", metal=0.1, tint=(0.08, 0.2, 0.5), bevel=0.003),
    "paint_yellow": dict(tex="painted", tile=1.0, surface="metal", metal=0.1, tint=(1.05, 0.72, 0.05), bevel=0.003),
    "paint_green": dict(tex="painted", tile=1.0, surface="metal", metal=0.1, tint=(0.16, 0.25, 0.14), bevel=0.003),
    "paint_white": dict(tex="painted", tile=1.0, surface="metal", metal=0.1, tint=(1.05, 1.05, 1.03), bevel=0.003),
    "plastic_black": dict(tex="rubber", tile=0.5, surface="rubber", tint=(1.8, 1.8, 1.8), rough=0.75, bevel=0.002),
    "plastic_grey": dict(tex="painted", tile=0.5, surface="rubber", tint=(0.3, 0.3, 0.31), rough=1.2, bevel=0.002),
    "plastic_orange": dict(tex="painted", tile=0.5, surface="rubber", tint=(1.15, 0.36, 0.05), bevel=0.002),
    "plastic_yellow": dict(tex="painted", tile=0.5, surface="rubber", tint=(1.1, 0.78, 0.06), bevel=0.002),
    "wrap": dict(tex="painted", tile=0.5, surface="cardboard", tint=(0.9, 0.92, 0.95), rough=0.55),
}

# Parts at or below this lightmap density are lit per vertex instead of by the lightmap:
# tiny lightmap charts cost more padding than texels, and dense meshes carry GI well per vertex.
VTX_MAX = 0.5
# Vertex-lit parts get their long edges split so GI has enough samples across big faces.
VTX_EDGE = 0.75
VTX_AREA = 0.3

# Glass and emitters are not lightmapped; the game builds them from level.json.
EMISSIVE = {}
GLASS = {}


def srgb_to_linear(c):
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)


def b2t(v):
    """Blender (x, y, z) -> three.js (x, y, z)."""
    return [round(v[0], 4), round(v[2], 4), round(-v[1], 4)]


def q2t(q: Quaternion):
    """Blender rotation -> three.js quaternion [x, y, z, w]."""
    return [round(q.x, 5), round(q.z, 5), round(-q.y, 5), round(q.w, 5)]


def frame_from(x_axis, up=(0, 0, 1)):
    """Orthonormal 3x3 (columns x, y, z) with x along x_axis and z as close to `up` as possible."""
    x = Vector(x_axis).normalized()
    upv = Vector(up)
    if abs(x.dot(upv.normalized())) > 0.99:
        upv = Vector((1, 0, 0)) if abs(x.x) < 0.9 else Vector((0, 1, 0))
    y = upv.cross(x).normalized()
    z = x.cross(y)
    return Matrix((x, y, z)).transposed()


def to_matrix(loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0):
    m = Euler(rot).to_matrix().to_4x4() if not isinstance(rot, Matrix) else rot.to_4x4()
    if not isinstance(scale, (tuple, list)):
        scale = (scale, scale, scale)
    s = Matrix.Diagonal((*scale, 1.0))
    return Matrix.Translation(loc) @ m @ s


BOX_CORNERS = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
               (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
BOX_FACES = [((0, 3, 2, 1), 2), ((4, 5, 6, 7), 2), ((0, 1, 5, 4), 1),
             ((1, 2, 6, 5), 0), ((2, 3, 7, 6), 1), ((3, 0, 4, 7), 0)]


class Batch:
    """Accumulates many parts into one mesh (world space) with per-corner UVs in metres."""

    def __init__(self, kit, name, mat, kind="static", lm=1.0, sharp=50.0, bevel=None, weld=False):
        self.kit = kit
        self.name = name
        self.mat = mat
        self.kind = kind
        self.lm = lm
        self.sharp = sharp
        self.bevel = bevel
        self.weld = weld
        self.verts: list = []
        self.faces: list = []
        self.fuv: list = []
        self.vd: list = []
        self.vc: list = []
        self.vp: list = []
        self.has_flat = False
        self.rng = random.Random(hash(name) & 0xFFFF)

    # ---------------------------------------------------------------- core
    def _add(self, verts, faces, fuv, lm=None, center=None, flat=None):
        o = len(self.verts)
        verts = [tuple(v) for v in verts]
        self.verts.extend(verts)
        if flat is not None:
            self.has_flat = True
            self.vp.extend(tuple(v) for v in flat)
        else:
            self.vp.extend(verts)
        self.faces.extend(tuple(i + o for i in f) for f in faces)
        self.fuv.extend(fuv)
        d = self.lm if lm is None else lm
        if center is None:
            n = len(verts)
            center = (sum(v[0] for v in verts) / n, sum(v[1] for v in verts) / n, sum(v[2] for v in verts) / n)
        self.vd.extend([d] * len(verts))
        self.vc.extend([tuple(center)] * len(verts))

    def _offset(self):
        return (self.rng.random() * 8.0, self.rng.random() * 8.0)

    # ---------------------------------------------------------------- primitives
    def box(self, center, size, rot=None, lm=None, uv="long", faces=None):
        """Axis box in a rotated frame. uv='long' runs u along the longest in-plane side,
        uv='world' uses world-aligned planar mapping (continuous across wall segments).
        `faces` optionally limits which sides are emitted: subset of '-x +x -y +y -z +z'."""
        m = rot if isinstance(rot, Matrix) else (Euler(rot).to_matrix() if rot is not None else Matrix.Identity(3))
        h = (size[0] / 2, size[1] / 2, size[2] / 2)
        c = Vector(center)
        local = [Vector((sx * h[0], sy * h[1], sz * h[2])) for sx, sy, sz in BOX_CORNERS]
        world = [c + m @ p for p in local]
        off = self._offset() if uv == "long" else (0.0, 0.0)
        names = {0: ("-x", "+x"), 1: ("-y", "+y"), 2: ("-z", "+z")}
        fs, fuv = [], []
        for idx, (f, axis) in enumerate(BOX_FACES):
            sign = 1 if idx in (1, 3, 4) else 0  # which side along the axis
            key = names[axis][sign]
            if faces is not None and key not in faces:
                continue
            fs.append(f)
            if uv == "world":
                n = (m @ Vector([1 if a == axis else 0 for a in range(3)]))
                fuv.append([_world_uv(world[i], n) for i in f])
            else:
                inplane = [a for a in range(3) if a != axis]
                ua, va = (inplane if size[inplane[0]] >= size[inplane[1]] else inplane[::-1])
                fuv.append([(local[i][ua] + h[ua] + off[0], local[i][va] + h[va] + off[1]) for i in f])
        self._add(world, fs, fuv, lm, center)

    def box_between(self, p0, p1, w, h, up=(0, 0, 1), lm=None, faces=None):
        """Box whose long axis runs from p0 to p1, cross-section w (side) x h (up)."""
        p0, p1 = Vector(p0), Vector(p1)
        m = frame_from(p1 - p0, up)
        self.box((p0 + p1) / 2, ((p1 - p0).length, w, h), m, lm=lm, faces=faces)

    def prism(self, profile, p0, p1, up=(0, 0, 1), lm=None, caps=True):
        """Extrude a closed 2D profile [(y, z)] (counter-clockwise) from p0 to p1."""
        p0, p1 = Vector(p0), Vector(p1)
        m = frame_from(p1 - p0, up)
        y, z = m.col[1], m.col[2]
        n = len(profile)
        length = (p1 - p0).length
        verts = [p + y * py + z * pz for p in (p0, p1) for (py, pz) in profile]
        per = [0.0]
        for i in range(n):
            a, b = profile[i], profile[(i + 1) % n]
            per.append(per[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        off = self._offset()
        fs, fuv = [], []
        for i in range(n):
            j = (i + 1) % n
            fs.append((i, j, n + j, n + i))
            fuv.append([(off[0], per[i] + off[1]), (off[0], per[i + 1] + off[1]),
                        (off[0] + length, per[i + 1] + off[1]), (off[0] + length, per[i] + off[1])])
        if caps:
            fs.append(tuple(reversed(range(n))))
            fuv.append([(profile[i][0], profile[i][1]) for i in reversed(range(n))])
            fs.append(tuple(range(n, 2 * n)))
            fuv.append([(profile[i][0], profile[i][1]) for i in range(n)])
        self._add(verts, fs, fuv, lm, (p0 + p1) / 2)

    def rod(self, p0, p1, r, sides=8, caps=False, lm=None, r1=None):
        """Cylinder (or cone when r1 is given) between two points."""
        p0, p1 = Vector(p0), Vector(p1)
        m = frame_from(p1 - p0)
        u, v = m.col[1], m.col[2]
        r1 = r if r1 is None else r1
        verts = []
        for p, rr in ((p0, r), (p1, r1)):
            for i in range(sides):
                t = 2 * math.pi * i / sides
                verts.append(p + (u * math.cos(t) + v * math.sin(t)) * rr)
        length = (p1 - p0).length
        circ = 2 * math.pi * max(r, r1)
        off = self._offset()
        fs, fuv = [], []
        for i in range(sides):
            j = (i + 1) % sides
            fs.append((i, j, sides + j, sides + i))
            a0, a1 = circ * i / sides, circ * (i + 1) / sides
            fuv.append([(off[0], off[1] + a0), (off[0], off[1] + a1), (off[0] + length, off[1] + a1), (off[0] + length, off[1] + a0)])
        if caps:
            for ring, rr, rev in ((0, r, True), (sides, r1, False)):
                idx = [ring + i for i in range(sides)]
                if rev:
                    idx = idx[::-1]
                fs.append(tuple(idx))
                fuv.append([(rr * math.cos(2 * math.pi * ((k - ring) % sides) / sides), rr * math.sin(2 * math.pi * ((k - ring) % sides) / sides)) for k in idx])
        self._add(verts, fs, fuv, lm, (p0 + p1) / 2)

    def lathe(self, profile, matrix=None, segments=24, closed=False, lm=None, phase=0.0):
        """Revolve [(r, z)] around local Z. Points with r == 0 become poles. `closed` joins last to first."""
        mat = matrix if matrix is not None else Matrix.Identity(4)
        verts, rings = [], []
        for (r, z) in profile:
            if r <= 1e-6:
                rings.append([len(verts)])
                verts.append(mat @ Vector((0, 0, z)))
            else:
                ring = []
                for i in range(segments):
                    t = 2 * math.pi * i / segments + phase
                    ring.append(len(verts))
                    verts.append(mat @ Vector((r * math.cos(t), r * math.sin(t), z)))
                rings.append(ring)
        rmax = max(r for r, _ in profile)
        circ = 2 * math.pi * rmax
        arc = [0.0]
        pts = profile + ([profile[0]] if closed else [])
        for a, b in zip(pts, pts[1:]):
            arc.append(arc[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        off = self._offset()
        fs, fuv = [], []
        nr = len(rings)
        pairs = [(k, k + 1) for k in range(nr - 1)] + ([(nr - 1, 0)] if closed else [])
        for pi_, (a, b) in enumerate(pairs):
            ra, rb = rings[a], rings[b]
            va, vb = arc[pi_] + off[1], arc[pi_ + 1] + off[1]
            for i in range(segments):
                j = i + 1
                ua, ub = off[0] + circ * i / segments, off[0] + circ * j / segments
                if len(ra) == 1 and len(rb) == 1:
                    continue
                if len(ra) == 1:
                    fs.append((ra[0], rb[j % segments], rb[i]))
                    fuv.append([((ua + ub) / 2, va), (ub, vb), (ua, vb)])
                elif len(rb) == 1:
                    fs.append((ra[i], ra[j % segments], rb[0]))
                    fuv.append([(ua, va), (ub, va), ((ua + ub) / 2, vb)])
                else:
                    fs.append((ra[i], ra[j % segments], rb[j % segments], rb[i]))
                    fuv.append([(ua, va), (ub, va), (ub, vb), (ua, vb)])
        center = mat @ Vector((0, 0, sum(z for _, z in profile) / len(profile)))
        self._add(verts, fs, fuv, lm, center)

    def add_mesh(self, verts, faces, matrix=None, fuv=None, lm=None):
        """Arbitrary mesh in local space. Without `fuv`, UVs come from a local box projection."""
        if fuv is None:
            fuv = _local_box_uv(verts, faces, self._offset())
        if matrix is not None:
            verts = [matrix @ Vector(v) for v in verts]
        self._add(verts, faces, fuv, lm)

    def quad(self, a, b, c, d, lm=None, uv="long", flat=None):
        """Single quad (counter-clockwise when seen from the front). `flat` optionally gives
        the corner positions projected onto the panel plane, used only to chart lightmap UVs
        (keeps ribbed sheet metal in one lightmap chart)."""
        a, b, c, d = map(Vector, (a, b, c, d))
        eu = (b - a)
        ev = (d - a)
        if uv == "world":
            n = eu.cross(ev)
            fuv = [[_world_uv(p, n) for p in (a, b, c, d)]]
        else:
            lu, lv = eu.length, ev.length
            eu.normalize()
            ev.normalize()
            off = self._offset()
            fuv = [[((p - a).dot(eu) + off[0], (p - a).dot(ev) + off[1]) for p in (a, b, c, d)]]
        self._add([a, b, c, d], [(0, 1, 2, 3)], fuv, lm, flat=flat)

    # ---------------------------------------------------------------- build
    def build(self):
        """Returns up to two objects: parts with lightmap density above VTX_MAX get lightmap
        UVs (`lit`='lm'); finer parts are lit per vertex (`lit`='vtx')."""
        if not self.faces:
            return []
        groups = {"lm": [], "vtx": []}
        for fi, f in enumerate(self.faces):
            vtx = self.kind == "static" and self.vd[f[0]] <= VTX_MAX + 1e-6  # occluders/glass/emissive stay whole
            groups["vtx" if vtx else "lm"].append(fi)
        out = []
        for mode, fids in groups.items():
            if not fids:
                continue
            obj = self._build_part(self.name if mode == "lm" else self.name + ".v", fids, subdivide=mode == "vtx")
            obj["lit"] = mode if self.kind == "static" else "none"
            out.append(obj)
        return out

    def _build_part(self, name, fids, subdivide=False):
        import bmesh

        kit = self.kit
        used = sorted({i for fi in fids for i in self.faces[fi]})
        remap = {old: new for new, old in enumerate(used)}
        verts = [self.verts[i] for i in used]
        faces = [tuple(remap[i] for i in self.faces[fi]) for fi in fids]
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.update()
        spec = MATERIALS.get(self.mat, {}) if isinstance(self.mat, str) else {}
        tile = spec.get("tile", 1.0)
        layer = me.uv_layers.new(name="UVMap")
        flat = np.array([c for fi in fids for uv in self.fuv[fi] for c in uv], dtype=np.float32) / tile
        layer.data.foreach_set("uv", flat)
        a = me.attributes.new("lm_d", "FLOAT", "POINT")
        a.data.foreach_set("value", np.array([self.vd[i] for i in used], dtype=np.float32))
        a = me.attributes.new("lm_c", "FLOAT_VECTOR", "POINT")
        a.data.foreach_set("vector", np.array([self.vc[i] for i in used], dtype=np.float32).ravel())
        if self.has_flat:
            a = me.attributes.new("lm_p", "FLOAT_VECTOR", "POINT")
            a.data.foreach_set("vector", np.array([self.vp[i] for i in used], dtype=np.float32).ravel())
        if self.weld or subdivide:
            bm = bmesh.new()
            bm.from_mesh(me)
            if self.weld:
                bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=2e-4)
            if subdivide:
                for _ in range(7):
                    long_edges = [e for e in bm.edges if e.calc_length() > VTX_EDGE
                                  and any(f.calc_area() > VTX_AREA for f in e.link_faces)]
                    if not long_edges:
                        break
                    bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1, use_grid_fill=True)
            bm.to_mesh(me)
            bm.free()
        me.shade_smooth()
        me.set_sharp_from_angle(angle=math.radians(self.sharp))
        obj = kit._obj(name, me, self.mat, self.kind)
        bevel = spec.get("bevel", 0.0) if self.bevel is None else self.bevel
        if bevel > 0:
            mod = obj.modifiers.new("Bevel", "BEVEL")
            mod.width = bevel
            mod.segments = 1
            mod.limit_method = "ANGLE"
            mod.angle_limit = math.radians(self.sharp)
            mod.harden_normals = True
            mod.use_clamp_overlap = True
        return obj


def _world_uv(p, n):
    n = Vector(n)
    ax = max(range(3), key=lambda k: abs(n[k]))
    if ax == 2:
        return (p.x, p.y)
    if ax == 0:
        return (p.y, p.z)
    return (p.x, p.z)


def _local_box_uv(verts, faces, off):
    vs = [Vector(v) for v in verts]
    out = []
    for f in faces:
        a, b, c = vs[f[0]], vs[f[1]], vs[f[2]]
        n = (b - a).cross(c - a)
        ax = max(range(3), key=lambda k: abs(n[k]))
        if ax == 2:
            out.append([(vs[i].x + off[0], vs[i].y + off[1]) for i in f])
        elif ax == 0:
            out.append([(vs[i].y + off[0], vs[i].z + off[1]) for i in f])
        else:
            out.append([(vs[i].x + off[0], vs[i].z + off[1]) for i in f])
    return out


class Kit:
    def __init__(self, tex_dir):
        self.tex_dir = tex_dir
        self.static: list = []
        self.glass: list = []
        self.emissive: list = []
        self.occluders: list = []
        self.colliders: list[dict] = []
        self.spots: dict[str, list] = {}
        self.lights: list[dict] = []
        self.mats: dict[str, bpy.types.Material] = {}
        self.batches: dict[tuple, Batch] = {}
        self.col = bpy.context.scene.collection

    def B(self, mat, kind="static", lm=None, sharp=50.0, tag="", weld=False):
        """Batch for a material (created on first use). `tag` splits a material into
        separate meshes (e.g. to give fine parts a different lightmap density default)."""
        key = (mat, kind, tag)
        if key not in self.batches:
            name = mat if not tag else f"{mat}.{tag}"
            self.batches[key] = Batch(self, name, mat, kind=kind, lm=1.0 if lm is None else lm, sharp=sharp, weld=weld)
        return self.batches[key]

    def build_batches(self):
        return [o for b in self.batches.values() for o in b.build()]

    # ---------------------------------------------------------------- materials
    def mat(self, name):
        if name in self.mats:
            return self.mats[name]
        spec = MATERIALS[name]
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        uv = nt.nodes.new("ShaderNodeUVMap")
        uv.uv_map = "UVMap"
        base = os.path.join(self.tex_dir, spec["tex"])
        alb = nt.nodes.new("ShaderNodeTexImage")
        alb.image = bpy.data.images.load(base + "_albedo.jpg", check_existing=True)
        nt.links.new(uv.outputs["UV"], alb.inputs["Vector"])
        tint = spec.get("tint")
        if tint:
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.clamp_result = True
            mix.inputs[0].default_value = 1.0
            nt.links.new(alb.outputs["Color"], mix.inputs[6])
            mix.inputs[7].default_value = (*tint, 1.0)
            nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        else:
            nt.links.new(alb.outputs["Color"], bsdf.inputs["Base Color"])
        rough = nt.nodes.new("ShaderNodeTexImage")
        rough.image = bpy.data.images.load(base + "_roughness.jpg", check_existing=True)
        rough.image.colorspace_settings.name = "Non-Color"
        nt.links.new(uv.outputs["UV"], rough.inputs["Vector"])
        rmul = nt.nodes.new("ShaderNodeMath")
        rmul.operation = "MULTIPLY"
        rmul.use_clamp = True
        rmul.inputs[1].default_value = spec.get("rough", 1.0)
        nt.links.new(rough.outputs["Color"], rmul.inputs[0])
        nt.links.new(rmul.outputs[0], bsdf.inputs["Roughness"])
        nrm = nt.nodes.new("ShaderNodeTexImage")
        nrm.image = bpy.data.images.load(base + "_normal.jpg", check_existing=True)
        nrm.image.colorspace_settings.name = "Non-Color"
        nt.links.new(uv.outputs["UV"], nrm.inputs["Vector"])
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.name = "NormalMap"
        nmap.uv_map = "UVMap"
        nt.links.new(nrm.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        bsdf.inputs["Metallic"].default_value = spec.get("metal", 0.0)
        m["tile"] = spec["tile"]
        self.mats[name] = m
        return m

    def glass_mat(self, name="glass", tint=(0.8, 0.85, 0.85), transparency=0.9, rough=0.08, frosted=False):
        if name in self.mats:
            return self.mats[name]
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        mix = nt.nodes.new("ShaderNodeMixShader")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        tr.inputs["Color"].default_value = (*tint, 1)
        gl = nt.nodes.new("ShaderNodeBsdfGlossy")
        gl.inputs["Roughness"].default_value = rough
        mix.inputs["Fac"].default_value = 1.0 - transparency
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(gl.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
        GLASS[name] = dict(tint=list(tint), opacity=round(1.0 - transparency, 3), rough=rough, frosted=frosted)
        self.mats[name] = m
        return m

    def emit_mat(self, name, color, strength, baked=True):
        """Emitter. `baked` emitters light the lightmap; others only glow in the game."""
        if name in self.mats:
            return self.mats[name]
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        em = nt.nodes.new("ShaderNodeEmission")
        em.name = "Emission"
        em.inputs["Color"].default_value = (*color, 1)
        em.inputs["Strength"].default_value = strength
        nt.links.new(em.outputs[0], out.inputs["Surface"])
        m["baked"] = baked
        m["strength"] = strength
        EMISSIVE[name] = dict(color=[round(c, 4) for c in color], strength=strength, baked=baked)
        self.mats[name] = m
        return m

    # ---------------------------------------------------------------- objects
    def _obj(self, name, mesh, mat, kind="static"):
        obj = bpy.data.objects.new(name, mesh)
        self.col.objects.link(obj)
        if isinstance(mat, str):
            obj.data.materials.append(self.mat(mat))
        elif mat is not None:
            obj.data.materials.append(mat)
        {"static": self.static, "glass": self.glass, "emissive": self.emissive, "occluder": self.occluders}[kind].append(obj)
        if kind == "emissive":
            # Bulbs and panels stand in for the light objects inside them: they glow, but must not
            # block the light (a spot inside a closed bulb mesh would otherwise be shadowed).
            obj.visible_shadow = False
        return obj

    # ---------------------------------------------------------------- colliders
    def collide_box(self, center, size, rot=(0, 0, 0), surface="concrete"):
        q = (rot if isinstance(rot, Matrix) else Euler(rot).to_matrix()).to_quaternion()
        self.colliders.append(dict(type="box", pos=b2t(center), half=[round(size[0] / 2, 4), round(size[2] / 2, 4), round(size[1] / 2, 4)],
                                   quat=q2t(q), surface=surface))

    def collide_cyl(self, center, r, h, surface="metal"):
        """Upright cylinder (Blender Z axis)."""
        self.colliders.append(dict(type="cyl", pos=b2t(center), radius=round(r, 4), half=round(h / 2, 4), quat=[0, 0, 0, 1], surface=surface))

    # ---------------------------------------------------------------- gameplay
    def spot(self, name, loc, yaw=0.0):
        self.spots.setdefault(name, []).append(dict(pos=b2t(loc), yaw=round(yaw, 4)))

    # ---------------------------------------------------------------- lights
    def spot_light(self, name, loc, target, candela, color=(1.0, 0.85, 0.65), angle_deg=55, penumbra=0.5,
                   radius=0.12, mode="mixed", shadow=True, flicker=False, volumetric=0.0, distance=0.0):
        """`candela` uses three.js units. Blender power = 4π·cd keeps the bake and the game in the same units."""
        ld = bpy.data.lights.new(name, "SPOT")
        ld.energy = candela * 4 * math.pi
        ld.color = color
        ld.spot_size = math.radians(angle_deg) * 2
        ld.spot_blend = penumbra
        ld.shadow_soft_size = radius
        obj = bpy.data.objects.new(name, ld)
        self.col.objects.link(obj)
        obj.location = loc
        direction = Vector(target) - Vector(loc)
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        self.lights.append(dict(
            name=name, type="spot", mode=mode, obj=obj,
            pos=b2t(loc), target=b2t(target), intensity=candela, color=[round(c, 4) for c in color],
            angle=round(math.radians(angle_deg), 4), penumbra=penumbra, shadow=shadow, flicker=flicker,
            volumetric=volumetric, distance=distance,
        ))
        return obj

    def point_light(self, name, loc, candela, color, radius=0.1, mode="baked"):
        ld = bpy.data.lights.new(name, "POINT")
        ld.energy = candela * 4 * math.pi
        ld.color = color
        ld.shadow_soft_size = radius
        obj = bpy.data.objects.new(name, ld)
        self.col.objects.link(obj)
        obj.location = loc
        self.lights.append(dict(name=name, type="point", mode=mode, obj=obj, pos=b2t(loc), intensity=candela,
                                color=[round(c, 4) for c in color]))
        return obj

    def area_light(self, name, loc, size, power, color, rot=(0, 0, 0), mode="baked", spread=180):
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape = "RECTANGLE"
        ld.size = size[0]
        ld.size_y = size[1]
        ld.energy = power
        ld.color = color
        ld.spread = math.radians(spread)
        obj = bpy.data.objects.new(name, ld)
        self.col.objects.link(obj)
        obj.location = loc
        obj.rotation_euler = rot
        normal = obj.rotation_euler.to_matrix() @ Vector((0.0, 0.0, -1.0))
        self.lights.append(dict(name=name, type="area", mode=mode, obj=obj, pos=b2t(loc), color=[round(c, 4) for c in color],
                                size=[size[0], size[1]], power=power, normal=b2t(normal)))
        return obj

    def sun(self, name, direction, strength, color, angle_deg=0.6, mode="baked"):
        """`direction` is the direction the light travels (from the sun into the scene)."""
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = strength
        ld.color = color
        ld.angle = math.radians(angle_deg)
        obj = bpy.data.objects.new(name, ld)
        self.col.objects.link(obj)
        obj.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
        d = Vector(direction).normalized()
        self.lights.append(dict(name=name, type="sun", mode=mode, obj=obj, dir=b2t(d), intensity=strength,
                                color=[round(c, 4) for c in color]))
        return obj


# ------------------------------------------------------------------------------------
# Profiles (y, z) in the cross-section plane, counter-clockwise
# ------------------------------------------------------------------------------------

def i_profile(depth, width, tw, tf):
    d, b = depth / 2, width / 2
    t = tw / 2
    return [(-b, -d), (b, -d), (b, -d + tf), (t, -d + tf), (t, d - tf), (b, d - tf), (b, d), (-b, d),
            (-b, d - tf), (-t, d - tf), (-t, -d + tf), (-b, -d + tf)]


def c_profile(depth, width, t):
    d = depth / 2
    return [(0, -d), (width, -d), (width, -d + t), (t, -d + t), (t, d - t), (width, d - t), (width, d), (0, d)]


def angle_profile(leg, t):
    return [(0, 0), (leg, 0), (leg, t), (t, t), (t, leg), (0, leg)]


def rect_tube(w, h, t):
    """Hollow rectangular tube as a single closed outline is not possible; return the outer box."""
    return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]


# ------------------------------------------------------------------------------------
# Lightmap UVs (xatlas) and helpers used before the bake
# ------------------------------------------------------------------------------------

def apply_modifiers(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.convert(target="MESH")


def triangulate(obj):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.to_mesh(obj.data)
    bm.free()


def lightmap_uvs(objs, resolution=2048, padding=3, texels_per_unit=0.0, name="LM"):
    """Pack every object into one lightmap atlas with xatlas. Parts are pre-scaled around
    their centre by their `lm_d` density so small details get proportionally fewer texels.
    Returns the achieved texels per metre (for density 1.0)."""
    import xatlas

    atlas = xatlas.Atlas()
    meshes = []
    for o in objs:
        triangulate(o)
        me = o.data
        nv = len(me.vertices)
        co = np.empty(nv * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        M = np.array(o.matrix_world, dtype=np.float32)
        co = co @ M[:3, :3].T + M[:3, 3]
        if "lm_p" in me.attributes:
            fp = np.empty(nv * 3, dtype=np.float32)
            me.attributes["lm_p"].data.foreach_get("vector", fp)
            co = fp.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        if "lm_d" in me.attributes:
            d = np.empty(nv, dtype=np.float32)
            me.attributes["lm_d"].data.foreach_get("value", d)
            c = np.empty(nv * 3, dtype=np.float32)
            me.attributes["lm_c"].data.foreach_get("vector", c)
            c = c.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            co = c + (co - c) * d[:, None]
        nl = len(me.loops)
        lv = np.empty(nl, dtype=np.int64)
        me.loops.foreach_get("vertex_index", lv)
        tris = lv.reshape(-1, 3).astype(np.uint32)
        atlas.add_mesh(co, tris)
        meshes.append((o, tris))
    copt = xatlas.ChartOptions()
    copt.max_iterations = 2
    popt = xatlas.PackOptions()
    popt.resolution = resolution
    popt.padding = padding
    popt.bilinear = True
    popt.rotate_charts = True
    popt.texels_per_unit = texels_per_unit
    popt.bruteForce = False
    popt.blockAlign = False
    atlas.generate(copt, popt, verbose=False)
    for i, (o, tris) in enumerate(meshes):
        vmap, idx, uvs = atlas[i]
        me = o.data
        layer = me.uv_layers.get(name) or me.uv_layers.new(name=name)
        uv = uvs[idx.reshape(-1)]  # per loop, in triangle order
        layer.data.foreach_set("uv", uv.astype(np.float32).ravel())
    return dict(width=atlas.width, height=atlas.height, atlases=atlas.atlas_count,
                charts=atlas.chart_count, texels_per_unit=atlas.texels_per_unit,
                utilization=[round(float(u), 3) for u in np.atleast_1d(atlas.utilization)])


def join(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    return obj
