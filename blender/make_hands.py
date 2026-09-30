"""First-person arms: gloved firing and support hands wrapped around the pistol
grip, with glove cuffs and jacket sleeves running out of frame.

Output: <out>/arms.glb (one mesh "Arms" with a baked 2K PBR atlas), in the same
axes and origin as pistol.glb so the game parents both to one viewmodel pivot.

Base mesh: the MIT "generic-hand" from the WebXR input profiles
(blender/assets/hands). Posing happens here:
  * numpy linear-blend skinning over the WebXR joint chain,
  * each hand is placed with a grip frame on the pistol (knuckle line along the
    raked grip, palm on the panel),
  * fingers close with a grasp solver: every joint of a finger curls at its own
    rate; when a phalanx touches the grip (or the other hand), it and the joints
    behind it stop while the rest keep curling -- the idea robot-hand grasp
    planners use,
  * thumbs are aimed with CCD inverse kinematics and pushed clear of the frame.
The posed hands are subdivided and inflated into gloves (nylon back, suede palm
and fingertips, moulded knuckle guard, recessed seams), then baked like the
pistol (texbake.py).

    python3 blender/make_hands.py --out public/models [--preview DIR] [--no-bake]
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import bmesh  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

import common as c  # noqa: E402
import hardsurface as hs  # noqa: E402
import make_pistol as mp  # noqa: E402
import texbake as tb  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "hands")
FINGERS = ("index", "middle", "ring", "pinky")
CHAIN = {f: [f"{f}-finger-metacarpal", f"{f}-finger-phalanx-proximal", f"{f}-finger-phalanx-intermediate",
             f"{f}-finger-phalanx-distal", f"{f}-finger-tip"] for f in FINGERS}
CHAIN["thumb"] = ["thumb-metacarpal", "thumb-phalanx-proximal", "thumb-phalanx-distal", "thumb-tip"]
GLOVE = 0.0011          # glove material thickness added over the skin

SURF = {
    #           id  base colour (linear)     metal rough
    "Glove":   (1, (0.030, 0.031, 0.032), 0.0, 0.82),   # split into nylon/suede/guard by painted fields
    "Rubber":  (3, (0.020, 0.020, 0.021), 0.0, 0.50),
    "Cuff":    (4, (0.026, 0.026, 0.027), 0.0, 0.70),
    "Sleeve":  (5, (0.018, 0.021, 0.030), 0.0, 0.84),   # navy-black softshell
}
NYLON = ((0.025, 0.025, 0.027), 0.80)
SUEDE = ((0.062, 0.058, 0.052), 0.90)
GUARD = ((0.018, 0.018, 0.019), 0.48)
THREAD = (0.11, 0.11, 0.10)
# Sleeve seams (flat-felled, two rows of topstitching) and the elastic hem band, in metres.
SEAM_HALF = 0.0040
STITCH_OFF = 0.0026
STITCH_PITCH = 0.0040
HEM_BAND = 0.034


def rot(axis, angle):
    return np.array(Matrix.Rotation(angle, 3, Vector(axis)))


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


# ----------------------------------------------------------------------------- collision
class Collider:
    """Closed-mesh proxy: a point is in contact when it is inside or closer than `margin`."""

    def __init__(self, verts, faces):
        self.bvh = BVHTree.FromPolygons([Vector(v) for v in verts], faces, all_triangles=False, epsilon=0.0)

    def signed(self, p):
        loc, nrm, _, dist = self.bvh.find_nearest(Vector(p), 0.08)
        if loc is None:
            return 1.0
        return dist if (Vector(p) - loc).dot(nrm) >= 0 else -dist

    def contact(self, pts, margin):
        for p in pts:
            if self.signed(p) < margin:
                return True
        return False

    def depth(self, pts, margin):
        """Worst penetration below the margin (0 when clear)."""
        worst = 0.0
        for p in pts:
            worst = max(worst, margin - self.signed(p))
        return worst


def mesh_arrays(obj):
    me = obj.data
    M = obj.matrix_world
    return [tuple(M @ v.co) for v in me.vertices], [tuple(p.vertices) for p in me.polygons]


def merged_collider(objs, extra=()):
    verts, faces = [], []
    for o in objs:
        v, f = mesh_arrays(o)
        base = len(verts)
        verts += v
        faces += [tuple(i + base for i in p) for p in f]
    for v, f in extra:
        base = len(verts)
        verts += [tuple(x) for x in v]
        faces += [tuple(i + base for i in p) for p in f]
    return Collider(verts, faces)


# ----------------------------------------------------------------------------- hand rig
class Hand:
    def __init__(self, side, scale=0.94):
        self.side = side                                   # +1 right hand, -1 left hand
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=os.path.join(ASSETS, "right.glb" if side > 0 else "left.glb"))
        new = [o for o in bpy.data.objects if o not in before]
        arm = next(o for o in new if o.type == "ARMATURE")
        mesh = next(o for o in new if o.type == "MESH" and len(o.vertex_groups))
        me = mesh.data
        M = mesh.matrix_world
        self.rest = np.array([tuple(M @ v.co) for v in me.vertices])
        N3 = M.to_3x3()
        self.rest_n = np.array([tuple((N3 @ v.normal).normalized()) for v in me.vertices])
        self.faces = [tuple(p.vertices) for p in me.polygons]
        self.joints = [b.name for b in arm.data.bones]
        self.jpos = {b.name: np.array(tuple(arm.matrix_world @ b.head_local)) for b in arm.data.bones}
        W = np.zeros((len(me.vertices), len(self.joints)))
        gmap = {g.index: self.joints.index(g.name) for g in mesh.vertex_groups if g.name in self.joints}
        for v in me.vertices:
            for g in v.groups:
                if g.group in gmap:
                    W[v.index, gmap[g.group]] += g.weight
        W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
        # glTF splits vertices along UV seams; weld them so the hand is one closed surface.
        key = np.round(self.rest / 1e-6).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        n_u = len(first)
        cnt = np.bincount(inv, minlength=n_u).astype(float)
        Wu = np.zeros((n_u, W.shape[1]))
        np.add.at(Wu, inv, W)
        Nu = np.zeros((n_u, 3))
        np.add.at(Nu, inv, self.rest_n)
        self.rest = self.rest[first]
        self.W = Wu / cnt[:, None]
        self.rest_n = Nu / np.maximum(np.linalg.norm(Nu, axis=1, keepdims=True), 1e-9)
        faces = []
        for f in self.faces:
            g = tuple(int(inv[i]) for i in f)
            if len(set(g)) == len(g):
                faces.append(g)
        self.faces = faces
        print(f"  hand mesh: {len(me.vertices)} -> {n_u} vertices after welding, {len(faces)} faces", flush=True)
        # The WebXR hand is a large adult hand; scale it to a medium glove size.
        self.rest *= scale
        self.jpos = {k: v * scale for k, v in self.jpos.items()}
        for o in new:
            data = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Armature):
                bpy.data.armatures.remove(data)

        self.parent = {"wrist": None}
        for chain in CHAIN.values():
            prev = "wrist"
            for j in chain:
                self.parent[j] = prev
                prev = j
        self.order = ["wrist"] + [j for ch in CHAIN.values() for j in ch]
        dom = self.W.argmax(1)
        self.seg = {j: np.nonzero(dom == k)[0] for k, j in enumerate(self.joints)}
        self.palm = np.array((-side, 0.0, 0.0))            # palm normal in the rest pose
        self.local = {j: np.eye(3) for j in self.joints}
        self.R = np.eye(3)
        self.T = np.zeros(3)

    # -- kinematics
    def globals(self):
        G, P = {}, {}
        for j in self.order:
            p = self.parent[j]
            if p is None:
                G[j] = self.R @ self.local[j]
                P[j] = self.R @ self.jpos[j] + self.T
            else:
                G[j] = G[p] @ self.local[j]
                P[j] = P[p] + G[p] @ (self.jpos[j] - self.jpos[p])
        return G, P

    def skin(self, idx=None):
        G, P = self.globals()
        V = self.rest if idx is None else self.rest[idx]
        W = self.W if idx is None else self.W[idx]
        out = np.zeros_like(V)
        for k, j in enumerate(self.joints):
            w = W[:, k]
            nz = w > 1e-6
            if nz.any():
                out[nz] += w[nz, None] * ((V[nz] - self.jpos[j]) @ G[j].T + P[j])
        return out

    def child(self, j):
        for ch in CHAIN.values():
            if j in ch and ch.index(j) + 1 < len(ch):
                return ch[ch.index(j) + 1]
        return None

    def flex_axis(self, j):
        d = unit(self.jpos[self.child(j)] - self.jpos[j])
        return unit(np.cross(d, self.palm))

    # -- placement
    def place(self, h, n, anchor_joint, target):
        """Rotate the rest hand so its axis (wrist -> middle knuckle) points along h and its palm
        faces n, then move `anchor_joint` onto `target`."""
        h0 = unit(self.jpos["middle-finger-phalanx-proximal"] - self.jpos["wrist"])
        n0 = unit(self.palm - h0 * np.dot(self.palm, h0))
        B0 = np.stack([h0, n0, np.cross(h0, n0)], 1)
        h = unit(h)
        n = unit(np.asarray(n) - h * np.dot(n, h))
        B1 = np.stack([h, n, np.cross(h, n)], 1)
        self.R = B1 @ B0.T
        self.T = np.zeros(3)
        _, P = self.globals()
        self.T = np.asarray(target) - P[anchor_joint]

    def fit(self, targets, weights=None):
        """Rigid placement that best maps joint positions onto targets (weighted Kabsch)."""
        names = list(targets)
        A = np.array([self.jpos[n] for n in names])
        B = np.array([targets[n] for n in names], float)
        w = np.ones(len(names)) if weights is None else np.array([weights.get(n, 1.0) for n in names])
        ca = (A * w[:, None]).sum(0) / w.sum()
        cb = (B * w[:, None]).sum(0) / w.sum()
        H = ((A - ca) * w[:, None]).T @ (B - cb)
        U, _, Vt = np.linalg.svd(H)
        dfix = np.sign(np.linalg.det(Vt.T @ U.T))
        self.R = Vt.T @ np.diag([1, 1, dfix]) @ U.T
        self.T = cb - self.R @ ca
        res = np.linalg.norm((A @ self.R.T + self.T) - B, axis=1)
        return {n: round(float(r) * 1000, 1) for n, r in zip(names, res)}

    def approach(self, col, direction, segments, margin=GLOVE + 0.0004, step=0.0004, max_dist=0.05):
        """Slide the whole hand along `direction` until the palm touches (grasp-planner approach)."""
        idx = np.concatenate([self.seg[j] for j in segments])
        d = unit(direction)
        n = 0
        while col.contact(self.skin(idx), margin) and n < 100:      # start clear
            self.T = self.T - d * step
            n += 1
        moved = 0.0
        while moved < max_dist:
            self.T = self.T + d * step
            if col.contact(self.skin(idx), margin):
                self.T = self.T - d * step
                break
            moved += step
        return moved

    # -- grasp
    def grasp(self, finger, col, margin=GLOVE + 0.0004, rates=(1.0, 1.3, 0.9), max_deg=(88, 105, 80), step=1.2, spread=0.0):
        chain = CHAIN[finger]
        joints = chain[1:4]
        segs = [self.seg[joints[0]], self.seg[joints[1]], np.concatenate([self.seg[joints[2]], self.seg[chain[4]]])]
        axes = [self.flex_axis(j) for j in joints]
        if spread:
            # abduction: rotate the proximal phalanx about the palm normal
            self.local[joints[0]] = rot(self.palm, math.radians(spread))
        base = self.local[joints[0]].copy()
        ang = [0.0, 0.0, 0.0]
        active = [True, True, True]
        idx = np.concatenate(segs)
        bounds = np.cumsum([0] + [len(s) for s in segs])

        def apply(a):
            self.local[joints[0]] = base @ rot(axes[0], math.radians(a[0]))
            self.local[joints[1]] = rot(axes[1], math.radians(a[1]))
            self.local[joints[2]] = rot(axes[2], math.radians(a[2]))

        for _ in range(600):
            if not any(active):
                break
            prev = list(ang)
            for i in range(3):
                if active[i]:
                    ang[i] = min(max_deg[i], ang[i] + step * rates[i])
                    if ang[i] >= max_deg[i]:
                        active[i] = False
            apply(ang)
            V = self.skin(idx)
            hit = -1
            for s in range(3):
                if col.contact(V[bounds[s]:bounds[s + 1]], margin):
                    hit = s
            if hit >= 0:
                for i in range(hit + 1):
                    ang[i] = prev[i]
                    active[i] = False
                apply(ang)
        return ang

    # -- thumb IK
    def ccd(self, joints, end, target, iters=40, damping=0.5, fixed=None):
        for j, a in (fixed or {}).items():
            self.local[j] = rot(self.flex_axis(j), math.radians(a))
        target = np.asarray(target)
        for _ in range(iters):
            for j in reversed(joints):
                G, P = self.globals()
                a = unit(P[end] - P[j])
                b = unit(target - P[j])
                ax = np.cross(a, b)
                s = np.linalg.norm(ax)
                if s < 1e-7:
                    continue
                ang = math.atan2(s, float(np.dot(a, b))) * damping
                Gp = G[self.parent[j]]
                self.local[j] = Gp.T @ rot(ax / s, ang) @ Gp @ self.local[j]
        _, P = self.globals()
        return float(np.linalg.norm(P[end] - target))

    def region_fields(self):
        """Smooth per-vertex fields painted on the rest mesh (they survive subdivision and are
        baked to texels, so region borders are clean curves rather than face staircases):
        R = palm facing (0.5 + 0.5 cos), G = knuckle-guard band, B = fingertip caps."""
        V, n = self.rest, self.rest_n
        facing = n @ self.palm
        a, b = self.jpos[CHAIN["index"][1]], self.jpos[CHAIN["pinky"][1]]
        ab = b - a
        t = np.clip(((V - a) @ ab) / (ab @ ab), -0.15, 1.15)
        d_line = np.linalg.norm(V - (a + np.outer(t, ab)), axis=1)
        guard = np.exp(-(d_line / 0.0135) ** 2) * np.clip((-facing - 0.1) / 0.4, 0, 1)
        tips = np.zeros(len(V))
        for f in FINGERS + ("thumb",):
            ch = CHAIN[f]
            p0, p1 = self.jpos[ch[-2]], self.jpos[ch[-1]]
            seg = p1 - p0
            frac = ((V - p0) @ seg) / (seg @ seg)
            radial = np.linalg.norm((V - p0) - np.outer(frac, seg), axis=1)
            tips = np.maximum(tips, np.clip((frac - 0.15) / 0.3, 0, 1) * (radial < 0.014))
        return np.stack([0.5 + 0.5 * facing, guard, tips], 1)


# ----------------------------------------------------------------------------- pistol reference
GRIP = mp.GRIP
O = np.array((0.0, -0.0040, -0.0040))
D = np.array((0.0, -math.sin(GRIP), -math.cos(GRIP)))       # down the grip
F = np.array((0.0, math.cos(GRIP), -math.sin(GRIP)))        # out of the front strap
X = np.array((1.0, 0.0, 0.0))


def grip_point(t, side=0.0, front=0.0):
    return O + D * t + X * side + F * front


def pose_right(hand, col):
    # Knuckle line along the raked grip on the right panel, fingers pointing forward and a little
    # inward, palm facing the grip.
    res = hand.fit({
        "index-finger-phalanx-proximal": (0.029, -0.006, -0.021),
        "middle-finger-phalanx-proximal": (0.030, -0.013, -0.039),
        "pinky-finger-phalanx-proximal": (0.026, -0.028, -0.075),
        "wrist": (0.018, -0.100, -0.026),
    })
    print(f"  right fit residuals (mm) {res}", flush=True)
    n = hand.R @ hand.palm
    palm = ["wrist"] + [CHAIN[f][0] for f in FINGERS]
    moved = hand.approach(col, n, palm)
    print(f"  right palm approach {moved * 1000:.1f} mm", flush=True)
    for f, spread in (("index", 2.0), ("middle", 0.0), ("ring", -2.0), ("pinky", -5.0)):
        a = hand.grasp(f, col, spread=spread)
        print(f"  right {f:6s} curl {a[0]:5.1f} {a[1]:5.1f} {a[2]:5.1f}", flush=True)
    # High thumb along the left side of the frame, just under the slide.
    _, P = hand.globals()
    print("  right thumb joints", {j: np.round(P[j], 3).tolist() for j in CHAIN["thumb"]}, flush=True)
    tgt = np.array((-0.0205, 0.014, 0.0005))
    err = hand.ccd(CHAIN["thumb"][:3], "thumb-tip", tgt, fixed={"thumb-phalanx-distal": 10})
    thumb = np.concatenate([hand.seg[j] for j in CHAIN["thumb"]])
    push = 0
    while col.contact(hand.skin(thumb), GLOVE) and push < 12:
        push += 1
        err = hand.ccd(CHAIN["thumb"][:3], "thumb-tip", tgt + np.array((-0.0012, 0.0, 0.0004)) * push)
    _, P = hand.globals()
    print("  right thumb posed", {j: np.round(P[j], 3).tolist() for j in CHAIN["thumb"]}, flush=True)
    print(f"  right thumb error {err * 1000:.1f} mm, pushed {push}", flush=True)


# Support wrist: behind and below the knuckles, so the forearm runs back and down from a
# thumbs-forward grip instead of rising over the slide.
LEFT_WRIST = (-0.060, -0.062, -0.052)


def pose_left(hand, col):
    # Support hand: palm heel on the left panel, fingers wrapping over the firing hand's fingers.
    res = hand.fit({
        "index-finger-phalanx-proximal": (-0.034, 0.022, -0.040),
        "middle-finger-phalanx-proximal": (-0.036, 0.012, -0.058),
        "pinky-finger-phalanx-proximal": (-0.032, -0.010, -0.090),
        "wrist": LEFT_WRIST,
    })
    print(f"  left fit residuals (mm) {res}", flush=True)
    n = hand.R @ hand.palm
    palm = ["wrist"] + [CHAIN[f][0] for f in FINGERS]
    moved = hand.approach(col, n, palm)
    print(f"  left palm approach {moved * 1000:.1f} mm", flush=True)
    for f, spread in (("index", 0.0), ("middle", 0.0), ("ring", -2.0), ("pinky", -4.0)):
        a = hand.grasp(f, col, spread=spread)
        print(f"  left  {f:6s} curl {a[0]:5.1f} {a[1]:5.1f} {a[2]:5.1f}", flush=True)
    err = hand.ccd(CHAIN["thumb"][:2], "thumb-tip", (-0.0205, 0.062, -0.0130), fixed={"thumb-phalanx-distal": 5})
    thumb = np.concatenate([hand.seg[j] for j in CHAIN["thumb"]])
    push = 0
    while col.contact(hand.skin(thumb), GLOVE) and push < 12:
        push += 1
        err = hand.ccd(CHAIN["thumb"][:2], "thumb-tip", (-0.0205 - 0.0012 * push, 0.062, -0.0130 - 0.0004 * push))
    print(f"  left  thumb error {err * 1000:.1f} mm, pushed {push}", flush=True)


# ----------------------------------------------------------------------------- meshes
def hand_object(name, hand, mats):
    V = hand.skin()
    bm = bmesh.new()
    verts = [bm.verts.new(tuple(v)) for v in V]
    for f in hand.faces:
        try:
            bm.faces.new([verts[i] for i in f]).smooth = True
        except ValueError:
            continue
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    obj = hs.new_object(name, bm, mats["Glove"])
    fields = hand.region_fields()
    attr = obj.data.color_attributes.new("glove", "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", np.concatenate([fields, np.ones((len(fields), 1))], 1).astype(np.float32).ravel())
    return obj


def seam_profile(x, centre, width):
    return np.exp(-((x - centre) / width) ** 2)


def glove(obj, subdiv=1):
    """Smooth the base mesh, then inflate it by the glove thickness with a raised knuckle guard
    and recessed seams along the palm edge and around the guard."""
    mod = obj.modifiers.new("Subd", "SUBSURF")
    mod.levels = mod.render_levels = subdiv
    hs.apply_all(obj)
    me = obj.data
    f = np.empty(len(me.vertices) * 4, np.float32)
    me.color_attributes["glove"].data.foreach_get("color", f)
    f = f.reshape(-1, 4)
    facing = f[:, 0] * 2 - 1
    guard = tb.smoothstep(0.35, 0.6, f[:, 1])
    off = (GLOVE + 0.0016 * guard - 0.00045 * seam_profile(facing, 0.25, 0.07)
           - 0.0003 * seam_profile(f[:, 1], 0.33, 0.05))
    nrm = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("normal", nrm)
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) + nrm.reshape(-1, 3) * off[:, None]
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    me.shade_smooth()
    return obj


def wrist_ring(hand):
    """Centre, forearm axis, major axis and radii of the posed hand's wrist end."""
    V = hand.skin()
    G, P = hand.globals()
    axis = unit(P["wrist"] - P["middle-finger-phalanx-proximal"])   # pointing up the forearm
    s = (V - P["wrist"]) @ axis
    pts = V[s > s.max() - 0.010]
    ctr = pts.mean(0)
    q = (pts - ctr) - np.outer((pts - ctr) @ axis, axis)
    evals, evecs = np.linalg.eigh(q.T @ q)
    major = evecs[:, 2]
    minor = np.cross(axis, major)
    rx = float(np.abs(q @ major).max())
    ry = float(np.abs(q @ minor).max())
    return ctr, axis, major, rx, ry


def tube(name, path, radii, mat, n=40, up=(0, 0, 1), folds=None, cap=False):
    """Loft of elliptical rings along a polyline. radii: (rx along `up`, ry across) per point."""
    rings = []
    up = np.asarray(up, float)
    for i, p in enumerate(path):
        p = np.asarray(p, float)
        t = unit(path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)])
        u = unit(up - t * np.dot(up, t))
        v = np.cross(t, u)
        rx, ry = radii[i]
        ring = []
        for k in range(n):
            a = 2 * math.pi * k / n
            r = 1.0 + (folds(i, a) if folds else 0.0)
            ring.append(tuple(p + (u * math.cos(a) * rx + v * math.sin(a) * ry) * r))
        rings.append(ring)
    obj = hs.loft(name, rings, mat, cap_start=cap, cap_end=cap)
    obj.data.shade_smooth()
    return obj


# ----------------------------------------------------------------------------- arm rig
# src/game/arms.ts poses the arms every frame with two-bone IK: the shoulders stay put beside the
# chest camera and the hands stay on the pistol, so the elbows follow the gun as it is raised,
# lowered, canted and kicked back. The mesh is built and skinned in the rest pose below, the aim
# pose, and the numbers here must match arms.ts and weapon.ts.
REST_VIEW = (0.0, -0.0488, -0.50)     # pistol origin in camera space (three.js axes) when aiming
SHOULDER = (0.185, -0.03, 0.10)       # right shoulder joint, camera space; the left mirrors x
POLE = (1.0, -0.6, 0.2)               # the way the right elbow points, camera space; the left mirrors x
UPPER, FORE = 0.31, 0.25              # shoulder to elbow, elbow to wrist (m)
WRIST_IN = 0.010                      # the wrist joint sits this far up the forearm from the glove opening
HEM = 0.026                           # the jacket sleeve starts this far up the glove cuff


def cam_point(p):
    """Camera space (three.js axes) -> pistol space (Blender axes), in the rest pose."""
    x, y, z = np.asarray(p, float) - np.asarray(REST_VIEW)
    return np.array((x, -z, y))


def cam_dir(d):
    x, y, z = d
    return unit(np.array((x, -z, y), float))


def reach(S, W, a, b):
    """When the hand is out of reach the shoulder comes forward along the arm, as in arms.ts."""
    d = W - S
    L = float(np.linalg.norm(d))
    lim = a + b - 0.002
    return W - d / L * lim if L > lim else S


def elbow_ik(S, W, a, b, pole):
    """Two-bone IK: the elbow on the side of `pole`, upper arm a, forearm b."""
    d = W - S
    L = float(np.linalg.norm(d))
    u = d / L
    ca = float(np.clip((a * a + L * L - b * b) / (2 * a * L), -1, 1))
    v = unit(pole - u * (pole @ u))
    return S + a * (ca * u + math.sqrt(1 - ca * ca) * v)


def arm_joints(hand):
    ctr, axis, major, rx, ry = wrist_ring(hand)
    m = np.array((hand.side, 1, 1), float)
    W = ctr + axis * WRIST_IN
    S = reach(cam_point(np.array(SHOULDER) * m), W, UPPER, FORE)
    E = elbow_ik(S, W, UPPER, FORE, cam_dir(np.array(POLE) * m))
    return S, E, W


def cuff_frame(hand, E, W, max_bend=22.0):
    """The glove gauntlet leaves the hand along the hand's axis and bends (at most max_bend
    degrees) towards the forearm, so the wrist bend is shared between cuff and sleeve.
    Returns (point, tangent) at arc length s from the glove opening."""
    ctr, axis, *_ = wrist_ring(hand)
    to_elbow = unit(E - W)
    ang = math.degrees(math.acos(float(np.clip(axis @ to_elbow, -1, 1))))
    fore = slerp(axis, to_elbow, min(1.0, max_bend / max(ang, 1e-6)))
    start = ctr - axis * 0.004

    def along(s):
        n = 24
        p, t = start.copy(), axis
        for i in range(n):
            si = s * (i + 0.5) / n
            t = slerp(axis, fore, float(np.clip((si - 0.006) / 0.024, 0, 1)))
            p = p + t * (s / n)
        return p, t

    return along


def slerp(a, b, f):
    a, b = unit(a), unit(b)
    ang = math.acos(float(np.clip(a @ b, -1, 1)))
    if ang < 1e-6:
        return a
    return unit((math.sin((1 - f) * ang) * a + math.sin(f * ang) * b) / math.sin(ang))


def glove_cuff(hand, along, mats):
    """The glove's gauntlet and velcro strap, rigid with the hand."""
    ctr, axis, major, rx, ry = wrist_ring(hand)
    cx, cy = rx + GLOVE + 0.0022, ry + GLOVE + 0.0022
    cuff = tube("cuff", [along(t)[0] for t in np.linspace(0, 0.044, 12)], [(cx, cy)] * 12, mats["Cuff"], n=40, up=major)
    strap = tube("strap", [along(t)[0] for t in np.linspace(0.010, 0.030, 6)], [(cx + 0.0022, cy + 0.0022)] * 6,
                 mats["Rubber"], n=40, up=major, cap=True)
    return [cuff, strap], max(cx, cy)


def _resample(P, spacing):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0.0, s[-1], spacing)
    t = np.append(t, s[-1]) if s[-1] - t[-1] > spacing * 0.3 else np.append(t[:-1], s[-1])
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1), t


def sleeve_centreline(along, S, E, W):
    """Hem -> wrist bend -> forearm -> rounded elbow -> upper arm -> into the shoulder. Returns
    points 2 mm apart and the arc length of the elbow and shoulder along it."""
    h0, t0 = along(HEM + 0.004)
    q = W + (E - W) * 0.30
    ctrl = h0 + t0 * 0.045
    bez = [(1 - t) ** 2 * h0 + 2 * (1 - t) * t * ctrl + t * t * q for t in np.linspace(0, 1, 60)]
    fore = [q + (E - q) * t for t in np.linspace(0, 1, 120)[1:]]
    upper = [E + (S - E) * t for t in np.linspace(0, 1, 120)[1:]]
    tail = [S + unit(S - E) * 0.07 * t for t in np.linspace(0, 1, 20)[1:]]
    P, s = _resample(np.array(bez + fore + upper + tail), 0.002)
    # Round the elbow like a sleeve over a bent arm (the hem end stays exact).
    k = 11                                                     # ~2 cm Gaussian
    w = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2)
    w /= w.sum()
    pad = np.concatenate([np.repeat(P[:1], 3 * k, 0), P, np.repeat(P[-1:], 3 * k, 0)])
    Ps = np.stack([np.convolve(pad[:, j], w, mode="valid") for j in range(3)], 1)
    keep = np.clip((s - 0.03) / 0.05, 0, 1)[:, None]
    P = P * (1 - keep) + Ps * keep
    P, s = _resample(P, 0.002)
    s_e = s[np.argmin(np.linalg.norm(P - E, axis=1))]
    s_s = s[np.argmin(np.linalg.norm(P - S, axis=1))]
    s_q = s[np.argmin(np.linalg.norm(P - q, axis=1))]
    return P, s, s_q, s_e, s_s


def sleeve_mesh(hand, along, S, E, W, cuff_r, mats, seed):
    """Softshell jacket sleeve from the glove cuff to the shoulder: tight hem band, fabric bunched
    above the wrist, shallow twist folds down the forearm, folds in the crook of the elbow, a
    looser upper arm. Returns the object and per-vertex bone weights."""
    rng = np.random.default_rng(seed)
    side = hand.side
    ctr, axis, major, *_ = wrist_ring(hand)
    P, s, s_q, s_e, s_s = sleeve_centreline(along, S, E, W)
    # Rings: dense where the folds are, sparser up the upper arm.
    step = np.where((s < 0.16) | (np.abs(s - s_e) < 0.09), 0.0025, 0.006)
    idx = [0]
    while s[idx[-1]] < s[-1] - 1e-9:
        nxt = np.searchsorted(s, s[idx[-1]] + step[idx[-1]])
        idx.append(min(nxt, len(s) - 1))
    idx = np.array(sorted(set(idx)))
    C, sr = P[idx], s[idx]
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    # parallel-transported frames from the glove's major axis
    Nn = np.zeros_like(C)
    n = unit(major - T[0] * (major @ T[0]))
    for i in range(len(C)):
        n = unit(n - T[i] * (n @ T[i]))
        Nn[i] = n
    B = np.cross(T, Nn)
    # the crook of the elbow faces the bisector of the two arm segments
    inside = unit(unit(S - E) + unit(W - E))
    ie = int(np.argmin(np.abs(sr - s_e)))
    th_in = math.atan2(inside @ B[ie], inside @ Nn[ie])
    # two-piece sleeve: the front seam runs along the top of the forearm (thumb side, towards the
    # middle), the back seam over the point of the elbow
    seams = (unit(np.array((-0.5 * side, 0.0, 1.0))), unit(np.array((side, 0.0, -0.5))))
    ph = rng.uniform(0, 2 * math.pi, 16)

    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def radius(x):
        keys = [(0.0, cuff_r + 0.0042), (0.030, cuff_r + 0.0048), (0.056, 0.037), (0.12, 0.040),
                ((s_q + s_e) * 0.5, 0.042), (s_e, 0.046), ((s_e + s_s) * 0.5, 0.050), (s_s, 0.054), (sr[-1], 0.056)]
        xs, ys = zip(*sorted(keys))
        return float(np.interp(x, xs, ys))

    ridges = [(0.040, 0.0045, 0.17), (0.055, 0.0050, 0.21), (0.071, 0.0055, 0.19), (0.090, 0.0060, 0.15), (0.112, 0.0070, 0.10)]
    NA = 56
    th = np.arange(NA) * 2 * math.pi / NA
    rings, attrs, attrs2, attrs3, weights = [], [], [], [], []
    for i, (c, t, nn, bb, x) in enumerate(zip(C, T, Nn, B, sr)):
        r = radius(x)
        ell = 1.0 + 0.08 * (1 - tb.smoothstep(s_e - 0.06, s_e + 0.04, x))      # flatter forearm
        f = np.zeros(NA)
        ridge = np.zeros(NA)
        if x < 0.034:
            # hem band: a rolled edge, then a smooth elastic band
            f += 0.06 * tb.smoothstep(0.004, 0.0, x) * -1 + 0.02 * np.sin(th * 7 + ph[0]) * tb.smoothstep(0.0, 0.01, x)
        for k, (sk, wk, ak) in enumerate(ridges):
            pos = sk + 0.006 * np.sin(th + ph[k]) + 0.004 * np.sin(2 * th + ph[k + 5])
            amp = ak * (0.55 + 0.45 * np.sin(th + ph[k + 10]))
            g = np.exp(-((x - pos) / wk) ** 2)
            f += amp * g
            ridge = np.maximum(ridge, g * amp / 0.2)
        # long, shallow twist folds down the forearm
        win = tb.smoothstep(0.10, 0.16, x) * tb.smoothstep(s_e - 0.02, s_e - 0.08, x)
        for k in range(3):
            a0 = ph[k] + (x - 0.1) * (7.0 if k % 2 else -5.0)
            g = np.exp(-(np.array([wrap(a - a0) for a in th]) / 0.30) ** 2) * win
            f += 0.06 * g
            ridge = np.maximum(ridge, g * 0.4)
        # the crook of the elbow: short folds fanning out from the inside, stretched smooth outside
        d_in = np.array([wrap(a - th_in) for a in th])
        crook = np.exp(-(d_in / 1.0) ** 2)
        for k in range(4):
            sk = s_e + (-0.045 + 0.030 * k) + 0.012 * d_in * (k - 1.5) / 1.5
            g = np.exp(-((x - sk) / 0.0075) ** 2) * crook
            f += 0.14 * g
            ridge = np.maximum(ridge, g)
        # upper arm: a couple of soft sags under the arm
        sag = np.exp(-(np.array([wrap(a - th_in - math.pi * 0.35) for a in th]) / 0.8) ** 2)
        for k, sk in enumerate((s_e + 0.07, s_e + 0.13)):
            f += 0.05 * sag * np.exp(-((x - sk) / 0.02) ** 2)
        # general lumpiness
        f += 0.025 * np.sin(th * 3 + x * 23 + ph[3]) * np.sin(x * 41 + ph[4])
        ring = c + (np.outer(np.cos(th) * r * ell, nn) + np.outer(np.sin(th) * r, bb)) * (1 + f)[:, None]
        rings.append([tuple(p) for p in ring])
        # Where the seams are, for the material and the texture bake: the angle from each seam as
        # (cos, sin), which interpolates exactly across the 6 degree faces, the arc length from
        # the hem and the ring radius (x10).
        phi = [th - math.atan2(sd @ bb, sd @ nn) for sd in (unit(d - t * (d @ t)) for d in seams)]
        one = np.ones(NA)
        attrs.append(np.stack([np.cos(phi[0]), np.sin(phi[0]), x * one, one], 1))
        attrs2.append(np.stack([np.cos(phi[1]), np.sin(phi[1]), r * 10 * one, one], 1))
        attrs3.append(np.stack([np.clip(ridge, 0, 1), 0 * one, 0 * one, one], 1))
        # bone weights: hand -> wrist bend -> twist (forearm) -> fore (elbow end) -> upper arm
        h = 1 - tb.smoothstep(0.02, 0.07, x)
        tf = float(np.clip((x - s_q) / max(s_e - s_q, 1e-6), 0, 1))
        e = tb.smoothstep(s_e - 0.045, s_e + 0.045, x)
        wv = np.array([h, (1 - h) * (1 - e) * (1 - tf), (1 - h) * (1 - e) * tf, (1 - h) * e])
        weights.append(np.tile(wv / wv.sum(), (NA, 1)))
    # tuck the hem edge in against the glove cuff so there's no gap to see into
    t0 = T[0]
    inner = [tuple(C[0] - t0 * 0.001 + (np.cos(a) * Nn[0] + np.sin(a) * B[0]) * (cuff_r + 0.0008)) for a in th]
    rings.insert(0, inner)
    for a in (attrs, attrs2, attrs3, weights):
        a.insert(0, a[0])
    obj = hs.loft("sleeve", rings, mats["Sleeve"], cap_start=False, cap_end=True)
    obj.data.shade_smooth()
    # loft vertex order: ring by ring, then the end cap reuses the last ring's vertices
    Wt = np.concatenate(weights)
    nv = len(obj.data.vertices)
    for name, a in (("sleeve", attrs), ("sleeve2", attrs2), ("sleeve3", attrs3)):
        A = np.concatenate(a)
        assert nv == len(A), (nv, len(A))
        at = obj.data.color_attributes.new(name, "FLOAT_COLOR", "POINT")
        at.data.foreach_set("color", A.astype(np.float32).ravel())
    return obj, Wt


BONES = ("hand", "twist", "fore", "upper")


def add_weights(obj, sfx, weights=None):
    """Vertex groups for one arm; `weights` is (n, 4) in BONES order, or None for all-hand."""
    groups = {b: obj.vertex_groups.new(name=f"{b}_{sfx}") for b in BONES}
    n = len(obj.data.vertices)
    W = np.zeros((n, 4)) if weights is None else weights
    if weights is None:
        W[:, 0] = 1.0
    for k, b in enumerate(BONES):
        g = groups[b]
        for vi in np.nonzero(W[:, k] > 1e-4)[0]:
            g.add([int(vi)], float(W[vi, k]), "REPLACE")


def build_rig(joints):
    """Armature with an unparented chain per arm: upper, fore and twist (both elbow -> wrist;
    arms.ts twists the second one with the hand) and hand."""
    arm = bpy.data.armatures.new("ArmsRig")
    rig = bpy.data.objects.new("ArmsRig", arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for sfx, (S, E, W, hand_dir) in joints.items():
        for name, head, tail in (("upper", S, E), ("fore", E, W), ("twist", E, W), ("hand", W, W + hand_dir * 0.08)):
            b = arm.edit_bones.new(f"{name}_{sfx}")
            b.head, b.tail = Vector(tuple(head)), Vector(tuple(tail))
            b.use_deform = True
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


# ----------------------------------------------------------------------------- textures
def composite(ids, cover, aoe, pos, nrm, tnrm, bmin, bmax, fields, sleeve):
    h, w = ids.shape
    ao, edge, cav = aoe[..., 0], aoe[..., 1], aoe[..., 2]
    p01 = (pos - bmin) / (bmax - bmin).max()
    base = np.zeros((h, w, 3), np.float32)
    rough = np.zeros((h, w), np.float32)
    for name, (i, col, _mt, rg) in SURF.items():
        sel = ids == i
        base[sel] = col
        rough[sel] = rg
    # Glove panels from the painted fields: suede palm and fingertips, moulded guard, nylon back.
    g = ids == SURF["Glove"][0]
    facing = fields[..., 0] * 2 - 1
    suede = np.maximum(tb.smoothstep(0.20, 0.30, facing), tb.smoothstep(0.35, 0.55, fields[..., 2]))
    guard = tb.smoothstep(0.40, 0.48, fields[..., 1]) * (1 - suede)
    nylon = 1 - np.maximum(suede, guard)
    gb = nylon[..., None] * np.array(NYLON[0], np.float32) + suede[..., None] * np.array(SUEDE[0], np.float32) + guard[..., None] * np.array(GUARD[0], np.float32)
    gr = nylon * NYLON[1] + suede * SUEDE[1] + guard * GUARD[1]
    # Stitching: a dashed thread line in a slight groove along each panel border.
    dash = (np.sin((pos @ np.array((1.3, 0.8, 1.1), np.float32)) * 2 * np.pi * 380) > 0.1).astype(np.float32)
    seam = np.maximum(seam_profile(facing, 0.25, 0.018), seam_profile(fields[..., 1], 0.40, 0.012))
    groove = np.maximum(seam_profile(facing, 0.25, 0.05), seam_profile(fields[..., 1], 0.40, 0.035))
    gb = gb * (1 - 0.45 * groove[..., None])
    gb = gb * (1 - (seam * dash)[..., None]) + np.array(THREAD, np.float32) * (seam * dash)[..., None]
    base[g] = gb[g]
    rough[g] = gr[g]
    # Sleeves: seams with darker grooves either side, faintly lighter thread, a slightly darker
    # and smoother hem band, and fabric faded along the tops of the folds.
    sl = ids == SURF["Sleeve"][0]
    seam, sgroove, stitch, band = sleeve_fields(sleeve[0], sleeve[1])
    fold = sleeve[2][..., 0]
    sb = base * (1 - 0.40 * sgroove[..., None]) * (1 - 0.12 * band[..., None]) * (1 + 0.30 * fold[..., None])
    sb = sb * (1 - 0.6 * stitch[..., None]) + (sb * 1.9 + 0.004) * (0.6 * stitch[..., None])
    sr = rough - 0.08 * band - 0.10 * stitch + 0.03 * fold
    base[sl] = sb[sl]
    rough[sl] = sr[sl]
    n_big = tb.fbm(p01, 5.0, 4, seed=11)
    n_mid = tb.fbm(p01, 30.0, 4, seed=12)
    n_fine = tb.fbm(p01, 160.0, 3, seed=13)
    crevice = tb.smoothstep(0.95, 0.6, cav)
    convex = np.clip(edge * 3.0, 0, 1) * tb.smoothstep(0.7, 0.97, cav)
    # Scuffed, faded high points (knuckles, fingertips, sleeve folds).
    scuff = tb.smoothstep(0.35, 0.8, convex * (0.5 + n_mid)) * 0.45
    base = base * (1 + scuff[..., None] * 0.8) + 0.006 * scuff[..., None]
    rough = rough + scuff * 0.05
    # Dust and dried dirt in creases.
    dust = crevice * tb.smoothstep(0.3, 0.7, n_mid * 0.6 + n_fine * 0.4)
    base = base * (1 - dust[..., None] * 0.6) + np.array((0.07, 0.064, 0.055), np.float32) * dust[..., None] * 0.6
    base *= (0.9 + 0.2 * n_big)[..., None]
    rough = np.clip(rough + (n_fine - 0.5) * 0.08, 0.3, 1.0)
    occ = np.clip(ao * (0.6 + 0.4 * cav), 0, 1)
    base *= (0.8 + 0.2 * cav)[..., None]
    base = tb.dilate(base, cover, 10)
    orm = tb.dilate(np.stack([occ, rough, np.zeros_like(rough)], -1), cover, 10)
    tn = tnrm.copy()
    tn[~cover] = (0.5, 0.5, 1.0)
    tn = tb.dilate(tn, cover, 10)
    return tb.linear_to_srgb(base), orm, tn


def materials():
    m = {}
    for name, (_id, col, metal, rough) in SURF.items():
        m[name] = c.material(name, col, roughness=rough, metallic=metal)
    # Suede nap, nylon and softshell crinkle for the normal bake. Features stay >1 mm so the
    # ~0.3 mm atlas texels resolve them (a finer weave only turns into moire).
    for name, scale, strength, distortion in (("Glove", 900.0, 0.25, 0.0), ("Cuff", 700.0, 0.25, 0.0), ("Sleeve", 420.0, 0.35, 1.5)):
        nt = m[name].node_tree
        tc = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = scale
        noise.inputs["Detail"].default_value = 3.0
        noise.inputs["Distortion"].default_value = distortion
        nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = strength
        bump.inputs["Distance"].default_value = 0.0001
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], nt.nodes["Principled BSDF"].inputs["Normal"])
    sleeve_seams(m["Sleeve"])
    return m


def sleeve_seams(mat):
    """Seams, topstitching and the hem band as bump on the sleeve material, so the normal bake
    carries them; sleeve_fields() paints the same features into the colour map."""
    nt = mat.node_tree
    N, L = nt.nodes, nt.links

    def op(kind, a, b=None):
        n = N.new("ShaderNodeMath")
        n.operation = kind
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            elif v is not None:
                L.new(v, n.inputs[i])
        return n.outputs[0]

    def ramp(x, lo, hi):
        """smoothstep from lo to hi (falls when lo > hi)"""
        n = N.new("ShaderNodeMapRange")
        n.interpolation_type = "SMOOTHSTEP"
        L.new(x, n.inputs["Value"])
        n.inputs["From Min"].default_value = lo
        n.inputs["From Max"].default_value = hi
        return n.outputs["Result"]

    def line(x, at, half=0.0007):
        return ramp(op("ABSOLUTE", op("SUBTRACT", x, at)), half, 0.0)

    def dashes(x):
        return op("GREATER_THAN", op("FRACT", op("DIVIDE", x, STITCH_PITCH)), 0.35)

    def attr(name):
        a = N.new("ShaderNodeAttribute")
        a.attribute_name = name
        sep = N.new("ShaderNodeSeparateColor")
        L.new(a.outputs["Color"], sep.inputs["Color"])
        return sep.outputs["Red"], sep.outputs["Green"], sep.outputs["Blue"]

    c1, s1, arc = attr("sleeve")
    c2, s2, r10 = attr("sleeve2")
    r = op("DIVIDE", r10, 10.0)
    around = op("MULTIPLY", op("ARCTAN2", s1, c1), r)          # arc round the arm from the front seam
    d = op("MINIMUM", op("ABSOLUTE", around), op("MULTIPLY", op("ABSOLUTE", op("ARCTAN2", s2, c2)), r))
    seam = ramp(d, SEAM_HALF, SEAM_HALF - 0.0015)
    stitch = op("MULTIPLY", line(d, STITCH_OFF), dashes(arc))
    band = ramp(arc, HEM_BAND + 0.002, HEM_BAND - 0.002)
    hem = op("MULTIPLY", op("MAXIMUM", line(arc, 0.006), line(arc, HEM_BAND - 0.005)), dashes(around))
    h = op("ADD", seam, op("MULTIPLY", band, 0.7))
    h = op("SUBTRACT", h, op("MULTIPLY", op("MAXIMUM", stitch, hem), 0.9))
    bsdf = N["Principled BSDF"]
    crinkle = bsdf.inputs["Normal"].links[0].from_socket
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = 0.0004
    L.new(h, bump.inputs["Height"])
    L.new(crinkle, bump.inputs["Normal"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def sleeve_fields(a1, a2):
    """The same seam features as sleeve_seams(), from the baked attributes: (seam band, groove
    along its edges, stitches, hem band)."""
    r = a2[..., 2] / 10
    around = np.arctan2(a1[..., 1], a1[..., 0]) * r
    arc = a1[..., 2]
    d = np.minimum(np.abs(around), np.abs(np.arctan2(a2[..., 1], a2[..., 0])) * r)

    def line(x, at, half=0.0007):
        return tb.smoothstep(half, 0.0, np.abs(x - at))

    def dashes(x):
        return (np.mod(x / STITCH_PITCH, 1.0) > 0.35).astype(np.float32)

    seam = tb.smoothstep(SEAM_HALF, SEAM_HALF - 0.0015, d)
    groove = line(d, SEAM_HALF, 0.0012)
    stitch = np.maximum(line(d, STITCH_OFF) * dashes(arc),
                        np.maximum(line(arc, 0.006), line(arc, HEM_BAND - 0.005)) * dashes(around))
    band = tb.smoothstep(HEM_BAND + 0.002, HEM_BAND - 0.002, arc)
    return seam, groove, stitch, band


def bake_textures(objs, out, size, density=None):
    t0 = time.time()
    tb.setup(samples=16)
    info = tb.atlas_uvs(objs, resolution=size, padding=max(4, size // 256), density=density)
    print(f"atlas: {info}", flush=True)
    corners = np.array([o.matrix_world @ Vector(v) for o in objs for v in o.bound_box])
    bmin, bmax = corners.min(0).astype(np.float32) - 0.001, corners.max(0).astype(np.float32) + 0.001
    ids, cover = tb.bake_ids(objs, size, {n: v[0] for n, v in SURF.items()})
    aoe = tb.bake_ao_edge(objs, size // 2, samples=24, ao_dist=0.04, cavity_dist=0.004, bevel_radius=0.0012)
    aoe = tb.upsample(tb.dilate(aoe, tb.upsample(cover.astype(np.float32), size // 2) > 0.5, 4), size)
    pos, nrm = tb.bake_position_normal(objs, size, bmin, bmax)
    fields = tb.bake_attribute(objs, size, "glove")
    sleeve = [tb.bake_attribute(objs, size, n) for n in ("sleeve", "sleeve2", "sleeve3")]
    tnrm = tb.bake_tangent_normal(objs, size, samples=6)
    base, orm, tn = composite(ids, cover, aoe, pos, nrm, tnrm, bmin, bmax, fields, sleeve)
    paths = {k: os.path.join(out, f"arms_{k}.jpg") for k in ("basecolor", "orm", "normal")}
    tb.save(base, paths["basecolor"], 90)
    tb.save(orm, paths["orm"], 92)
    tb.save(tn, paths["normal"], 95)
    print(f"textures in {time.time() - t0:.1f}s", flush=True)
    return paths


# Viewmodel offsets from src/game/weapon.ts: pistol origin position in camera space and
# Euler rotation (x, y, z; order YXZ), both in three.js axes.
VIEWMODEL = {
    "hip": ((0.075, -0.15, -0.37), (0.02, 0.14, -0.1)),
    "ads": (REST_VIEW, (0.0, 0.0, 0.0)),
    "ads_zoom": (REST_VIEW, (0.0, 0.0, 0.0)),        # the same, 36 degree lens: the wrists up close
}


def debug_views(out_dir, root, names):
    """Quick low-sample orthographic-ish views for checking the pose."""
    sc = bpy.context.scene
    tb.setup(samples=12)
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = 800, 600
    sc.view_settings.view_transform = "Standard"
    world = sc.world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.36, 0.38, 1)
    for i, (loc, e) in enumerate((((0.3, -0.2, 0.3), 8.0), ((-0.3, 0.1, 0.2), 5.0), ((0.0, 0.2, -0.3), 3.0))):
        L = bpy.data.objects.new(f"dl{i}", bpy.data.lights.new(f"dl{i}", "AREA"))
        L.data.energy, L.data.size = e, 0.5
        L.location = loc
        L.rotation_euler = (Vector((0, 0, -0.04)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(L)
    cam = bpy.data.objects.new("dcam", bpy.data.cameras.new("dcam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    shots = {
        "right": ((0.32, -0.02, -0.03), (0.0, -0.02, -0.03)),
        "left": ((-0.32, -0.02, -0.03), (0.0, -0.02, -0.03)),
        "front": ((0.10, 0.32, 0.05), (0.0, 0.0, -0.04)),
        "top": ((0.0, -0.04, 0.32), (0.0, -0.02, -0.03)),
        "back": ((0.05, -0.32, 0.10), (0.0, -0.02, -0.04)),
        "under": ((0.05, 0.10, -0.32), (0.0, 0.0, -0.04)),
        # whole arms, rest (aim) pose
        "arm_side": ((1.3, -0.30, 0.15), (0.0, -0.30, -0.02)),
        "arm_top": ((0.05, -0.32, 1.4), (0.0, -0.32, 0.0)),
        "arm_back": ((0.35, -1.3, 0.35), (0.0, -0.25, 0.0)),
    }
    for name in names:
        if name in VIEWMODEL:
            # The game camera, seen from the pistol (three.js axes, Euler order YXZ).
            (px, py, pz), (rx, ry, rz) = VIEWMODEL[name]
            R = Matrix.Rotation(ry, 4, "Y") @ Matrix.Rotation(rx, 4, "X") @ Matrix.Rotation(rz, 4, "Z")
            M = Matrix.Translation((px, py, pz)) @ R
            to_three = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
            cam.matrix_world = to_three.inverted() @ M.inverted()
            cam.data.sensor_fit = "VERTICAL"
            cam.data.angle = math.radians(36 if name.endswith("_zoom") else 78)
            sc.render.resolution_x, sc.render.resolution_y = 960, 540
            root.rotation_euler = (0, 0, 0)
        else:
            loc, tgt = shots[name]
            cam.location = loc
            cam.data.sensor_fit = "AUTO"
            cam.data.lens = 50
            cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
            root.rotation_euler = (0, 0, 0)
        cam.data.clip_start = 0.005
        sc.render.filepath = os.path.join(out_dir, f"{name}.png")
        bpy.ops.render.render(write_still=True)
    root.rotation_euler = (0, 0, 0)


# ----------------------------------------------------------------------------- main
def main():
    args = c.parse_args(lambda p: (p.add_argument("--size", type=int, default=2048),
                                   p.add_argument("--preview", default=""),
                                   p.add_argument("--no-bake", action="store_true"),
                                   p.add_argument("--no-sleeves", action="store_true"),
                                   p.add_argument("--views", default=""),
                                   p.add_argument("--vm", default="", help="name=px,py,pz,rx,ry,rz;... (experiments)"),
                                   p.add_argument("--lwrist", default="", help="x,y,z support wrist target (experiments)")))
    global LEFT_WRIST
    if args.lwrist:
        LEFT_WRIST = tuple(float(x) for x in args.lwrist.split(","))
    c.reset_scene()
    t0 = time.time()
    pm = mp.materials()
    frame = mp.build_frame(pm)
    slide = mp.build_slide(pm)
    mats = materials()

    right = Hand(+1)
    left = Hand(-1)
    col = merged_collider([frame, slide])
    print("posing firing hand", flush=True)
    pose_right(right, col)
    r_obj = hand_object("hand_r", right, mats)
    glove(r_obj)
    print("posing support hand", flush=True)
    col2 = merged_collider([frame, slide, r_obj])
    pose_left(left, col2)
    l_obj = hand_object("hand_l", left, mats)
    glove(l_obj)
    for nm, hnd in (("right", right), ("left", left)):
        ctr, axis, major, rx, ry = wrist_ring(hnd)
        print(f"  {nm} wrist ring centre {np.round(ctr, 4)} axis {np.round(axis, 3)} major {np.round(major, 3)} r {rx * 1000:.1f} x {ry * 1000:.1f} mm", flush=True)
    hand_parts, sleeves, joints = [], [], {}
    for sfx, hnd, obj, seed in (("R", right, r_obj, 1), ("L", left, l_obj, 2)):
        S, E, W = arm_joints(hnd)
        along = cuff_frame(hnd, E, W)
        cuff, cuff_r = glove_cuff(hnd, along, mats)
        for o in [obj] + cuff:
            add_weights(o, sfx)
        hand_parts += [obj] + cuff
        _, P = hnd.globals()
        joints[sfx] = (S, E, W, unit(P["middle-finger-phalanx-proximal"] - P["wrist"]))
        print(f"  {sfx} arm: shoulder {np.round(S, 3)} elbow {np.round(E, 3)} wrist {np.round(W, 3)}, "
              f"elbow bend {math.degrees(math.pi - math.acos(float(unit(S - E) @ unit(W - E)))):.0f} deg", flush=True)
        if not args.no_sleeves:
            sl, wts = sleeve_mesh(hnd, along, S, E, W, cuff_r, mats, seed)
            add_weights(sl, sfx, wts)
            sleeves.append(sl)
    hands = hs.merge("ArmsHands", hand_parts)
    objs = [hands] + ([hs.merge("ArmsSleeves", sleeves)] if sleeves else [])
    print(f"arms built in {time.time() - t0:.1f}s, {sum(len(p.vertices) - 2 for o in objs for p in o.data.polygons)} tris", flush=True)

    if not args.no_bake:
        # The sleeves are big but mostly seen at the edge of frame: fewer texels than the gloves.
        paths = bake_textures(objs, args.out, args.size, density={"ArmsSleeves": 0.55})
        pbr = tb.pbr_material("ArmsPBR", paths["basecolor"], paths["orm"], paths["normal"])
        # Cloth sheen (KHR_materials_sheen): the soft bright rim fabric gets at grazing angles.
        bsdf = pbr.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Sheen Weight"].default_value = 1.0
        # Kept faint: three.js adds it on top of the (dark) base, and more washes the fabric out.
        bsdf.inputs["Sheen Tint"].default_value = (0.05, 0.05, 0.055, 1.0)
        bsdf.inputs["Sheen Roughness"].default_value = 0.5
        for o in objs:
            o.data.materials.clear()
            o.data.materials.append(pbr)
            for p in o.data.polygons:
                p.material_index = 0
    arms = hs.merge("Arms", objs)
    rig = build_rig(joints)
    arms.parent = rig
    mod = arms.modifiers.new("Rig", "ARMATURE")
    mod.object = rig

    for item in filter(None, args.vm.split(";")):
        name, vals = item.split("=")
        v = [float(x) for x in vals.split(",")]
        VIEWMODEL[name] = (tuple(v[:3]), tuple(v[3:]))
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        root = c.empty("Pistol", (0, 0, 0))
        light, lens = mp.build_light(pm)
        for o in (frame, slide, light, lens, rig):
            o.parent = root
        if args.views:
            debug_views(args.preview, root, args.views.split(","))
        else:
            mp.preview(args.preview, root)
    for o in (frame, slide):
        bpy.data.objects.remove(o, do_unlink=True)
    for o in list(bpy.data.objects):
        if o.name.startswith(("WeaponLight", "LightLens", "Pistol")):
            bpy.data.objects.remove(o, do_unlink=True)
    # The colour attributes only feed the bake.
    c.export_glb(os.path.join(args.out, "arms.glb"), [arms, rig], export_vertex_color="NONE")


if __name__ == "__main__":
    main()
