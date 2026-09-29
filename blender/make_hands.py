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
    "Sleeve":  (5, (0.040, 0.042, 0.043), 0.0, 0.84),   # charcoal softshell
}
NYLON = ((0.030, 0.031, 0.032), 0.80)
SUEDE = ((0.062, 0.058, 0.052), 0.90)
GUARD = ((0.018, 0.018, 0.019), 0.48)
THREAD = (0.11, 0.11, 0.10)


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


def pose_left(hand, col):
    # Support hand: palm heel on the left panel, fingers wrapping over the firing hand's fingers.
    res = hand.fit({
        "index-finger-phalanx-proximal": (-0.034, 0.022, -0.040),
        "middle-finger-phalanx-proximal": (-0.036, 0.012, -0.058),
        "pinky-finger-phalanx-proximal": (-0.032, -0.010, -0.090),
        "wrist": (-0.058, -0.046, 0.005),
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


def slerp(a, b, f):
    a, b = unit(a), unit(b)
    ang = math.acos(float(np.clip(a @ b, -1, 1)))
    if ang < 1e-6:
        return a
    return unit((math.sin((1 - f) * ang) * a + math.sin(f * ang) * b) / math.sin(ang))


def sleeve_for(hand, elbow, mats, seed, max_bend=40.0):
    ctr, axis, major, rx, ry = wrist_ring(hand)
    rng = np.random.default_rng(seed)
    elbow = np.asarray(elbow, float)
    # The wrist flexes towards the elbow (at most max_bend degrees); the sleeve takes up the rest.
    to_elbow = unit(elbow - ctr)
    ang = math.degrees(math.acos(float(np.clip(axis @ to_elbow, -1, 1))))
    fore = slerp(axis, to_elbow, min(1.0, max_bend / max(ang, 1e-6)))
    start = ctr - axis * 0.004

    def along(s):
        """Centre and tangent at arc length s of the cuff, bending from the hand axis to the forearm."""
        n = 24
        p, t = start.copy(), axis
        for i in range(n):
            si = s * (i + 0.5) / n
            t = slerp(axis, fore, float(np.clip((si - 0.006) / 0.024, 0, 1)))
            p = p + t * (s / n)
        return p, t

    # glove cuff with a velcro strap, starting just inside the hand opening
    cuff_path = [along(s)[0] for s in np.linspace(0, 0.045, 10)]
    cx, cy = rx + GLOVE + 0.0022, ry + GLOVE + 0.0022
    cuff = tube("cuff", cuff_path, [(cx, cy)] * 10, mats["Cuff"], n=36, up=major)
    strap_path = [along(s)[0] for s in np.linspace(0.010, 0.032, 6)]
    strap = tube("strap", strap_path, [(cx + 0.0022, cy + 0.0022)] * 6, mats["Rubber"], n=36, up=major, cap=True)
    # jacket sleeve: leaves the cuff along the forearm, then sags towards the elbow;
    # loose, with compression folds near the wrist and a rolled hem.
    s0, t0 = along(0.020)
    ctrl = s0 + fore * 0.10
    L = np.linalg.norm(elbow - s0)
    steps = 26
    phase = rng.uniform(0, 6.28, 4)

    def bez(u):
        return (1 - u) ** 2 * s0 + 2 * (1 - u) * u * ctrl + u ** 2 * elbow

    def bez_t(u):
        return unit(2 * (1 - u) * (ctrl - s0) + 2 * u * (elbow - ctrl))

    def folds(u, a):
        ring = 0.05 * math.sin(u * 38 + phase[0]) * math.exp(-u * 2.2)
        wrinkle = 0.035 * math.sin(a * 3 + u * 9 + phase[1]) * math.sin(u * 22 + phase[2])
        return (ring + wrinkle) * min(1.0, u * 10)

    r0x, r0y = cx + 0.0032, cy + 0.0032
    spec = []   # (centre, tangent, rx, ry, fold u or None)
    for off, dr in ((0.010, -0.0026), (0.002, -0.0021), (-0.0006, -0.0010)):
        spec.append((s0 + t0 * off, t0, r0x + dr, r0y + dr, None))
    for i in range(steps + 1):
        u = i / steps
        grow = 0.016 * u ** 0.8
        spec.append((bez(u), bez_t(u), r0x + grow, r0y + grow * 1.1, u))
    loops = []
    for p, t, rx, ry, u in spec:
        uu = unit(major - t * np.dot(major, t))
        vv = np.cross(t, uu)
        ring = []
        for k in range(48):
            a = 2 * math.pi * k / 48
            r = 1.0 + (folds(u, a) if u is not None else 0.0)
            ring.append(tuple(p + (uu * math.cos(a) * rx + vv * math.sin(a) * ry) * r))
        loops.append(ring)
    sleeve = hs.loft("sleeve", loops, mats["Sleeve"], cap_start=False, cap_end=True)
    sleeve.data.shade_smooth()
    return [cuff, strap, sleeve], L


# ----------------------------------------------------------------------------- textures
def composite(ids, cover, aoe, pos, nrm, tnrm, bmin, bmax, fields):
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
    return m


def bake_textures(objs, out, size):
    t0 = time.time()
    tb.setup(samples=16)
    info = tb.atlas_uvs(objs, resolution=size, padding=max(4, size // 256))
    print(f"atlas: {info}", flush=True)
    corners = np.array([o.matrix_world @ Vector(v) for o in objs for v in o.bound_box])
    bmin, bmax = corners.min(0).astype(np.float32) - 0.001, corners.max(0).astype(np.float32) + 0.001
    ids, cover = tb.bake_ids(objs, size, {n: v[0] for n, v in SURF.items()})
    aoe = tb.bake_ao_edge(objs, size // 2, samples=24, ao_dist=0.04, cavity_dist=0.004, bevel_radius=0.0012)
    aoe = tb.upsample(tb.dilate(aoe, tb.upsample(cover.astype(np.float32), size // 2) > 0.5, 4), size)
    pos, nrm = tb.bake_position_normal(objs, size, bmin, bmax)
    fields = tb.bake_attribute(objs, size, "glove")
    tnrm = tb.bake_tangent_normal(objs, size, samples=6)
    base, orm, tn = composite(ids, cover, aoe, pos, nrm, tnrm, bmin, bmax, fields)
    paths = {k: os.path.join(out, f"arms_{k}.jpg") for k in ("basecolor", "orm", "normal")}
    tb.save(base, paths["basecolor"], 90)
    tb.save(orm, paths["orm"], 92)
    tb.save(tn, paths["normal"], 95)
    print(f"textures in {time.time() - t0:.1f}s", flush=True)
    return paths


# Elbow ends of the sleeves in pistol space (x right, y along the bore, z up).
ELBOW_R = (0.124, -0.291, -0.245)
ELBOW_L = (-0.192, -0.229, -0.210)

# Viewmodel offsets from src/game/weapon.ts: pistol origin position in camera space and
# Euler rotation (x, y, z; order YXZ), both in three.js axes.
VIEWMODEL = {
    "hip": ((0.075, -0.15, -0.37), (0.02, 0.14, -0.1)),
    "ads": ((0.0, -0.0488, -0.40), (0.0, 0.0, 0.0)),
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
            cam.data.angle = math.radians(78)
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
                                   p.add_argument("--elbows", default="", help="rx,ry,rz,lx,ly,lz (experiments)"),
                                   p.add_argument("--vm", default="", help="name=px,py,pz,rx,ry,rz;... (experiments)")))
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
    parts = [r_obj, l_obj]
    for nm, hnd in (("right", right), ("left", left)):
        ctr, axis, major, rx, ry = wrist_ring(hnd)
        print(f"  {nm} wrist ring centre {np.round(ctr, 4)} axis {np.round(axis, 3)} major {np.round(major, 3)} r {rx * 1000:.1f} x {ry * 1000:.1f} mm", flush=True)
    if not args.no_sleeves:
        er, el = ELBOW_R, ELBOW_L
        if args.elbows:
            v = [float(x) for x in args.elbows.split(",")]
            er, el = tuple(v[:3]), tuple(v[3:])
        sr, _ = sleeve_for(right, er, mats, 1)
        sl, _ = sleeve_for(left, el, mats, 2)
        parts += sr + sl
    arms = hs.merge("Arms", parts)
    print(f"arms built in {time.time() - t0:.1f}s, {sum(len(p.vertices) - 2 for p in arms.data.polygons)} tris", flush=True)

    if not args.no_bake:
        paths = bake_textures([arms], args.out, args.size)
        pbr = tb.pbr_material("ArmsPBR", paths["basecolor"], paths["orm"], paths["normal"])
        arms.data.materials.clear()
        arms.data.materials.append(pbr)
        for p in arms.data.polygons:
            p.material_index = 0

    for item in filter(None, args.vm.split(";")):
        name, vals = item.split("=")
        v = [float(x) for x in vals.split(",")]
        VIEWMODEL[name] = (tuple(v[:3]), tuple(v[3:]))
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        root = c.empty("Pistol", (0, 0, 0))
        light, lens = mp.build_light(pm)
        for o in (frame, slide, light, lens, arms):
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
    c.export_glb(os.path.join(args.out, "arms.glb"), [arms])


if __name__ == "__main__":
    main()
