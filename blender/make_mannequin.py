"""Training mannequin: a fibreglass shop-window figure pressed into range service.

    python3 blender/make_mannequin.py --out public/models [--size 2048] [--preview DIR]

The body is sculpted as a signed distance field (sdf.py): anatomical masses
(rib cage, pectorals, deltoids, quadriceps, calves...) are ellipsoids and
tapered capsules joined with smooth unions, eye sockets are smooth cuts, and
Surface Nets turns the field into a mesh that is then relaxed and decimated.
The mannequin wears a plate carrier; its satin paint shows the life of a range
target: duct-tape patches over old hits, marker tallies, chips, grime.

Conventions (src/game/targets.ts): origin between the feet, facing Blender -Y
(three.js +Z), about 1.85 m tall with the head centred near 1.72 m.
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

import common as c  # noqa: E402
import decals  # noqa: E402
import sdf  # noqa: E402
import texbake as tb  # noqa: E402

VOXEL = 0.0035


def axis_frame(d, roll_hint=(1, 0, 0)):
    """Rotation whose local z runs along d (for primitives aligned with a limb)."""
    z = np.asarray(d, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(roll_hint, float) - z * np.dot(roll_hint, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])


def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])


def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])


def lerp(a, b, t):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * t


def joints(s):
    """Shoulder, elbow, wrist, hip, knee and ankle centres of side s (+1 = the figure's left)."""
    return dict(S=np.array((s * 0.188, 0.010, 1.450)), E=np.array((s * 0.248, 0.034, 1.140)),
                W=np.array((s * 0.270, -0.022, 0.892)), H=np.array((s * 0.092, 0.006, 0.925)),
                K=np.array((s * 0.104, -0.008, 0.505)), A=np.array((s * 0.110, 0.020, 0.085)))


# Fingers of the relaxed hand: offset across the knuckles (towards the thumb), knuckle stagger
# along the hand, phalanx lengths, radii from knuckle to tip, and flexion at each joint (degrees).
FINGERS = (
    (0.028, 0.000, (0.044, 0.026, 0.021), (0.0092, 0.0084, 0.0077, 0.0068), (10, 20, 12)),    # index
    (0.009, 0.004, (0.048, 0.030, 0.023), (0.0095, 0.0087, 0.0079, 0.0070), (13, 24, 13)),    # middle
    (-0.010, 0.000, (0.045, 0.028, 0.022), (0.0090, 0.0082, 0.0075, 0.0066), (16, 28, 14)),   # ring
    (-0.027, -0.011, (0.036, 0.021, 0.019), (0.0080, 0.0073, 0.0067, 0.0059), (20, 32, 15)),  # little
)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def hand_frame(s, W, El):
    """d runs down the hand, fwd towards the thumb (the figure's front), pal out of the palm (towards the thigh)."""
    d = unit(W - El)
    fwd = np.array((0.0, -1.0, 0.0))
    fwd = unit(fwd - d * (fwd @ d))
    pal = unit(s * np.cross(d, fwd))
    return d, fwd, pal


def hand(f, s, W, El):
    """A relaxed hand hanging beside the thigh: palm towards the leg, fingers together, gently curled."""
    E, C = sdf.ellipsoid, sdf.cone
    d, fwd, pal = hand_frame(s, W, El)
    R = np.stack([pal, fwd, d], 1)
    f.add(sdf.rbox(W + d * 0.050 + pal * 0.001, (0.0125, 0.042, 0.046), 0.011, R), 0.014)             # palm
    f.add(E(W + d * 0.036 + fwd * 0.022 + pal * 0.008, (0.013, 0.017, 0.027), R), 0.008)             # thenar
    f.add(E(W + d * 0.048 - fwd * 0.027 + pal * 0.005, (0.010, 0.012, 0.030), R), 0.008)             # hypothenar
    for off, stagger, lengths, radii, flex in FINGERS:
        p = W + d * (0.093 + stagger) + fwd * off + pal * 0.001
        ang = 0.0
        for k in range(3):
            ang += math.radians(flex[k])
            q = p + (d * math.cos(ang) + pal * math.sin(ang)) * lengths[k]
            f.add(C(p, q, radii[k], radii[k + 1]), 0.006 if k == 0 else 0.0025)
            p = q
    # thumb: lies along the index finger, tip curled in towards the palm
    t0 = W + d * 0.020 + fwd * 0.024 + pal * 0.006
    t1 = W + d * 0.056 + fwd * 0.043 + pal * 0.016
    t2 = t1 + d * 0.027 + fwd * 0.004 + pal * 0.011
    t3 = t2 + d * 0.021 + fwd * 0.000 + pal * 0.009
    f.add(C(t0, t1, 0.0125, 0.0108), 0.008)
    f.add(C(t1, t2, 0.0108, 0.0097), 0.003)
    f.add(C(t2, t3, 0.0097, 0.0084), 0.0025)


# Face profile, chin to forehead: height, half width, and the most forward point of the section
# (the nose, lips' red, eyes and ears are added on top).
FACE = (
    (1.612, 0.032, -0.060),
    (1.620, 0.037, -0.079),
    (1.630, 0.040, -0.089),    # chin
    (1.641, 0.045, -0.088),
    (1.651, 0.050, -0.085),    # below the lower lip; the jaw angles
    (1.660, 0.053, -0.089),    # lower lip
    (1.670, 0.055, -0.090),    # upper lip
    (1.683, 0.058, -0.087),    # under the nose
    (1.698, 0.062, -0.083),
    (1.714, 0.066, -0.081),    # cheekbones
    (1.728, 0.068, -0.081),    # eyes
    (1.745, 0.068, -0.088),    # brow
    (1.762, 0.064, -0.086),
    (1.785, 0.056, -0.078),    # forehead
    (1.810, 0.042, -0.064),
    (1.830, 0.020, -0.040),
)


def head(f):
    """Head with adult male proportions (0.24 m chin to crown, 0.15 m wide). The face is
    a loft of measured sections; the finest features (mouth line, lids, nostrils, ear folds)
    are detail parts that only reach the normal map."""
    E, C = sdf.ellipsoid, sdf.cone
    f.add(E((0, 0.018, 1.752), (0.076, 0.099, 0.094)), 0.03)                                    # cranium
    f.add(sdf.loft(FACE, 0.004, 0.034, round_bottom=0.010), 0.02)                              # face
    for s in (-1, 1):
        f.add(E((s * 0.052, -0.046, 1.713), (0.018, 0.022, 0.014)), 0.012)                     # cheekbones
        f.add(E((s * 0.029, -0.082, 1.749), (0.025, 0.0075, 0.0065), rot_z(s * 0.18)), 0.016)   # brow ridge
    f.add(E((0, -0.085, 1.751), (0.013, 0.0075, 0.009)), 0.012)                                 # glabella
    f.add(C((0, -0.083, 1.733), (0, -0.103, 1.697), 0.0056, 0.0090), 0.008)                    # nose bridge
    f.add(E((0, -0.101, 1.693), (0.0105, 0.0100, 0.0095)), 0.006)                               # nose tip
    for s in (-1, 1):
        f.add(E((s * 0.0115, -0.090, 1.690), (0.0080, 0.0085, 0.0070)), 0.009)                 # nostril wings
    # the lips are one smooth mound on the grid (a groove between two lip shapes stair-steps
    # at 3.5 mm voxels); the two lips and the line between them are normal-map detail
    f.add(E((0, -0.0872, 1.6645), (0.0225, 0.0070, 0.0110)), 0.005)                             # mouth
    f.add(E((0, -0.0885, 1.6695), (0.0220, 0.0062, 0.0055)), 0.003, detail=True)               # upper lip
    f.add(E((0, -0.0880, 1.6590), (0.0195, 0.0062, 0.0055)), 0.003, detail=True)               # lower lip
    for s in (-1, 1):
        f.cut(E((s * 0.032, -0.092, 1.728), (0.019, 0.022, 0.0120)), 0.010)                    # eye sockets
    for s in (-1, 1):
        f.add(E((s * 0.032, -0.068, 1.728), (0.0125, 0.012, 0.0110)), 0.003)                  # eyes
        f.add(E((s * 0.075, 0.018, 1.722), (0.007, 0.019, 0.030), rot_z(s * 0.3) @ rot_x(-0.25)), 0.006)   # ears
    # --- detail parts (normal map only) ---
    f.cut(E((0, -0.096, 1.6640), (0.021, 0.012, 0.0016)), 0.002, detail=True)                   # mouth line
    for s in (-1, 1):
        f.cut(E((s * 0.0225, -0.086, 1.6640), (0.0035, 0.004, 0.003)), 0.003, detail=True)      # mouth corners
        f.cut(E((s * 0.007, -0.098, 1.6855), (0.0042, 0.006, 0.0022)), 0.0015, detail=True)     # nostrils
        f.cut(E((s * 0.032, -0.083, 1.7375), (0.016, 0.010, 0.0012), rot_x(-0.3)), 0.0012, detail=True)   # lid creases
        f.add(E((s * 0.032, -0.0775, 1.7335), (0.0135, 0.0045, 0.0030)), 0.002, detail=True)   # upper lids
        f.cut(E((s * 0.081, 0.020, 1.717), (0.004, 0.011, 0.016), rot_z(s * 0.3) @ rot_x(-0.25)), 0.003, detail=True)   # ear bowls
    f.cut(E((0, -0.0915, 1.679), (0.0035, 0.0022, 0.0055)), 0.003, detail=True)                 # philtrum


def torso(f):
    E = sdf.ellipsoid
    f.add(E((0, 0.012, 1.318), (0.155, 0.108, 0.200)))                          # rib cage
    f.add(E((0, 0.050, 1.350), (0.172, 0.066, 0.145)), 0.04)                    # lats / upper back
    f.add(E((0, -0.004, 1.462), (0.185, 0.072, 0.048)), 0.05)                   # clavicles / shoulder girdle
    for s in (-1, 1):
        f.add(E((s * 0.066, -0.060, 1.368), (0.078, 0.042, 0.060), rot_y(s * 0.25) @ rot_x(-0.15)), 0.03)   # pectorals
        f.add(E((s * 0.072, 0.084, 1.392), (0.062, 0.030, 0.072), rot_y(-s * 0.2)), 0.03)                   # shoulder blades
        f.add(E((s * 0.114, 0.004, 1.062), (0.054, 0.072, 0.080)), 0.045)                                  # obliques
    f.add(E((0, -0.008, 1.128), (0.138, 0.096, 0.150)), 0.05)                   # abdomen
    f.add(E((0, -0.066, 1.150), (0.074, 0.034, 0.120)), 0.03)                   # rectus abdominis
    f.add(E((0, 0.014, 0.968), (0.158, 0.104, 0.112)), 0.05)                    # pelvis
    for s in (-1, 1):
        f.add(E((s * 0.070, 0.060, 0.905), (0.084, 0.074, 0.100), rot_y(s * 0.15)), 0.03)                  # glutes
    f.add(E((0, -0.060, 0.862), (0.038, 0.034, 0.044)), 0.03)                   # groin (smooth shop-mannequin form)


def neck(f):
    C = sdf.cone
    f.add(C((0, 0.022, 1.500), (0, 0.012, 1.650), 0.063, 0.055), 0.03)                          # neck
    for s in (-1, 1):
        f.add(C((s * 0.030, 0.034, 1.560), (s * 0.168, 0.016, 1.482), 0.040, 0.034), 0.04)       # trapezius
        f.add(C((s * 0.046, 0.016, 1.650), (s * 0.014, -0.042, 1.500), 0.012, 0.011), 0.016)     # sternocleidomastoid


def body_field():
    """The figure as a signed distance field. +x is the figure's left, it faces -y."""
    f = sdf.Field((-0.34, -0.22, -0.005), (0.34, 0.18, 1.87), VOXEL)
    E, C = sdf.ellipsoid, sdf.cone
    torso(f)
    neck(f)
    head(f)

    # --- arms: hanging, elbows soft, hands turned in towards the thighs ---------------------
    for s in (-1, 1):
        j = joints(s)
        S, El, W = j["S"], j["E"], j["W"]
        f.add(E((s * 0.198, 0.006, 1.447), (0.052, 0.060, 0.082), rot_y(s * 0.22)), 0.05)                 # deltoid
        f.add(C(S, El, 0.050, 0.040), 0.03)                                                                  # upper arm
        ua = El - S
        f.add(E(lerp(S, El, 0.52) + (0, -0.012, 0), (0.036, 0.036, 0.085), axis_frame(ua)), 0.02)           # biceps
        f.add(E(lerp(S, El, 0.45) + (0, 0.022, 0), (0.035, 0.034, 0.092), axis_frame(ua)), 0.02)            # triceps
        fa = W - El
        f.add(C(El, W, 0.040, 0.027), 0.02)                                                                  # forearm
        f.add(E(lerp(El, W, 0.28) + (s * 0.004, 0.0, 0), (0.038, 0.034, 0.078), axis_frame(fa)), 0.025)    # forearm muscles
        hand(f, s, W, El)

    # --- legs ---------------------------------------------------------------------
    for s in (-1, 1):
        j = joints(s)
        H, K, A = j["H"], j["K"], j["A"]
        th = K - H
        f.add(C(H, K, 0.090, 0.056), 0.04)                                                                   # thigh
        f.add(E(lerp(H, K, 0.47) + (0, -0.030, 0), (0.066, 0.052, 0.175), axis_frame(th)), 0.03)            # quadriceps
        f.add(E(lerp(H, K, 0.75) + (-s * 0.024, -0.024, 0), (0.034, 0.034, 0.062), axis_frame(th)), 0.02)   # vastus medialis
        f.add(E(lerp(H, K, 0.45) + (0, 0.032, 0), (0.055, 0.045, 0.160), axis_frame(th)), 0.03)            # hamstrings
        f.add(E(lerp(H, K, 0.2) + (s * 0.040, 0.0, 0), (0.034, 0.050, 0.090), axis_frame(th)), 0.03)       # outer hip
        f.add(E(K + (0, -0.002, 0), (0.051, 0.049, 0.052)), 0.02)                                           # knee
        f.add(E(K + (0, -0.042, 0.002), (0.028, 0.018, 0.033)), 0.012)                                      # kneecap
        sh = A - K
        f.add(C(K, A, 0.048, 0.034), 0.02)                                                                   # shin
        f.add(E(lerp(K, A, 0.30) + (s * 0.002, 0.030, 0), (0.050, 0.045, 0.100), axis_frame(sh)), 0.03)    # calf
        f.add(E(A + (s * 0.020, 0.004, 0.002), (0.012, 0.014, 0.016)), 0.01)                                # ankle bones
        f.add(E(A + (-s * 0.018, 0.000, 0.008), (0.012, 0.014, 0.016)), 0.01)
        # bare stylised foot, turned out a little
        Rt = rot_z(s * 0.14)
        heel = A + np.array((0, 0.022, -0.045))
        f.add(E(heel, (0.030, 0.036, 0.040)), 0.02)
        ball = A + Rt @ np.array((0, -0.150, -0.063))
        f.add(E(lerp(A, ball, 0.38) + (0, 0, -0.008), (0.034, 0.060, 0.030), Rt @ rot_x(-0.45)), 0.03)     # instep
        f.add(sdf.rbox(lerp(heel, ball, 0.5) + (0, 0, 0.002), (0.038, 0.082, 0.026), 0.022, Rt), 0.03)
        f.add(E(ball + Rt @ np.array((-s * 0.004, -0.026, -0.004)), (0.043, 0.036, 0.020), Rt), 0.02)      # toes
    # flat soles: nothing below the floor
    f.F[:, :, 0] = np.maximum(f.F[:, :, 0], 0.001)
    return f


# ----------------------------------------------------------------------------- clothes
def _near_surface(fb, lo=-0.03, hi=0.05):
    """Grid indices close to the body surface, where clothing detail can matter."""
    return np.nonzero((fb.F > lo) & (fb.F < hi))


def _wrinkle_offset(fb, base, amp, scale, seed, extra=None):
    """Per-sample shell offset: base thickness plus anisotropic fabric folds."""
    import texbake as tb
    off = np.full(fb.F.shape, base, np.float32)
    idx = _near_surface(fb)
    P = fb.o + np.stack(idx, -1) * fb.h
    n = tb.fbm((P * np.asarray(scale, float)).astype(np.float32), 1.0, 3, seed=seed)
    w = amp * (n - 0.5) * 2.0
    if extra is not None:
        w = w + extra(P)
    off[idx] += w.astype(np.float32)
    return off


def _arm_region(s, pad):
    """Solid covering a whole arm from the upper arm to the fingertips."""
    j = joints(s)
    a = j["S"] + (j["E"] - j["S"]) * 0.15
    d = (j["W"] - j["E"]) / np.linalg.norm(j["W"] - j["E"])
    parts = [sdf.cone(a, j["E"], 0.06 + pad, 0.05 + pad), sdf.cone(j["E"], j["W"], 0.05 + pad, 0.04 + pad),
             sdf.ellipsoid(j["W"] + d * 0.1 + (-s * 0.01, 0, 0), (0.04 + pad, 0.06 + pad, 0.13 + pad), axis_frame(d))]
    return sdf.custom(lambda p: np.min([q.fn(p) for q in parts], axis=0),
                      np.min([q.lo for q in parts], axis=0), np.max([q.hi for q in parts], axis=0))


def shirt_field(fb):
    """Crew-neck T-shirt, worn untucked over the jeans: hangs off the chest, loose short sleeves."""
    def drape(P):
        # below the chest the jersey falls straight instead of following the waist
        return np.clip(1.30 - P[:, 2], 0, 0.4) * 0.05
    g = fb.like(fb.F - _wrinkle_offset(fb, 0.005, 0.002, (9, 9, 14), 5, drape))
    lo, hi = fb.o, fb.o + (fb.n - 1) * fb.h
    for side in (-1, 1):
        j = joints(side)
        ua = (j["E"] - j["S"]) / np.linalg.norm(j["E"] - j["S"])
        g.add(sdf.cone(j["S"] + ua * 0.01, j["S"] + ua * 0.16, 0.066, 0.060), 0.03)          # sleeve
    for side in (-1, 1):
        j = joints(side)
        ua = (j["E"] - j["S"]) / np.linalg.norm(j["E"] - j["S"])
        cut_at = j["S"] + ua * 0.15
        arm = sdf.cone(cut_at - ua * 0.002, j["W"] + ua * 0.3, 0.09, 0.09)
        g.cut(sdf.both(arm, sdf.halfspace(cut_at, -ua, arm.lo, arm.hi)))                   # sleeve hem
        g.cut(sdf.cone(cut_at - ua * 0.06, cut_at + ua * 0.02, 0.051, 0.057), 0.003)       # open sleeve
    g.cut(sdf.halfspace((0, 0, 0.945), (0, 0, 1), lo, (hi[0], hi[1], 0.95)))                 # hem
    g.cut(sdf.halfspace((0, 0, 1.598), (0, 0, -1), (lo[0], lo[1], 1.58), hi))                # head and jaw
    g.cut(sdf.ellipsoid((0, -0.012, 1.588), (0.074, 0.070, 0.064), rot_x(-0.35)), 0.004)    # neckline
    # nothing above the collar: a cone opening upwards crosses the steep neck-to-shoulder
    # fillet at an angle, where a vertical cylinder grazed it and left a ragged edge
    g.cut(sdf.cone((0, 0.004, 1.545), (0, 0.004, 1.70), 0.068, 0.150), 0.006)
    return g


def jeans_field(fb):
    """Straight-leg jeans: snug over the seat, loose below the knee, stacked at the hem."""
    def folds(P):
        # compression folds behind the knee and stacking above the hem
        z = P[:, 2]
        knee = np.exp(-((z - 0.52) / 0.05) ** 2) * 0.0025 * np.sin(z * 260)
        hem = np.exp(-((z - 0.12) / 0.06) ** 2) * 0.003 * np.sin(z * 200 + P[:, 0] * 40)
        return knee + hem
    g = fb.like(fb.F - _wrinkle_offset(fb, 0.0045, 0.0018, (10, 10, 22), 9, folds))
    lo, hi = fb.o, fb.o + (fb.n - 1) * fb.h
    for side in (-1, 1):
        j = joints(side)
        hem = np.array((j["A"][0], j["A"][1] - 0.004, 0.05))
        g.add(sdf.cone(j["K"] + (0, -0.002, 0.06), hem, 0.066, 0.063), 0.035)                # leg below the knee
        g.cut(_arm_region(side, 0.012))
    g.cut(sdf.halfspace((0, 0, 1.035), (0, 0, -1), (lo[0], lo[1], 1.02), hi))               # waistband
    g.cut(sdf.halfspace((0, 0, 0.058), (0, 0, 1), lo, (hi[0], hi[1], 0.07)))                # hem
    for side in (-1, 1):
        j = joints(side)
        c0 = np.array((j["A"][0], j["A"][1] - 0.004, 0.0))
        g.cut(sdf.cone(c0, c0 + (0, 0, 0.16), 0.058, 0.05), 0.004)                          # open leg bottoms
    return g


# ----------------------------------------------------------------------------- plate carrier
PLATE_R = 0.32      # plates curve around a vertical axis
PLATE_T = 0.026     # plate bag thickness
FRONT_IN = -0.125   # y of the front bag's inner face on the centre line
BACK_IN = 0.138


def _inset(poly, r):
    """Shrink a convex polygon by r (for rounded outline corners: sdf(inset) - r)."""
    P = np.asarray(poly, float)
    n = len(P)
    out = []
    for i in range(n):
        a, b, c_ = P[i - 1], P[i], P[(i + 1) % n]
        n1 = np.array((-(b - a)[1], (b - a)[0]))
        n2 = np.array((-(c_ - b)[1], (c_ - b)[0]))
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        # polygons are counter-clockwise, so the inward normals point left
        m = n1 + n2
        m /= np.linalg.norm(m)
        out.append(b + m * r / max(float(m @ n1), 1e-6))
    return out


class PlateFrame:
    """Cylindrical frame of a curved plate: s = arc length across, rho = distance from the axis."""

    def __init__(self, front):
        self.sgn = -1.0 if front else 1.0
        y_in = FRONT_IN if front else BACK_IN
        self.yc = y_in - self.sgn * PLATE_R

    def coords(self, p):
        u = self.sgn * (p[:, 1] - self.yc)
        rho = np.sqrt(p[:, 0] ** 2 + u * u)
        return PLATE_R * np.arctan2(p[:, 0], u), rho, p[:, 2]

    def point(self, s_, rho, z):
        a = s_ / PLATE_R
        return np.array((rho * math.sin(a), self.yc + self.sgn * rho * math.cos(a), z))

    def rotation(self, s_):
        """Local axes: x across the plate, y outwards, z up."""
        a = s_ / PLATE_R
        x = np.array((math.cos(a), -self.sgn * math.sin(a), 0.0))
        y = np.array((math.sin(a), self.sgn * math.cos(a), 0.0))
        return np.stack([x, y, (0, 0, 1)], 1)


def curved_panel(frame, outline, r0, r1, round_r=0.008, corner_r=0.015):
    """Solid between radii r0..r1 of a PlateFrame, clipped to an (s, z) outline."""
    poly = sdf.poly2d(_inset(outline, corner_r))
    rm, hw = (r0 + r1) / 2, (r1 - r0) / 2

    def fn(p):
        s_, rho, z = frame.coords(p)
        d_r = np.abs(rho - rm) - hw
        d_o = poly(np.stack([s_, z], -1)) - corner_r
        return sdf.round_inter(d_r, d_o, round_r)

    O = np.asarray(outline, float)
    lo = np.array((-(O[:, 0].max() + 0.05), -0.4, O[:, 1].min() - 0.02))
    hi = np.array((O[:, 0].max() + 0.05, 0.4, O[:, 1].max() + 0.02))
    return sdf.custom(fn, lo, hi)


FRONT_OUTLINE = [(-0.130, 1.140), (0.130, 1.140), (0.130, 1.370), (0.078, 1.468), (-0.078, 1.468), (-0.130, 1.370)]
BACK_OUTLINE = [(-0.135, 1.100), (0.135, 1.100), (0.135, 1.440), (0.110, 1.482), (-0.110, 1.482), (-0.135, 1.440)]
MOLLE_Z = (1.165, 1.215, 1.265, 1.315)
POUCH_S = (-0.082, 0.0, 0.082)


def carrier_fields(fb):
    """Plate bags + shoulder straps + cummerbund, and the pouches, as two fields. The soft
    parts are fitted to the torso and neck alone, so the arms hanging beside them can't
    carve them up."""
    fr, bk = PlateFrame(True), PlateFrame(False)
    bag = fb.like()
    bag.add(curved_panel(fr, FRONT_OUTLINE, PLATE_R, PLATE_R + PLATE_T))
    bag.add(curved_panel(bk, BACK_OUTLINE, PLATE_R, PLATE_R + PLATE_T))
    lo, hi = fb.o, fb.o + (fb.n - 1) * fb.h
    trunk = fb.like()
    torso(trunk)
    neck(trunk)
    P = trunk.points().reshape(-1, 3)
    # the T-shirt hangs off the chest (see shirt_field), so the cummerbund stands off with it
    drape = (np.clip(1.30 - P[:, 2], 0, 0.4) * 0.05).reshape(trunk.F.shape)
    inner = 0.009 + drape
    band = trunk.like(np.maximum(trunk.F - (inner + 0.012), inner - trunk.F))
    band.intersect(sdf.custom(lambda p: np.maximum(np.abs(p[:, 2] - 1.225) - 0.075, 0.105 - np.abs(p[:, 0])), lo, hi),
                   0.006, everywhere=True)
    # straps over the trapezius, kept clear of the neck
    straps = trunk.like(np.maximum(trunk.F - 0.021, 0.009 - trunk.F))

    def strap_zone(p):
        across = np.abs(np.abs(p[:, 0]) - 0.105) - 0.023
        neck_gap = 0.074 - np.hypot(p[:, 0], p[:, 1] - 0.017)
        return np.maximum(np.maximum(across, 1.43 - p[:, 2]), np.maximum(p[:, 2] - 1.60, neck_gap))

    straps.intersect(sdf.custom(strap_zone, lo, hi), 0.005, everywhere=True)
    bag.F = sdf.smin(sdf.smin(bag.F, band.F, 0.01), straps.F, 0.01).astype(np.float32)
    # drag handle on the back
    bag.add(sdf.rbox(bk.point(0.0, PLATE_R + PLATE_T * 0.5, 1.492), (0.036, 0.009, 0.018), 0.007, bk.rotation(0.0)), 0.004)

    pouch = fb.like()
    for s_ in POUCH_S:
        R = fr.rotation(s_)
        pouch.add(sdf.rbox(fr.point(s_, PLATE_R + PLATE_T + 0.019, 1.192), (0.036, 0.018, 0.068), 0.009, R), 0.003)
        pouch.add(sdf.rbox(fr.point(s_, PLATE_R + PLATE_T + 0.021, 1.250), (0.038, 0.020, 0.018), 0.006, R), 0.002)   # flap
    # hook-and-loop ID panel above the pouches
    pouch.add(curved_panel(fr, [(-0.075, 1.365), (0.075, 1.365), (0.075, 1.43), (-0.075, 1.43)],
                           PLATE_R + PLATE_T - 0.001, PLATE_R + PLATE_T + 0.003, 0.0015, 0.006))
    return bag, pouch


MOLLE_PITCH = 0.0381        # bar tacks every 1.5 inches
MOLLE_W = 0.025             # 1 inch webbing


def molle_rows():
    """(frame, row height, half length) of every MOLLE row, front and back."""
    fr, bk = PlateFrame(True), PlateFrame(False)
    return [(fr, z, 0.118) for z in MOLLE_Z] + [(bk, z + 0.02, 0.12) for z in MOLLE_Z]


def molle_loft(s_):
    """How far the webbing stands off the bag at arc position s: flat on the bar tacks,
    a couple of millimetres proud over the loops between them."""
    k = np.clip(np.round(s_ / MOLLE_PITCH), -3, 3)
    d = np.abs(s_ - k * MOLLE_PITCH)
    t = np.clip(d / 0.007, 0, 1)
    return 0.0022 * t * t * (3 - 2 * t) * (np.abs(s_) <= 3 * MOLLE_PITCH)


def _slab(bm, corners):
    """A closed box from its 8 corners (bottom ring then top ring, counter-clockwise)."""
    v = [bm.verts.new(c) for c in corners]
    for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        bm.faces.new([v[i] for i in f])


def molle_webbing(mat):
    """Webbing strips as real geometry (too thin to survive the voxel grid): each row is a
    closed band hugging the plate bag's curve, sampled densely only around the tacks.
    Also the pull tabs hanging from the pouch flaps."""
    base = PLATE_R + PLATE_T
    bm = bmesh.new()
    fr = PlateFrame(True)
    for s_ in POUCH_S:
        R = fr.rotation(s_)
        c0 = fr.point(s_, PLATE_R + PLATE_T + 0.021, 1.250)          # flap centre (see carrier_fields)
        pts = []
        for lz in (-0.0105, -0.031):                                  # sewn under the flap .. free end
            for lx in (-0.008, 0.008):
                for ly in (0.0, 0.0022):
                    out = 0.0195 + ly + (-0.0105 - lz) * 0.18         # the free end lifts off the pouch
                    pts.append((lx, out, lz))
        # order: bottom ring (free end) then top ring, each counter-clockwise seen from above
        ring = lambda lz_i: [pts[lz_i * 4 + k] for k in (0, 2, 3, 1)]
        corners = [c0 + R @ np.array(p_) for p_ in ring(1) + ring(0)]
        _slab(bm, corners)
    for frame, z, half in molle_rows():
        ss = {-half, half}
        for k in range(-3, 4):
            for o in (0.0, 0.0015, 0.0035, 0.0055, 0.0075, MOLLE_PITCH / 2):
                for sg in (-1, 1):
                    v = k * MOLLE_PITCH + sg * o
                    if abs(v) < half - 0.001:
                        ss.add(round(v, 6))
        ss = sorted(ss)
        w = MOLLE_W / 2
        ring = []
        for s_ in ss:
            out = base + 0.0011 + float(molle_loft(np.array([s_]))[0])
            # section: inner face tucked into the bag, outer face with softened edges
            sec = [(base - 0.001, -w), (out - 0.0005, -w), (out, -w + 0.0006),
                   (out, w - 0.0006), (out - 0.0005, w), (base - 0.001, w)]
            ring.append([bm.verts.new(frame.point(s_, r, z + dz)) for r, dz in sec])
        n = len(ring[0])
        for a, b in zip(ring[:-1], ring[1:]):
            for i in range(n):
                bm.faces.new((a[i], a[(i + 1) % n], b[(i + 1) % n], b[i]))
        bm.faces.new(ring[0])
        bm.faces.new(ring[-1][::-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("Webbing")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("Webbing", me)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def body_under_clothes(body, shells, margin=0.002):
    """Delete body faces hidden inside the clothes (saves triangles and texels)."""
    me = body.data
    V = np.array([v.co for v in me.vertices])
    hidden = np.zeros(len(V), bool)
    for g in shells:
        hidden |= g.sample(V) < -margin
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    kill = [f for f in bm.faces if all(hidden[v.index] for v in f.verts)]
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bm.to_mesh(me)
    bm.free()


def field_mesh(name, f, mat):
    t0 = time.time()
    verts, quads = f.mesh()
    print(f"  surface nets: {len(verts)} verts, {len(quads)} quads in {time.time() - t0:.1f}s", flush=True)
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], quads.tolist())
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    obj.data.materials.append(mat)
    return obj


def relax_and_reduce(obj, tris=26000, iterations=6):
    sm = obj.modifiers.new("relax", "CORRECTIVE_SMOOTH")
    sm.iterations = iterations
    sm.factor = 0.5
    sm.use_only_smooth = True
    sm.smooth_type = "SIMPLE"
    c_apply(obj)
    n = len(obj.data.polygons) * 2
    dec = obj.modifiers.new("reduce", "DECIMATE")
    dec.ratio = min(1.0, tris / max(n, 1))
    c_apply(obj)
    obj.data.shade_smooth()


def c_apply(obj):
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        for m in list(obj.modifiers):
            bpy.ops.object.modifier_apply(modifier=m.name)


def preview(out_dir, objs, names=("front", "side", "three_quarter"), samples=24):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = 720, 960
    sc.view_settings.view_transform = "AgX"
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.052, 0.055, 1)
    for i, (loc, e, sz) in enumerate((((1.6, -2.2, 2.6), 260.0, 1.5), ((-2.4, -1.2, 1.4), 60.0, 2.0), ((0.4, 2.5, 2.2), 180.0, 1.0))):
        L = bpy.data.objects.new(f"pl{i}", bpy.data.lights.new(f"pl{i}", "AREA"))
        L.data.energy, L.data.size = e, sz
        L.location = loc
        L.rotation_euler = (Vector((0, 0, 1.0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(L)
    floor = bpy.data.objects.new("floor", bpy.data.meshes.new("floor"))
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=4)
    bm.to_mesh(floor.data)
    bm.free()
    floor.data.materials.append(c.material("floorm", (0.18, 0.18, 0.18), roughness=0.8))
    sc.collection.objects.link(floor)
    cam = bpy.data.objects.new("pcam", bpy.data.cameras.new("pcam"))
    cam.data.lens = 50
    sc.collection.objects.link(cam)
    sc.camera = cam
    shots = {
        "front": ((0, -4.2, 1.0), (0, 0, 0.95)),
        "side": ((4.2, 0, 1.0), (0, 0, 0.95)),
        "three_quarter": ((2.3, -3.4, 1.35), (0, 0, 0.95)),
        "head": ((0.35, -0.9, 1.72), (0, 0, 1.70)),
        "back": ((-1.2, 3.8, 1.2), (0, 0, 0.95)),
        "wrist": ((-0.9, 0.9, 1.0), (-0.25, 0.0, 0.9)),
        "face": ((0.0, -0.75, 1.71), (0, 0, 1.71)),
        "profile": ((0.75, -0.05, 1.71), (0, -0.05, 1.71)),
        "mouth": ((0.06, -0.36, 1.69), (0, -0.08, 1.672)),
        "hand": ((0.75, -0.55, 0.85), (0.27, -0.02, 0.80)),
        "hand_in": ((-0.05, -0.55, 0.80), (0.27, -0.02, 0.80)),
        "chest": ((0.35, -1.0, 1.35), (0, -0.1, 1.28)),
        "backpack": ((-0.3, 1.0, 1.35), (0, 0.1, 1.28)),
    }
    os.makedirs(out_dir, exist_ok=True)
    for n in names:
        loc, tgt = shots[n]
        cam.location = loc
        cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(out_dir, f"{n}.png")
        bpy.ops.render.render(write_still=True)


# ----------------------------------------------------------------------------- stand
BASE_T = 0.012                  # steel base plate; the figure stands on it
ROD = (-0.106, 0.125, 0.33)     # calf rod into the right leg: x and y of the upright, height of the socket


def tube(name, pts, radius, mat, n=18):
    """A round bar swept along a polyline, its rings carried along by parallel transport."""
    P = [np.asarray(p, float) for p in pts]
    T = [unit(P[1] - P[0])] + [unit(P[i + 1] - P[i - 1]) for i in range(1, len(P) - 1)] + [unit(P[-1] - P[-2])]
    ref = np.array((1.0, 0, 0)) if abs(T[0][0]) < 0.9 else np.array((0, 1.0, 0))
    nv = unit(np.cross(T[0], ref))
    bm = bmesh.new()
    rings = []
    for p, t in zip(P, T):
        nv = unit(nv - t * (nv @ t))
        bv = np.cross(t, nv)
        rings.append([bm.verts.new(tuple(p + radius * (math.cos(a) * nv + math.sin(a) * bv)))
                      for a in np.linspace(0, 2 * math.pi, n, endpoint=False)])
    for r0, r1 in zip(rings[:-1], rings[1:]):
        for k in range(n):
            bm.faces.new((r0[k], r0[(k + 1) % n], r1[(k + 1) % n], r1[k]))
    bm.faces.new(rings[0][::-1])
    bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    c.smooth(obj, 50)
    return obj


def stand(mats, jn):
    """Shop-mannequin stand: a steel base plate and an L-shaped rod into the back of the right calf."""
    steel, zinc = mats["Steel"], mats["Zinc"]
    x, y, zr = ROD
    # where the rod meets the trouser leg, found in the jeans field (figure space)
    ys = np.linspace(0.2, -0.05, 1001)
    vals = jn.sample(np.stack([np.full_like(ys, x), ys, np.full_like(ys, zr)], 1))
    y_s = float(ys[np.argmax(vals < 0)])
    z = zr + BASE_T
    bend = 0.035
    path = [(x, y, BASE_T + 0.004), (x, y, z - bend)]
    for k in range(1, 8):
        a = k / 8 * math.pi / 2
        path.append((x, y - bend * (1 - math.cos(a)), z - bend + bend * math.sin(a)))
    path += [(x, y - bend, z), (x, y_s - 0.03, z)]
    parts = [c.box("Base", (0.48, 0.42, BASE_T), (0, -0.045, BASE_T / 2), steel, bevel=0.0025, segments=2),
             c.cylinder("Flange", 0.030, 0.006, (x, y, BASE_T + 0.003), mat=steel, verts=32, bevel=0.001),
             tube("Rod", path, 0.008, steel),
             tube("Socket", [(x, y_s + 0.008, z), (x, y_s - 0.004, z)], 0.0125, zinc, n=24)]
    for k in range(3):
        a = math.radians(90 + 120 * k)
        parts.append(c.cylinder(f"Bolt{k}", 0.0052, 0.004, (x + 0.021 * math.cos(a), y + 0.021 * math.sin(a), BASE_T + 0.008),
                                mat=zinc, verts=6, bevel=0.0006))
    return c.join("Stand", parts)


# ----------------------------------------------------------------------------- range history
class Hit:
    """A bullet hole: where a shot struck (figure space), the surface normal there and the material hit."""

    def __init__(self, p, n, mat, seed):
        self.p, self.n, self.mat, self.seed = p, n, mat, seed
        t = np.cross(n, (0, 0, 1.0)) if abs(n[2]) < 0.9 else np.cross(n, (1.0, 0, 0))
        self.decal = decals.Decal(p, n, t, reach=0.03, depth=0.02, facing=0.2)


class Mark:
    """A projected decal (tape or marker) with the function that draws it in decal coordinates."""

    def __init__(self, c_, n, t, reach, draw, seed, mat="Paint", ink=None):
        self.decal = decals.Decal(c_, n, t, reach=reach, depth=0.04, facing=0.05)
        self.draw, self.seed, self.mat = draw, seed, mat
        self.ink = INK if ink is None else np.asarray(ink, float)


def _scene_bvh(objs):
    from mathutils.bvhtree import BVHTree
    verts, polys, owner = [], [], []
    for o in objs:
        me = o.data
        base = len(verts)
        verts += [tuple(o.matrix_world @ v.co) for v in me.vertices]
        for p in me.polygons:
            polys.append([base + i for i in p.vertices])
            owner.append(o.material_slots[p.material_index].material.name)
    return BVHTree.FromPolygons(verts, polys), owner


def shoot(objs, seed=7):
    """Fire the range's history at the figure: rays from shooting positions to aim points.
    The first surface each ray meets gets a hole. Returns hits in figure space."""
    tree, owner = _scene_bvh(objs)
    rng = np.random.default_rng(seed)
    groups = (  # count, aim point (figure space), spread, from behind
        (5, (0, 0, 1.73), (0.035, 0.0, 0.05), False),
        (8, (0, 0, 1.25), (0.11, 0.0, 0.14), False),
        (3, (0, 0, 1.02), (0.09, 0.0, 0.04), False),
        (4, (0.25, 0, 1.12), (0.03, 0.0, 0.16), False),
        (3, (-0.25, 0, 1.12), (0.03, 0.0, 0.16), False),
        (4, (0.10, 0, 0.70), (0.04, 0.0, 0.20), False),
        (3, (-0.10, 0, 0.62), (0.04, 0.0, 0.20), False),
        (3, (0, 0, 1.30), (0.10, 0.0, 0.15), True),
    )
    hits = []
    for count, aim, spread, behind in groups:
        for _ in range(count):
            tgt = np.array(aim) + rng.uniform(-1, 1, 3) * spread + (0, 0, BASE_T)
            org = np.array((rng.uniform(-2.0, 2.0), 7.0 if behind else -7.0, rng.uniform(1.2, 1.7)))
            loc, nor, idx, _d = tree.ray_cast(Vector(org), Vector(unit(tgt - org)), 20.0)
            if loc is not None:
                hits.append(Hit(np.array(loc) - (0, 0, BASE_T), np.array(nor), owner[idx], 11 + 7 * len(hits)))
    return hits


def surface_point(tree, frm, towards):
    """Where a ray from frm towards a point meets the figure (figure space), with its normal."""
    frm = np.asarray(frm, float) + (0, 0, BASE_T)
    to = np.asarray(towards, float) + (0, 0, BASE_T)
    loc, nor, _i, _d = tree.ray_cast(Vector(frm), Vector(unit(to - frm)), 5.0)
    return np.array(loc) - (0, 0, BASE_T), np.array(nor)


def paint_marks(objs, hits, seed=5):
    """Duct tape over old holes, marker rings round a couple of fresh ones, and a tally on the forearm."""
    rng = np.random.default_rng(seed)
    tree, _ = _scene_bvh(objs)
    marks = []
    skin = [h for h in hits if h.mat == "Paint"]
    face = [h for h in skin if h.p[2] > 1.6 and h.n[1] < -0.4 and abs(h.p[0]) < 0.07]
    # tape goes over old holes on the arms, neck and scalp; the face keeps its fresh hits
    taped = [h for h in skin if not any(h is f_ for f_ in face)][:3]
    for k, h in enumerate(taped):
        ang = rng.uniform(0, math.pi)
        t = np.cross(h.n, (0, 0, 1.0)) if abs(h.n[2]) < 0.9 else np.array((1.0, 0, 0))
        t = unit(t) * math.cos(ang) + np.cross(h.n, unit(t)) * math.sin(ang)
        strips = [(t, 0.075 + 0.02 * rng.random())]
        if k == 0:                                           # an X of two strips
            strips.append((np.cross(h.n, t), 0.07))
        for j, (tt, length) in enumerate(strips):
            sd = seed * 100 + k * 10 + j
            marks.append(Mark(h.p + h.n * 0.001, h.n, tt, 0.06,
                              (lambda L, s_: lambda u, v: decals.tape_strip(u, v, L, 0.048, s_))(length, sd), sd))
    for k, h in enumerate(face[:2]):
        sd = seed * 100 + 50 + k
        marks.append(Mark(h.p, h.n, (1.0, 0, 0), 0.04,
                          (lambda s_: lambda u, v: decals.marker_circle(u, v, 0.017, seed=s_))(sd), sd,
                          ink=(0.30, 0.018, 0.02) if k == 0 else None))
    # tally of hits on the figure's left forearm
    j = joints(1)
    p, n = surface_point(tree, (0.75, -0.35, 1.02), lerp(j["E"], j["W"], 0.42))
    axis = unit(j["W"] - j["E"])
    strokes = [((k * 0.0065 - 0.0095, -0.012), (k * 0.0065 - 0.0085, 0.012)) for k in range(4)]
    strokes.append(((-0.0145, -0.009), (0.0135, 0.010)))
    marks.append(Mark(p, n, axis, 0.03, lambda u, v: decals.marker_strokes(u, v, strokes, seed=3), 99))
    return marks


# ----------------------------------------------------------------------------- textures
SURF = {  # material: (id, base colour (linear), roughness)
    "Paint": (1, (0.49, 0.475, 0.445), 0.40),
    "Shirt": (2, (0.021, 0.022, 0.025), 0.92),
    "Denim": (3, (0.034, 0.050, 0.105), 0.88),
    "Cordura": (4, (0.205, 0.140, 0.078), 0.86),
    "Webbing": (5, (0.180, 0.122, 0.068), 0.72),
    "Pouch": (6, (0.205, 0.140, 0.078), 0.86),
    "Steel": (7, (0.020, 0.020, 0.022), 0.50),
    "Zinc": (8, (0.55, 0.555, 0.56), 0.35),
}
# Micro relief added in the normal bake: noise scale, bump strength, distortion.
MICRO = {"Paint": (2200.0, 0.03, 0.0), "Shirt": (1500.0, 0.10, 0.0), "Denim": (900.0, 0.14, 1.0),
         "Cordura": (1400.0, 0.12, 0.0), "Webbing": (1000.0, 0.10, 0.0), "Pouch": (1400.0, 0.12, 0.0),
         "Steel": (1600.0, 0.05, 0.0)}
GRIME = np.array((0.16, 0.14, 0.11))
FILLER = np.array((0.30, 0.295, 0.28))
GLASS = np.array((0.40, 0.36, 0.25))
TAPE = np.array((0.27, 0.28, 0.29))
INK = np.array((0.022, 0.022, 0.028))
GOLD_THREAD = np.array((0.34, 0.19, 0.05))
FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")


def mix(a, b, t):
    t = np.asarray(t, float)
    if np.ndim(a) == 2 or np.ndim(b) == 2:
        t = t[:, None]
    return a + (np.asarray(b, float) - a) * t


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def laplacian(fn, P, eps):
    """Laplacian of a distance field: > 0 on convex forms, < 0 in hollows (2 x mean curvature)."""
    lap = -6.0 * fn(P)
    for a in range(3):
        e = np.zeros(3)
        e[a] = eps
        lap += fn(P + e) + fn(P - e)
    return lap / (eps * eps)


class Texels:
    """The texels of one material: figure-space positions and normals, baked AO/edge/cavity,
    and the holes and marks that land on it."""

    def __init__(self, name, P, N, AO, fields, hits, marks):
        self.name, self.P, self.N = name, P, N
        self.ao, self.edge, self.cav = AO[:, 0], AO[:, 1], AO[:, 2]
        self.fields = fields
        self.hits = [h for h in hits if h.mat == name]
        self.marks = [m for m in marks if m.mat == name]

    def noise(self, freq, octaves=3, seed=0, scale=(1.0, 1.0, 1.0), P=None):
        Q = self.P if P is None else P
        return tb.fbm((Q * np.asarray(scale)).astype(np.float32), freq, octaves, seed=seed).astype(np.float64)

    def holes(self, P, brittle):
        M = len(P)
        core, ring, cracks, wipe, h = (np.zeros(M) for _ in range(5))
        for hit in self.hits:
            u, v, ok = hit.decal.local(P, self.N)
            if not ok.any():
                continue
            c_, r_, k_, w_, h_ = decals.bullet_hole(u[ok], v[ok], hit.seed, brittle=brittle)
            core[ok] = np.maximum(core[ok], c_)
            ring[ok] = np.maximum(ring[ok], r_)
            cracks[ok] = np.maximum(cracks[ok], k_)
            wipe[ok] = np.maximum(wipe[ok], w_)
            h[ok] += h_
        return core, ring, cracks, wipe, h

    def apply_holes(self, base, rough, brittle, inner):
        core, ring, cracks, wipe, _ = self.holes(self.P, brittle)
        base = mix(base, base * 0.35 + 0.02, wipe * 0.8)
        if brittle:
            base = mix(base, np.array((0.66, 0.65, 0.62)), ring)
            rough = mix(rough, 0.85, ring)
            base = mix(base, GRIME * 0.5, cracks * 0.8)
        else:
            base = mix(base, base * 1.5 + 0.02, ring * 0.6)
        base = mix(base, inner, core)
        rough = mix(rough, 0.95, core)
        return base, rough


def stitch_rows(d, along, rows, pitch=0.0036, width=0.0007):
    """Parallel stitch rows at offsets `rows` from a seam line: (thread cover, height)."""
    cover = np.zeros_like(d)
    h = np.zeros_like(d)
    for r in rows:
        t, hh = decals.stitch(d - r, along, pitch, width)
        cover = np.maximum(cover, t)
        h += hh
    return cover, h


# --- mannequin paint ------------------------------------------------------------------------
def layer_paint(t):
    f = t.fields["body"]
    n0 = f.exact_normal(t.P)
    # Where the mesh strays from the exact surface (a detail cut behind it, decimation) the
    # gradient at the texel can point anywhere, even into the head: a cut's field grows
    # towards the middle of the cut. Take the normal where a cage ray meets the surface.
    off = (np.abs(f.exact(t.P)) > 0.0003) | (np.sum(n0 * t.N, 1) < 0.6)
    if off.any():
        Ps, hit = f.project(t.P[off], t.N[off])
        n0[off] = np.where(hit[:, None], f.exact_normal(Ps), t.N[off])
    curv = laplacian(f.exact, t.P, 0.004)
    col = np.array(SURF["Paint"][1])
    n_big, n_mid, n_fine = t.noise(3.0, 3, 1), t.noise(24.0, 3, 2), t.noise(150.0, 3, 3)
    base = col * (0.94 + 0.09 * n_big)[:, None] * (1 + (n_mid - 0.5)[:, None] * np.array((0.04, 0.025, -0.02)))
    rough = 0.34 + 0.10 * n_mid + 0.06 * n_fine
    convex = decals.smoothstep(80.0, 260.0, curv)
    hollow = decals.smoothstep(-15.0, -110.0, curv)
    # dust and grime settle in the hollows; hands and forearms are grubby from being carried about
    grab = np.zeros(len(t.P))
    for side in (-1, 1):
        j = joints(side)
        grab += np.exp(-(np.linalg.norm(t.P - (j["W"] + unit(j["W"] - j["E"]) * 0.07), axis=1) / 0.11) ** 2)
        grab += 0.5 * np.exp(-(np.linalg.norm(t.P - lerp(j["E"], j["W"], 0.4), axis=1) / 0.08) ** 2)
    crevice = np.maximum(decals.smoothstep(0.93, 0.6, t.cav), hollow * 0.5)
    # grime is soft-edged (no fine noise: it stippled the lips and lids)
    grime = np.clip(crevice * decals.smoothstep(0.2, 0.7, n_mid) * 0.8
                    + grab * decals.smoothstep(0.3, 0.8, n_mid * 0.7 + n_fine * 0.3) * 0.7, 0, 1)
    base = mix(base, GRIME, grime * 0.5)
    rough = rough + grime * 0.15
    # a grey film of dust on the surfaces facing straight up (shoulders, crown), not on every bump
    dust = decals.smoothstep(0.75, 0.97, n0[:, 2]) * decals.smoothstep(0.3, 0.8, n_big * 0.6 + n_mid * 0.4)
    base = mix(base, np.array((0.40, 0.39, 0.37)), dust * 0.25)
    rough = rough + dust * 0.15

    def chips(P):
        n = t.noise(55.0, 4, 4, P=P)
        chip = decals.smoothstep(0.665, 0.70, convex * 0.14 + n * 0.75 + grab * 0.04)
        deep = chip * decals.smoothstep(0.80, 0.84, n + convex * 0.05)
        return chip, deep

    chip, deep = chips(t.P)
    base = mix(base, FILLER, chip)
    base = mix(base, GLASS, deep)
    rough = mix(rough, 0.72, chip)

    def decal_layers(P):
        tape_m, tape_e, tape_w, ink = (np.zeros(len(P)) for _ in range(4))
        ink_col = np.tile(INK, (len(P), 1))
        h = np.zeros(len(P))
        for m in t.marks:
            u, v, ok = m.decal.local(P, t.N)
            if not ok.any():
                continue
            out = m.draw(u[ok], v[ok])
            if isinstance(out, tuple):                       # tape: (mask, edge, weave)
                mk, ed, wv = out
                cr = np.zeros(ok.sum())
                for k in range(3):                           # creases pressed into the tape
                    a = decals.hash01(m.seed, k) * math.pi
                    off = (decals.hash01(m.seed, k, 1) - 0.5) * 0.05
                    cr = np.maximum(cr, decals.ridge(u[ok] * math.cos(a) + v[ok] * math.sin(a) - off, 0.0012))
                tape_m[ok] = np.maximum(tape_m[ok], mk)
                tape_e[ok] = np.maximum(tape_e[ok], ed * mk)
                tape_w[ok] = np.maximum(tape_w[ok], wv * mk)
                h[ok] += mk * (0.00028 + 0.00003 * wv) + cr * mk * 0.00012
            else:
                ink[ok] = np.maximum(ink[ok], out)
                ink_col[np.nonzero(ok)[0][out > 0.3]] = m.ink
        return tape_m, tape_e, tape_w, ink, ink_col, h

    tape_m, tape_e, tape_w, ink, ink_col, _ = decal_layers(t.P)
    tape_col = TAPE * (0.95 + 0.1 * tape_w)[:, None] * (0.94 + 0.12 * n_fine)[:, None]
    tape_col = mix(tape_col, GRIME * 0.8, tape_e * 0.6)
    base = mix(base, tape_col, tape_m)
    rough = mix(rough, 0.36 - 0.05 * tape_w + tape_e * 0.25, tape_m)
    metal = tape_m * (0.3 - 0.2 * tape_e)
    base, rough = t.apply_holes(base, rough, True, np.array((0.012, 0.011, 0.01)))
    base = mix(base, ink_col, ink * (0.85 + 0.15 * n_fine))
    rough = mix(rough, 0.24, ink)
    metal = metal * (1 - ink)

    def height(P):
        c_, d_ = chips(P)
        _, _, _, _, hh = t.holes(P, True)
        *_, hd = decal_layers(P)
        return -0.00012 * c_ - 0.0002 * d_ + hh + hd

    return base, rough, metal, height, n0


# --- T-shirt ------------------------------------------------------------------------------------
NECKLINE = sdf.ellipsoid((0, -0.012, 1.588), (0.074, 0.070, 0.064), rot_x(-0.35))


def shirt_lines(P):
    """Seams of the T-shirt: list of (distance to the line, along-coordinate, stitch rows, seam?)."""
    x, y, z = P.T
    s = np.where(x >= 0, 1.0, -1.0)
    S = np.where(s[:, None] > 0, joints(1)["S"], joints(-1)["S"])
    Eb = np.where(s[:, None] > 0, joints(1)["E"], joints(-1)["E"])
    ua = (Eb - S) / np.linalg.norm(Eb - S, axis=1, keepdims=True)
    a = np.sum((P - S) * ua, 1)                         # along the upper arm from the shoulder joint
    radial = np.linalg.norm((P - S) - ua * a[:, None], axis=1)
    ref = np.cross(ua, (0, 1.0, 0))
    ang = np.arctan2(np.sum((P - S) * np.cross(ua, ref), 1), np.sum((P - S) * ref, 1))
    sleeve = (a > 0) & (radial < 0.1)
    far = np.full(len(P), 1.0)
    lines = []
    body = ~sleeve & (np.abs(x) > 0.07) & (z < 1.62)
    lines.append((np.where(body, y - 0.012, far), z + np.abs(x), (), True))                # side and shoulder seams
    lines.append((np.where(radial < 0.1, a, far), ang * 0.06, (0.004,), True))             # armholes
    lines.append((np.where(sleeve, a - 0.15, far), ang * 0.06, (-0.013, -0.019), False))  # sleeve hems
    lines.append((z - 0.945, np.arctan2(x, y) * 0.16, (0.013, 0.019), False))              # bottom hem
    e = NECKLINE.fn(P)
    around = np.arctan2(x, -(y + 0.012))
    lines.append((np.where(z > 1.5, e, far), around * 0.075, (0.017, 0.023), True))       # collar band seam
    return lines, e, around


def layer_shirt(t):
    P, N = t.P, t.N
    curv = laplacian(t.fields["shirt"].sample, P, VOXEL)
    col = np.array(SURF["Shirt"][1])
    heather = t.noise(650.0, 2, 21)
    n_mid = t.noise(18.0, 3, 22)
    # washed-out black: sun-faded across the shoulders, worn paler along fold ridges
    fade = decals.smoothstep(0.2, 0.9, N[:, 2]) * 0.7 + decals.smoothstep(30.0, 160.0, curv) * 0.5 + n_mid * 0.25
    base = col * (0.8 + 0.4 * heather)[:, None] + np.array((0.012, 0.011, 0.010)) * fade[:, None]
    rough = 0.88 + 0.08 * heather
    lines, e, around = shirt_lines(P)
    rib_band = (e < 0.017) & (P[:, 2] > 1.5)
    thread = np.zeros(len(P))
    for d, along, rows, _is_seam in lines:
        cv, _ = stitch_rows(d, along, rows)
        thread = np.maximum(thread, cv)
    base = mix(base, base * 1.35 + 0.004, thread * 0.8)
    base = mix(base, base * 0.85, rib_band * (0.5 + 0.5 * np.sin(around * 214)))
    base, rough = t.apply_holes(base, rough, False, np.array((0.004, 0.004, 0.005)))
    dust = decals.smoothstep(0.5, 0.95, N[:, 2]) * decals.smoothstep(0.4, 0.8, n_mid)
    base = mix(base, np.array((0.14, 0.13, 0.115)), dust * 0.25)

    def height(Q):
        ls, e_, ar = shirt_lines(Q)
        h = np.zeros(len(Q))
        for d, along, rows, is_seam in ls:
            if is_seam:
                h += decals.seam(d)
            h += stitch_rows(d, along, rows)[1]
        rib = (e_ < 0.017) & (Q[:, 2] > 1.5)
        h += rib * 0.00022 * (0.5 + 0.5 * np.sin(ar * 214))
        h += t.holes(Q, False)[4]
        return h

    return base, rough, np.zeros(len(P)), height, N


# --- jeans ----------------------------------------------------------------------------------------
BACK_POCKET = [(-0.066, 0.080), (-0.060, -0.050), (0.0, -0.080), (0.060, -0.050), (0.066, 0.080)]


def leg_frame(P):
    """Side of each point (+1 = the figure's left leg), radius and angle around the leg's axis
    (0 at the front, +pi/2 on the outside, -pi/2 on the inside)."""
    x, y, z = P.T
    s = np.where(x >= 0, 1.0, -1.0)
    cen = np.zeros_like(P)
    for side in (-1, 1):
        j = joints(side)
        top, knee = j["H"], j["K"] + (0, -0.002, 0.06)
        hem = np.array((j["A"][0], j["A"][1] - 0.004, 0.05))
        m = s == side
        zz = z[m]
        t1 = np.clip((top[2] - zz) / (top[2] - knee[2]), 0, 1)[:, None]
        t2 = np.clip((knee[2] - zz) / (knee[2] - hem[2]), 0, 1)[:, None]
        cen[m] = np.where((zz > knee[2])[:, None], top + (knee - top) * t1, knee + (hem - knee) * t2)
    v = P - cen
    return s, np.hypot(v[:, 0], v[:, 1]), np.arctan2(s * v[:, 0], -v[:, 1])


_pocket = sdf.poly2d(BACK_POCKET)


def jeans_lines(P):
    x, y, z = P.T
    s, r, phi = leg_frame(P)
    far = np.full(len(P), 1.0)
    lines = []
    lines.append((np.where(z < 1.0, r * wrap(phi - math.pi / 2), far), z, (0.004,), True))                 # outseam
    lines.append((np.where(z < 0.845, r * wrap(phi + math.pi / 2), far), z, (-0.0035, 0.0035), True))     # inseam
    lines.append((np.where((z > 0.80) & (y > 0), x, far), z, (-0.0035, 0.0035), True))                    # seat seam
    j_curve = np.where(z > 0.90, x - 0.035, np.hypot(x, z - 0.90) - 0.035)                                 # fly J-stitch
    lines.append((np.where((y < 0) & (z > 0.86) & (x > -0.01), j_curve, far), z + x, (0.0, -0.006), False))
    lines.append((np.where((z > 0.84) & (z < 0.90) & (y < 0), x, far), z, (), True))                      # front rise
    lines.append((z - 0.058, np.arctan2(x - s * 0.1, y) * 0.07, (0.013,), False))                          # hem
    q = np.stack([s * x - 0.076, z - 0.915], -1)
    dp = np.where((y > 0.02) & (z > 0.8), _pocket(q), far)
    lines.append((dp, z + x * s, (-0.003, -0.0085), False))                                                 # back pockets
    return lines, s, r, phi, dp


def layer_denim(t):
    P, N = t.P, t.N
    x, y, z = P.T
    curv = laplacian(t.fields["jeans"].sample, P, VOXEL)
    indigo = np.array(SURF["Denim"][1])
    faded = np.array((0.075, 0.105, 0.185))
    warp = t.noise(1.0, 3, 31, scale=(95, 95, 8))          # slubby warp yarn: streaks along the leg
    fleck = t.noise(420.0, 2, 32)
    n_mid = t.noise(14.0, 3, 33)
    lines, s, r, phi, dp = jeans_lines(P)
    front = decals.smoothstep(0.9, 0.2, np.abs(phi))
    thigh = front * decals.smoothstep(0.52, 0.62, z) * decals.smoothstep(0.92, 0.80, z)
    knee = front * np.exp(-((z - 0.50) / 0.05) ** 2)
    seat = (y > 0) * decals.smoothstep(0.82, 0.88, z) * decals.smoothstep(0.98, 0.92, z) * decals.smoothstep(0.02, 0.1, s * x)
    # whiskers fanning out from the crotch across the front of the hips
    wk = np.sin(((z - 0.87) - 0.30 * (s * x - 0.02)) * 2 * math.pi / 0.014) + 0.6 * np.sin(((z - 0.87) - 0.5 * (s * x)) * 2 * math.pi / 0.021)
    whisk = decals.smoothstep(0.4, 1.4, wk) * front * decals.smoothstep(0.80, 0.85, z) * decals.smoothstep(0.93, 0.88, z) \
        * decals.smoothstep(0.02, 0.05, s * x) * decals.smoothstep(0.15, 0.1, s * x)
    ridges = decals.smoothstep(40.0, 180.0, curv)
    wear = np.clip(0.35 * thigh + 0.45 * knee + 0.35 * seat + 0.35 * whisk + 0.4 * ridges + 0.2 * (n_mid - 0.5), 0, 1)
    wear = wear * (0.7 + 0.6 * warp)
    base = mix(np.broadcast_to(indigo, P.shape) * (0.8 + 0.45 * warp)[:, None], faded, wear)
    base = base * (0.9 + 0.2 * fleck)[:, None]
    rough = 0.86 + 0.06 * fleck
    thread = np.zeros(len(P))
    seam_wear = np.zeros(len(P))
    for d, along, rows, is_seam in lines:
        cv, _ = stitch_rows(d, along, rows, pitch=0.0034, width=0.0008)
        thread = np.maximum(thread, cv)
        if is_seam:
            seam_wear = np.maximum(seam_wear, decals.ridge(np.abs(d) - 0.004, 0.003))
    pocket = decals.smoothstep(0.001, -0.001, dp)
    seam_wear = np.maximum(seam_wear, decals.ridge(dp, 0.0025) * pocket)
    base = mix(base, faded * 1.2, seam_wear * 0.6)
    base = mix(base, GOLD_THREAD, thread * 0.9)
    base, rough = t.apply_holes(base, rough, False, np.array((0.004, 0.004, 0.006)))
    grime = decals.smoothstep(0.35, 0.1, z) * decals.smoothstep(0.3, 0.7, n_mid)
    base = mix(base, GRIME * 0.6, grime * 0.35)

    def height(Q):
        ls, s_, r_, ph, dp_ = jeans_lines(Q)
        h = np.zeros(len(Q))
        for d, along, rows, is_seam in ls:
            if is_seam:
                h += decals.seam(d, 0.003, 0.0006)
            h += stitch_rows(d, along, rows, pitch=0.0034, width=0.0008)[1]
        h += 0.0005 * decals.smoothstep(0.0012, -0.0012, dp_)                         # pocket patch thickness
        twill = np.sin((Q[:, 2] + r_ * ph) * 2 * math.pi / 0.0024)                   # 45 degree twill
        h += 0.00004 * twill
        h += t.holes(Q, False)[4]
        return h

    return base, rough, np.zeros(len(P)), height, N


# --- plate carrier ----------------------------------------------------------------------------
_front_poly = sdf.poly2d(_inset(FRONT_OUTLINE, 0.015))
_back_poly = sdf.poly2d(_inset(BACK_OUTLINE, 0.015))


def bag_lines(P):
    fr, bk = PlateFrame(True), PlateFrame(False)
    sf, rf, zf = fr.coords(P)
    sb, rb, zb = bk.coords(P)
    df = _front_poly(np.stack([sf, zf], -1)) - 0.015
    db = _back_poly(np.stack([sb, zb], -1)) - 0.015
    on_f = (rf > PLATE_R - 0.005) & (rf < PLATE_R + PLATE_T + 0.005) & (df < 0.005) & (P[:, 1] < 0)
    on_b = (rb > PLATE_R - 0.005) & (rb < PLATE_R + PLATE_T + 0.005) & (db < 0.005) & (P[:, 1] > 0)
    d_o = np.where(on_f, df, np.where(on_b, db, 1.0))
    panel = on_f | on_b
    along = np.where(on_f, np.arctan2(zf - 1.3, sf), np.arctan2(zb - 1.3, sb)) * 0.15
    x, y, z = P.T
    strap = ~panel & (z > 1.40)
    band = ~panel & ~strap & (np.abs(x) > 0.09)
    e_strap = np.where(strap, 0.026 - np.abs(np.abs(x) - 0.098), 1.0)
    e_band = np.where(band, 0.075 - np.abs(z - 1.225), 1.0)
    handle = (np.abs(x) < 0.05) & (z > 1.465) & (z < 1.52) & (y > 0.15)
    return panel, d_o, along, strap, e_strap, band, e_band, handle


def layer_bag(t):
    P, N = t.P, t.N
    x, y, z = P.T
    col = np.array(SURF["Cordura"][1])
    fleck = t.noise(500.0, 2, 41)
    n_mid = t.noise(16.0, 3, 42)
    curv = laplacian(t.fields["bag"].sample, P, VOXEL)
    panel, d_o, along, strap, e_strap, band, e_band, handle = bag_lines(P)
    binding = panel & (d_o > -0.011)
    base = col * (0.88 + 0.24 * fleck)[:, None] * (0.94 + 0.12 * n_mid)[:, None]
    base = mix(base, col * 0.78, binding)
    base = mix(base, np.array(SURF["Webbing"][1]), handle)
    rough = 0.84 + 0.08 * fleck - binding * 0.1
    thread = np.maximum.reduce([stitch_rows(d_o, along, (-0.0085,))[0] * panel,
                                stitch_rows(e_strap, z + y, (0.004,))[0] * strap,
                                stitch_rows(e_band, y + x, (0.004,))[0] * band])
    base = mix(base, col * 0.6, thread * 0.8)
    # sun-bleached on top, dirt low down and in the folds, grubby straps from being handled
    sun = decals.smoothstep(0.1, 0.8, N[:, 2]) + decals.smoothstep(60.0, 250.0, curv) * 0.5
    base = mix(base, np.array((0.30, 0.235, 0.155)), np.clip(sun, 0, 1) * 0.35)
    dirt = np.clip(decals.smoothstep(1.2, 1.1, z) * 0.5 + decals.smoothstep(0.9, 0.6, t.cav) + strap * 0.4, 0, 1) \
        * decals.smoothstep(0.3, 0.75, n_mid)
    base = mix(base, GRIME * 0.7, dirt * 0.45)
    base, rough = t.apply_holes(base, rough, False, np.array((0.01, 0.008, 0.006)))

    def height(Q):
        pn, do, al, st, es, bd, eb, _hd = bag_lines(Q)
        h = pn * (0.0004 * decals.smoothstep(-0.013, -0.009, do))                   # binding tape over the edge
        h += stitch_rows(do, al, (-0.0085,))[1] * pn
        h += stitch_rows(es, Q[:, 2] + Q[:, 1], (0.004,))[1] * st
        h += stitch_rows(eb, Q[:, 1] + Q[:, 0], (0.004,))[1] * bd
        h += t.holes(Q, False)[4]
        return h

    return base, rough, np.zeros(len(P)), height, N


def webbing_coords(P):
    """Arc position along the row and height across the webbing (0 on the row's centre line)."""
    front = P[:, 1] < 0
    fr, bk = PlateFrame(True), PlateFrame(False)
    sf, _, _ = fr.coords(P)
    sb, _, _ = bk.coords(P)
    rows_f = np.array(MOLLE_Z)
    rows_b = rows_f + 0.02
    z = P[:, 2]
    rows = np.where(front[:, None], rows_f[None, :], rows_b[None, :])
    dz = z[:, None] - rows
    across = dz[np.arange(len(P)), np.argmin(np.abs(dz), 1)]
    return np.where(front, sf, sb), across


def _on_plate(P):
    """Texels of the webbing lying on a plate bag (as opposed to the pouch pull tabs)."""
    front = P[:, 1] < 0
    _, rf, _ = PlateFrame(True).coords(P)
    _, rb, _ = PlateFrame(False).coords(P)
    return np.where(front, rf, rb) < PLATE_R + PLATE_T + 0.008


def layer_webbing(t):
    P, N = t.P, t.N
    s_, across = webbing_coords(P)
    col = np.array(SURF["Webbing"][1])
    rib = t.noise(1.0, 2, 51, scale=(40, 40, 900))
    n_mid = t.noise(16.0, 3, 52)
    pitch = MOLLE_PITCH
    k = np.round(s_ / pitch)
    on_row = _on_plate(P)                                       # MOLLE rows, not the pull tabs
    on_tack = (np.abs(k) <= 3) & (np.abs(across) < MOLLE_W / 2 - 0.002) & on_row
    tack = decals.ridge(s_ - k * pitch, 0.0017) * on_tack
    # bar tack: a dense zigzag of thread across the webbing
    zig = 0.5 + 0.5 * np.sin(across * 2 * math.pi / 0.0009)
    edge = decals.smoothstep(MOLLE_W / 2 - 0.0016, MOLLE_W / 2 - 0.0004, np.abs(across)) * on_row     # woven selvage
    base = col * (0.85 + 0.3 * rib)[:, None] * (0.94 + 0.12 * n_mid)[:, None]
    base = mix(base, col * (0.55 + 0.2 * zig)[:, None], tack)
    base = mix(base, col * 0.78, edge * 0.5)
    base = mix(base, np.array((0.29, 0.22, 0.14)), decals.smoothstep(0.2, 0.8, N[:, 2]) * 0.3)
    rough = 0.70 + 0.1 * rib
    base, rough = t.apply_holes(base, rough, False, np.array((0.01, 0.008, 0.006)))

    def height(Q):
        sq, aq = webbing_coords(Q)
        kk = np.round(sq / pitch)
        row = _on_plate(Q)
        on = (np.abs(kk) <= 3) & (np.abs(aq) < MOLLE_W / 2 - 0.002) & row
        tk = decals.ridge(sq - kk * pitch, 0.0017) * on
        twill = 0.00005 * np.sin(aq * 2 * math.pi / 0.0012)                       # ribs along the webbing
        sel = -0.00012 * decals.smoothstep(MOLLE_W / 2 - 0.0016, MOLLE_W / 2 - 0.0004, np.abs(aq)) * row
        return twill + sel + tk * (0.0001 * np.sin(aq * 2 * math.pi / 0.0009) - 0.00015) + t.holes(Q, False)[4]

    return base, rough, np.zeros(len(P)), height, N


def pouch_coords(P):
    """For each point: its pouch index (-1 = the ID panel), local coordinates in that pouch's body
    and flap frames."""
    fr = PlateFrame(True)
    best = np.full(len(P), 1e9)
    k_of = np.full(len(P), -1)
    lb = np.zeros_like(P)
    lf = np.zeros_like(P)
    for k, s_ in enumerate(POUCH_S):
        R = fr.rotation(s_)
        cb = fr.point(s_, PLATE_R + PLATE_T + 0.019, 1.192)
        cf = fr.point(s_, PLATE_R + PLATE_T + 0.021, 1.250)
        qb = (P - cb) @ R
        qf = (P - cf) @ R
        d = np.abs(qb[:, 0])
        m = (d < best) & (P[:, 2] < 1.30)
        best = np.where(m, d, best)
        k_of = np.where(m, k, k_of)
        lb = np.where(m[:, None], qb, lb)
        lf = np.where(m[:, None], qf, lf)
    return k_of, lb, lf


NAME_TAPE = None


def layer_pouch(t):
    global NAME_TAPE
    P, N = t.P, t.N
    col = np.array(SURF["Pouch"][1])
    fleck = t.noise(500.0, 2, 61)
    n_mid = t.noise(16.0, 3, 62)
    k_of, lb, lf = pouch_coords(P)
    pouch = k_of >= 0
    flap_face = pouch & (lf[:, 1] > 0.012) & (lf[:, 2] > -0.02)
    body_face = pouch & (lb[:, 1] > 0.010) & ~flap_face
    flap_edge = flap_face & (lf[:, 2] < -0.011)
    tab = pouch & (np.abs(lb[:, 0]) < 0.008) & (lf[:, 2] > -0.03) & (lf[:, 2] < -0.008) & (lb[:, 1] > 0.01)
    base = col * (0.88 + 0.24 * fleck)[:, None] * (0.94 + 0.12 * n_mid)[:, None]
    base = mix(base, col * 0.8, flap_edge)
    base = mix(base, np.array(SURF["Webbing"][1]) * 0.9, tab)
    rough = 0.84 + 0.08 * fleck
    thr = np.maximum.reduce([
        stitch_rows(lf[:, 2] + 0.0125, lf[:, 0], (0.0,))[0] * flap_face,
        stitch_rows(np.abs(lb[:, 0]) - 0.030, lb[:, 2], (0.0,))[0] * body_face,
        stitch_rows(np.abs(lb[:, 0]) - 0.0065, lf[:, 2], (0.0,))[0] * tab])
    base = mix(base, col * 0.6, thr * 0.8)
    # hook-and-loop ID panel with a name tape
    panel = ~pouch
    fr = PlateFrame(True)
    s_, _, z_ = fr.coords(P)
    loop = t.noise(900.0, 2, 63)
    base = mix(base, np.array((0.15, 0.105, 0.058)) * (0.8 + 0.4 * loop)[:, None], panel)
    rough = mix(rough, 0.97, panel)
    if NAME_TAPE is None:
        NAME_TAPE = decals.text_image("RANGE 3", 120, FONTS, pad=6, stretch=0.9)
    x0, x1, z0, z1 = -0.055, 0.055, 1.382, 1.413
    tape_ = panel & (s_ > x0) & (s_ < x1) & (z_ > z0) & (z_ < z1)
    txt = decals.sample_image(NAME_TAPE, (s_ - x0 + 0.004) / (x1 - x0 + 0.008), (z1 - z_ + 0.002) / (z1 - z0 + 0.004)) * tape_
    edge = tape_ * (decals.ridge(s_ - x0, 0.0015) + decals.ridge(s_ - x1, 0.0015) + decals.ridge(z_ - z0, 0.0015) + decals.ridge(z_ - z1, 0.0015))
    base = mix(base, col * (0.95 + 0.1 * fleck)[:, None], tape_)
    base = mix(base, np.array((0.018, 0.017, 0.016)), np.clip(txt * 1.2, 0, 1))
    base = mix(base, col * 0.55, np.clip(edge, 0, 1))
    rough = mix(rough, 0.8, tape_)
    sun = decals.smoothstep(0.1, 0.8, N[:, 2])
    base = mix(base, np.array((0.30, 0.235, 0.155)), sun * 0.3)
    dirt = decals.smoothstep(0.9, 0.55, t.cav) * decals.smoothstep(0.3, 0.75, n_mid)
    base = mix(base, GRIME * 0.7, dirt * 0.5)
    base, rough = t.apply_holes(base, rough, False, np.array((0.01, 0.008, 0.006)))

    def height(Q):
        k2, lb2, lf2 = pouch_coords(Q)
        pch = k2 >= 0
        ff = pch & (lf2[:, 1] > 0.012) & (lf2[:, 2] > -0.02)
        bf = pch & (lb2[:, 1] > 0.010) & ~ff
        h = stitch_rows(lf2[:, 2] + 0.0125, lf2[:, 0], (0.0,))[1] * ff
        h += stitch_rows(np.abs(lb2[:, 0]) - 0.030, lb2[:, 2], (0.0,))[1] * bf
        h += 0.0004 * decals.smoothstep(-0.009, -0.012, lf2[:, 2]) * ff                    # folded flap edge
        s2, _, z2 = fr.coords(Q)
        tp = ~pch & (s2 > x0) & (s2 < x1) & (z2 > z0) & (z2 < z1)
        h += tp * (0.0006 + 0.00025 * decals.sample_image(NAME_TAPE, (s2 - x0 + 0.004) / (x1 - x0 + 0.008),
                                                           (z1 - z2 + 0.002) / (z1 - z0 + 0.004)))
        h += t.holes(Q, False)[4]
        return h

    return base, rough, np.zeros(len(P)), height, N


# --- stand -------------------------------------------------------------------------------------
STENCIL = None


def layer_steel(t):
    global STENCIL
    P, N = t.P, t.N
    x, y, z = P.T
    col = np.array(SURF["Steel"][1])
    n_mid = t.noise(20.0, 3, 71)
    n_fine = t.noise(200.0, 3, 72)
    top = (N[:, 2] > 0.9) & (np.abs(z) < 0.002)
    # scratches: long thin marks in a few directions across the plate's top
    scr = np.zeros(len(P))
    for k in range(3):
        a = 0.4 + k * 1.1
        u = x * math.cos(a) + y * math.sin(a)
        v = -x * math.sin(a) + y * math.cos(a)
        lines = t.noise(1.0, 2, 73 + k, P=np.stack([u * 5, v * 700, np.zeros_like(u)], 1))
        scr = np.maximum(scr, decals.smoothstep(0.78, 0.84, lines) * decals.smoothstep(0.45, 0.7, t.noise(9.0, 2, 80 + k)))
    wear = np.clip(np.clip(t.edge * 4, 0, 1) * decals.smoothstep(0.35, 0.65, n_mid + n_fine * 0.3) + scr * top, 0, 1)
    bare = np.array((0.50, 0.50, 0.51))
    rust = decals.smoothstep(0.92, 0.7, t.cav) * decals.smoothstep(0.45, 0.8, n_mid) + wear * decals.smoothstep(0.6, 0.85, n_fine) * 0.6
    rust = np.clip(rust, 0, 1)
    base = col * (0.9 + 0.2 * n_fine)[:, None]
    base = mix(base, bare, wear)
    base = mix(base, np.array((0.20, 0.085, 0.035)) * (0.7 + 0.6 * n_fine)[:, None], rust * 0.8)
    metal = wear * (1 - rust)
    rough = 0.48 + 0.08 * n_fine
    rough = mix(rough, 0.32, wear)
    rough = mix(rough, 0.9, rust)
    # range number sprayed through a stencil at the front of the plate
    if STENCIL is None:
        STENCIL = decals.text_image("03", 160, FONTS, pad=10)
    x0, x1, y0, y1 = 0.075, 0.195, -0.245, -0.185
    paint = decals.sample_image(STENCIL, (x - x0) / (x1 - x0), (y1 - y) / (y1 - y0)) * top
    over = decals.smoothstep(0.2, 0.7, t.noise(300.0, 2, 90)) * 0.15                  # overspray speckle
    paint = np.clip(paint * (0.85 + 0.3 * n_fine) + over * decals.sample_image(STENCIL, (x - x0) / (x1 - x0) * 0.96 + 0.02,
                                                                                  (y1 - y) / (y1 - y0) * 0.9 + 0.05) * top, 0, 1)
    paint *= 1 - decals.smoothstep(0.5, 0.8, scr + wear * 0.5)
    base = mix(base, np.array((0.62, 0.60, 0.55)), paint)
    rough = mix(rough, 0.7, paint)
    metal = metal * (1 - paint)
    dust = top * decals.smoothstep(0.4, 0.8, n_mid) * 0.5
    base = mix(base, np.array((0.25, 0.23, 0.2)), dust * 0.4)

    def height(Q):
        return -0.00008 * np.clip(np.clip(t.edge * 4, 0, 1) * decals.smoothstep(0.35, 0.65, t.noise(20.0, 3, 71, P=Q)), 0, 1)

    return base, rough, metal, height, N


def layer_zinc(t):
    n = t.noise(60.0, 3, 91)
    base = np.array(SURF["Zinc"][1]) * (0.85 + 0.3 * n)[:, None]
    base = mix(base, GRIME * 0.5, decals.smoothstep(0.9, 0.6, t.cav) * 0.6)
    return base, 0.3 + 0.2 * n, np.ones(len(t.P)), (lambda Q: np.zeros(len(Q))), t.N


LAYERS = (("Paint", layer_paint), ("Shirt", layer_shirt), ("Denim", layer_denim), ("Cordura", layer_bag),
          ("Webbing", layer_webbing), ("Pouch", layer_pouch), ("Steel", layer_steel), ("Zinc", layer_zinc))


RGSS = ((-0.125, -0.375), (0.375, -0.125), (0.125, 0.375), (-0.375, 0.125))     # rotated-grid subsamples


def antialias_normals(Nd, P, N, I, AO, sel, shape, fields, hits, marks, cos_max=0.94):
    """Supersample the detail normals where they turn sharply from one texel to the next
    (hole rims, the mouth line, stitches): sampled once per texel, those edges alias into
    staircases along the texel grid. The subsamples sit on the texel's footprint, which
    the neighbouring texels' positions give."""
    t0 = time.time()
    h, w = shape
    index = np.full((h, w), -1, np.int64)
    index[sel] = np.arange(len(P))
    ys, xs = sel

    def neighbour(dy, dx):
        y2, x2 = ys + dy, xs + dx
        inside = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)
        j = np.full(len(P), -1, np.int64)
        j[inside] = index[y2[inside], x2[inside]]
        ok = j >= 0
        j = np.where(ok, j, 0)
        # same material and the same UV island (adjacent on the surface, not just in the atlas)
        ok &= (I[j] == I) & (np.linalg.norm(P[j] - P, axis=1) < 0.004)
        return j, ok

    right, left, down, up = neighbour(0, 1), neighbour(0, -1), neighbour(1, 0), neighbour(-1, 0)
    worst = np.ones(len(P))
    for j, ok in (right, left, down, up):
        worst = np.where(ok, np.minimum(worst, np.sum(Nd * Nd[j], 1)), worst)
    edge = worst < cos_max

    def step(fwd, back):
        (jf, of), (jb, ob) = fwd, back
        return np.where(of[:, None], P[jf] - P, np.where(ob[:, None], P - P[jb], 0.0))

    du, dv = step(right, left), step(down, up)
    acc = Nd.copy()
    for name, fn in LAYERS:
        m = edge & (I == SURF[name][0])
        if not m.any():
            continue
        for a, b in RGSS:
            Q = P[m] + du[m] * a + dv[m] * b
            _, _, _, height, n0 = fn(Texels(name, Q, N[m], AO[m], fields, hits, marks))
            acc[m] += decals.perturb(Q, n0, height)
    print(f"  antialiased {int(edge.sum())} edge texels in {time.time() - t0:.1f}s", flush=True)
    return acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-9)


def composite(ids, cover, aoe, pos, nrm, fields, hits, marks):
    sel = np.nonzero(cover)
    P = pos[sel].astype(np.float64) - (0, 0, BASE_T)
    N = nrm[sel].astype(np.float64)
    I = ids[sel]
    AO = aoe[sel].astype(np.float64)
    M = len(P)
    base = np.zeros((M, 3))
    rough = np.zeros(M)
    metal = np.zeros(M)
    Nd = N.copy()
    for name, fn in LAYERS:
        m = I == SURF[name][0]
        if not m.any():
            continue
        t0 = time.time()
        tx = Texels(name, P[m], N[m], AO[m], fields, hits, marks)
        b, r, mt, height, n0 = fn(tx)
        base[m], rough[m], metal[m] = b, r, mt
        Nd[m] = decals.perturb(tx.P, n0, height)
        print(f"  layer {name:8s} {int(m.sum()):8d} texels {time.time() - t0:5.1f}s", flush=True)
    Nd = antialias_normals(Nd, P, N, I, AO, sel, ids.shape, fields, hits, marks)
    occ = np.clip(AO[:, 0] * (0.55 + 0.45 * AO[:, 2]), 0, 1)
    base *= (0.82 + 0.18 * AO[:, 2])[:, None]
    h, w = ids.shape
    out_b = np.zeros((h, w, 3), np.float32)
    out_o = np.zeros((h, w, 3), np.float32)
    out_n = np.zeros((h, w, 3), np.float32)
    out_b[sel] = base
    out_o[sel] = np.stack([occ, np.clip(rough, 0.05, 1), np.clip(metal, 0, 1)], -1)
    out_n[sel] = Nd
    out_b = tb.dilate(out_b, cover, 12)
    out_o = tb.dilate(out_o, cover, 12)
    out_n = tb.dilate(out_n, cover, 4)
    return tb.linear_to_srgb(out_b), out_o, out_n


def bake_detail_normals(objs, size, Nd):
    """Tangent-space normal map of the detail normals (object space) plus each material's micro relief."""
    img = tb.new_image("detail_nrm", size)
    rgba = np.concatenate([Nd * 0.5 + 0.5, np.ones(Nd.shape[:2] + (1,), np.float32)], -1)
    img.pixels.foreach_set(rgba.astype(np.float32).ravel())

    def build(nt, m):
        uv = nt.nodes.new("ShaderNodeUVMap")
        uv.uv_map = tb.UV
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.interpolation = "Closest"
        nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
        mad = nt.nodes.new("ShaderNodeVectorMath")
        mad.operation = "MULTIPLY_ADD"
        mad.inputs[1].default_value = (2.0, 2.0, 2.0)
        mad.inputs[2].default_value = (-1.0, -1.0, -1.0)
        nt.links.new(tex.outputs["Color"], mad.inputs[0])
        nrm = nt.nodes.new("ShaderNodeVectorMath")
        nrm.operation = "NORMALIZE"
        nt.links.new(mad.outputs["Vector"], nrm.inputs[0])
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        normal = nrm.outputs["Vector"]
        scale, strength, distortion = MICRO.get(m.name, (0, 0, 0))
        if strength > 0:
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
            nt.links.new(normal, bump.inputs["Normal"])
            normal = bump.outputs["Normal"]
        nt.links.new(normal, bsdf.inputs["Normal"])
        return bsdf.outputs["BSDF"]

    with tb._Swap(objs, build):
        out = tb.bake(objs, tb.new_image("bake_tnrm", size), "NORMAL", samples=4, normal_space="TANGENT")
    bpy.data.images.remove(img)
    return out[..., :3]


def bake_textures(objs, out, size, fields, hits, marks):
    t0 = time.time()
    tb.setup(samples=16)
    info = tb.atlas_uvs(objs, resolution=size, padding=max(4, size // 256),
                        density={"Body": 1.3, "Carrier": 1.0, "Pouches": 1.0, "Webbing": 1.0,
                                 "Shirt": 0.85, "Jeans": 0.85, "Stand": 0.7})
    print(f"atlas: {info}", flush=True)
    corners = np.array([o.matrix_world @ Vector(v) for o in objs for v in o.bound_box])
    bmin, bmax = corners.min(0).astype(np.float32) - 0.001, corners.max(0).astype(np.float32) + 0.001
    ids, cover = tb.bake_ids(objs, size, {n: v[0] for n, v in SURF.items()})
    aoe = tb.bake_ao_edge(objs, size // 2, samples=24, ao_dist=0.08, cavity_dist=0.006, bevel_radius=0.0015)
    aoe = tb.upsample(tb.dilate(aoe, tb.upsample(cover.astype(np.float32), size // 2) > 0.5, 4), size)
    pos, nrm = tb.bake_position_normal(objs, size, bmin, bmax)
    t1 = time.time()
    base, orm, Nd = composite(ids, cover, aoe, pos, nrm, fields, hits, marks)
    print(f"composite in {time.time() - t1:.1f}s", flush=True)
    tn = bake_detail_normals(objs, size, Nd)
    tn[~cover] = (0.5, 0.5, 1.0)
    tn = tb.dilate(tn, cover, 12)
    paths = {k: os.path.join(out, f"mannequin_{k}.jpg") for k in ("basecolor", "orm", "normal")}
    tb.save(base, paths["basecolor"], 90)
    tb.save(orm, paths["orm"], 90)
    tb.save(tn, paths["normal"], 94)
    print(f"textures in {time.time() - t0:.1f}s", flush=True)
    return paths


# ----------------------------------------------------------------------------- main
def build(args):
    t0 = time.time()
    f = body_field()
    print(f"field {tuple(f.n)} in {time.time() - t0:.1f}s", flush=True)
    mats = {name: c.material(name, col, roughness=rg, metallic=1.0 if name == "Zinc" else 0.0)
            for name, (_i, col, rg) in SURF.items()}
    if args.parts == "body":
        body = field_mesh("Body", f, mats["Paint"])
        relax_and_reduce(body, 22000)
        return {"body": f}, [body]
    t1 = time.time()
    sh = shirt_field(f)
    jn = jeans_field(f)
    bag_f, pouch_f = carrier_fields(f)
    print(f"clothes and carrier fields in {time.time() - t1:.1f}s", flush=True)
    body = field_mesh("Body", f, mats["Paint"])
    relax_and_reduce(body, 22000)
    body_under_clothes(body, [sh, jn])
    shirt = field_mesh("Shirt", sh, mats["Shirt"])
    relax_and_reduce(shirt, 12000)
    jeans = field_mesh("Jeans", jn, mats["Denim"])
    relax_and_reduce(jeans, 14000)
    bag = field_mesh("Carrier", bag_f, mats["Cordura"])
    relax_and_reduce(bag, 9000, iterations=2)
    web = molle_webbing(mats["Webbing"])
    pouch = field_mesh("Pouches", pouch_f, mats["Pouch"])
    relax_and_reduce(pouch, 4000, iterations=1)
    figure = [body, shirt, jeans, bag, web, pouch]
    for o in figure:
        o.data.transform(Matrix.Translation((0, 0, BASE_T)))
    parts = figure + [stand(mats, jn)]
    fields = {"body": f, "shirt": sh, "jeans": jn, "bag": bag_f, "pouch": pouch_f}
    return fields, parts


def main():
    args = c.parse_args(lambda p: (p.add_argument("--size", type=int, default=2048),
                                   p.add_argument("--preview", default=""),
                                   p.add_argument("--views", default="front,side,three_quarter"),
                                   p.add_argument("--no-bake", action="store_true"),
                                   p.add_argument("--parts", default="all", help="all | body (quick sculpt checks)")))
    c.reset_scene()
    fields, parts = build(args)
    for o in parts:
        print(f"  {o.name}: {sum(len(p.vertices) - 2 for p in o.data.polygons)} tris", flush=True)
    if args.parts == "all" and not args.no_bake:
        hits = shoot(parts)
        marks = paint_marks(parts, hits)
        print(f"{len(hits)} hits: " + ", ".join(f"{m}={sum(h.mat == m for h in hits)}" for m in SURF), flush=True)
        paths = bake_textures(parts, args.out, args.size, fields, hits, marks)
        pbr = tb.pbr_material("MannequinPBR", paths["basecolor"], paths["orm"], paths["normal"])
        for o in parts:
            o.data.materials.clear()
            o.data.materials.append(pbr)
            for p in o.data.polygons:
                p.material_index = 0
    if args.preview:
        preview(args.preview, parts, args.views.split(","))
    if args.parts == "all" and not args.no_bake:
        merged = c.join("Mannequin", parts)
        merged.data.validate(clean_customdata=False)
        c.export_glb(os.path.join(args.out, "mannequin.glb"), [merged], export_tangents=True)


if __name__ == "__main__":
    main()
