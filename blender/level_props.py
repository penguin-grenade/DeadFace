"""Procedural set dressing for the warehouse: structure, racking, pallets, drums,
lamps, pipes and furniture. Every function appends parts to per-material batches
on a `levelkit.Kit`, so thousands of parts still end up as a few meshes."""
from __future__ import annotations

import math
import random

from mathutils import Euler, Matrix, Vector

from levelkit import angle_profile, c_profile, i_profile

LM_BIG, LM_PROP, LM_SMALL, LM_DETAIL, LM_TINY = 1.0, 0.45, 0.4, 0.3, 0.12
TAU = 2 * math.pi


class Frame:
    """Local placement: position + yaw (+ optional extra rotation)."""

    def __init__(self, origin=(0, 0, 0), yaw=0.0, rot=None):
        self.o = Vector(origin)
        self.R = Matrix.Rotation(yaw, 3, "Z")
        if rot is not None:
            self.R = self.R @ Euler(rot).to_matrix()

    def p(self, x, y, z):
        return self.o + self.R @ Vector((x, y, z))

    def r(self, rx=0.0, ry=0.0, rz=0.0):
        return self.R @ Euler((rx, ry, rz)).to_matrix()

    def m4(self, x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0, s=1.0):
        sc = s if isinstance(s, (tuple, list)) else (s, s, s)
        return Matrix.Translation(self.p(x, y, z)) @ self.r(rx, ry, rz).to_4x4() @ Matrix.Diagonal((*sc, 1.0))


def jitter(rng, a):
    return (rng.random() * 2 - 1) * a


# --------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------

def w_column(k, x, y, z0, z1, rng, yaw=0.0):
    st = k.B("steel")
    f = Frame((x, y, 0), yaw)
    st.prism(i_profile(0.3, 0.3, 0.012, 0.018), f.p(0, 0, z0 + 0.025), f.p(0, 0, z1), up=tuple(f.R @ Vector((1, 0, 0))), lm=LM_SMALL)
    # Base plate, anchor bolts with nuts and washers, grout pad.
    k.B("concrete").box(f.p(0, 0, 0.012), (0.5, 0.5, 0.024), f.r(), lm=LM_DETAIL)
    st.box(f.p(0, 0, 0.036), (0.42, 0.42, 0.024), f.r(), lm=LM_DETAIL)
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = f.p(sx * 0.16, sy * 0.16, 0.048)
            st.rod(p, p + Vector((0, 0, 0.006)), 0.028, sides=12, caps=True, lm=LM_TINY)
            st.rod(p + Vector((0, 0, 0.006)), p + Vector((0, 0, 0.03)), 0.019, sides=6, caps=True, lm=LM_TINY)
            st.rod(p + Vector((0, 0, 0.03)), p + Vector((0, 0, 0.05)), 0.01, sides=8, caps=True, lm=LM_TINY)
    # Cap plate.
    st.box(f.p(0, 0, z1 + 0.01), (0.34, 0.34, 0.02), f.r(), lm=LM_DETAIL)
    k.collide_box(f.p(0, 0, (z0 + z1) / 2), (0.3, 0.3, z1 - z0), f.r(), "metal")


def girder(k, y, x0, x1, z_top, depth=0.6, width=0.22):
    st = k.B("steel")
    zc = z_top - depth / 2
    st.prism(i_profile(depth, width, 0.011, 0.017), (x0, y, zc), (x1, y, zc), up=(0, 0, 1), lm=LM_SMALL)
    # Web stiffeners every 1.5 m.
    x = x0 + 0.75
    while x < x1 - 0.3:
        for s in (-1, 1):
            st.box((x, y + s * 0.055, zc), (0.012, 0.1, depth - 0.036), lm=LM_TINY)
        x += 1.5


def bar_joist(k, x, y0, y1, z_top, depth=0.45, panel=0.6):
    """Open-web steel joist spanning along Y: double-angle chords and round-bar webs."""
    st = k.B("steel", tag="joist", lm=LM_DETAIL)
    st.bevel = 0.0
    leg, t = 0.05, 0.005
    zt, zb = z_top - leg, z_top - depth
    ys, ye = y0 + 0.12, y1 - 0.12
    for s in (-1, 1):
        prof_top = [(s * 0.008 + s * py, leg - pz) for (py, pz) in angle_profile(leg, t)]
        if s > 0:
            prof_top = prof_top[::-1]
        st.prism(prof_top, (x, y0 + 0.02, zt), (x, y1 - 0.02, zt), up=(0, 0, 1))
        prof_bot = [(s * 0.008 + s * py, pz) for (py, pz) in angle_profile(0.04, 0.004)]
        if s < 0:
            prof_bot = prof_bot[::-1]
        st.prism(prof_bot, (x, ys + 0.25, zb), (x, ye - 0.25, zb), up=(0, 0, 1))
    # Webs: zig-zag round bars from the bottom chord to the top chord.
    n = max(2, int(round((ye - ys) / panel)))
    step = (ye - ys) / n
    for i in range(n):
        ya = ys + i * step
        yb = ya + step
        if i % 2 == 0:
            a, b = (x, ya, zt), (x, yb, zb + 0.04)
        else:
            a, b = (x, ya, zb + 0.04), (x, yb, zt)
        st.rod(a, b, 0.011, sides=5)
    # Bearing seats at each end.
    for yy, sgn in ((y0, 1), (y1, -1)):
        st.box((x, yy + sgn * 0.08, z_top - 0.11), (0.12, 0.16, 0.012))
        st.box((x, yy + sgn * 0.08, z_top - 0.065), (0.02, 0.16, 0.09))


def deck(k, x0, x1, y0, y1, z, holes=(), pitch=0.15, rib=0.038):
    """Trapezoidal roof deck seen from below. Ribs run along X. `holes` are (x0, x1, y0, y1)."""
    b = k.B("deck", lm=0.6, sharp=80, weld=True)
    y = y0
    top_w, bot_w = 0.06, 0.05
    while y < y1 - 1e-6:
        y_next = min(y + pitch, y1)
        # Profile within one pitch: bottom flat, slope up, top flat, slope down.
        prof = [(y, z), (y + bot_w * 0.5, z), (y + bot_w * 0.5 + 0.02, z + rib), (y + bot_w * 0.5 + 0.02 + top_w, z + rib),
                (y + bot_w * 0.5 + 0.04 + top_w, z), (y_next, z)]
        cuts = [(hx0, hx1) for (hx0, hx1, hy0, hy1) in holes if hy0 < y_next and hy1 > y]
        spans = _subtract([(x0, x1)], cuts)
        for (sx0, sx1) in spans:
            for (pa, pb) in zip(prof, prof[1:]):
                if pb[0] - pa[0] < 1e-6 and abs(pb[1] - pa[1]) < 1e-6:
                    continue
                # Faces point down (seen from inside the hall). Flattened copies keep the
                # whole ribbed sheet in one lightmap chart.
                b.quad((sx0, pa[0], pa[1]), (sx0, pb[0], pb[1]), (sx1, pb[0], pb[1]), (sx1, pa[0], pa[1]), uv="world",
                       flat=((sx0, pa[0], z), (sx0, pb[0], z), (sx1, pb[0], z), (sx1, pa[0], z)))
        y = y_next


def _subtract(spans, cuts):
    out = []
    for (a, b) in spans:
        segs = [(a, b)]
        for (c0, c1) in cuts:
            nxt = []
            for (s0, s1) in segs:
                if c1 <= s0 or c0 >= s1:
                    nxt.append((s0, s1))
                    continue
                if c0 > s0:
                    nxt.append((s0, c0))
                if c1 < s1:
                    nxt.append((c1, s1))
            segs = nxt
        out.extend(segs)
    return [(s0, s1) for (s0, s1) in out if s1 - s0 > 1e-4]


def cladding(k, axis, fixed, s0, s1, z0, z1, inward, openings=(), pitch=0.305):
    """Ribbed metal wall panel seen from inside. axis='x' runs along X at y=fixed, 'y' along Y at x=fixed.
    `inward` is +1/-1 for the direction the inner face points along the other axis."""
    b = k.B("corrugated", lm=0.85, sharp=80, weld=True)
    rib = 0.035
    s = s0
    while s < s1 - 1e-6:
        sn = min(s + pitch, s1)
        prof = [(s, 0.0), (s + 0.1, 0.0), (s + 0.12, rib), (s + 0.16, rib), (s + 0.18, 0.0), (sn, 0.0)]
        prof = [(a, d) for (a, d) in prof if a <= sn + 1e-6]
        cuts = [(oz0, oz1) for (os0, os1, oz0, oz1) in openings if os0 < sn - 1e-4 and os1 > s + 1e-4]
        spans = _subtract([(z0, z1)], cuts)
        for (za, zb) in spans:
            for (pa, pb) in zip(prof, prof[1:]):
                def P(a, d, zz):
                    off = fixed + inward * (d - 0.0)  # ribs protrude toward the inside
                    return (a, off, zz) if axis == "x" else (off, a, zz)
                qa, qb, qc, qd = P(pa[0], pa[1], za), P(pb[0], pb[1], za), P(pb[0], pb[1], zb), P(pa[0], pa[1], zb)
                fa, fb, fc, fd = P(pa[0], 0.0, za), P(pb[0], 0.0, za), P(pb[0], 0.0, zb), P(pa[0], 0.0, zb)
                # Winding so the face points inward.
                if (axis == "y") == (inward > 0):
                    b.quad(qa, qb, qc, qd, uv="world", flat=(fa, fb, fc, fd))
                else:
                    b.quad(qb, qa, qd, qc, uv="world", flat=(fb, fa, fd, fc))
        s = sn


# --------------------------------------------------------------------------------------
# Racking
# --------------------------------------------------------------------------------------

def rack_upright_frame(k, x0, x1, y, height, rng):
    """Two perforated uprights (front x0, back x1) with horizontal/diagonal bracing."""
    blue = k.B("rack_blue", lm=LM_SMALL)
    for x in (x0, x1):
        prof = [(-0.045, -0.035), (0.045, -0.035), (0.045, 0.035), (0.03, 0.035), (0.03, -0.02), (-0.03, -0.02), (-0.03, 0.035), (-0.045, 0.035)]
        blue.prism(prof, (x, y, 0.012), (x, y, height), up=(1, 0, 0))
        # Foot plate with anchor.
        blue.box((x, y, 0.006), (0.14, 0.12, 0.012), lm=LM_TINY)
        k.B("steel").rod((x + 0.045, y, 0.012), (x + 0.045, y, 0.03), 0.01, sides=6, caps=True, lm=LM_TINY)
    # Bracing (small C-channels) in a Z pattern.
    z = 0.15
    zs = [0.15]
    while z < height - 0.4:
        z += 0.6
        zs.append(min(z, height - 0.1))
    for i, z in enumerate(zs):
        blue.box_between((x0 + 0.03, y, z), (x1 - 0.03, y, z), 0.028, 0.02, up=(0, 0, 1), lm=LM_DETAIL)
        if i + 1 < len(zs):
            za, zb = z, zs[i + 1]
            a, bb = ((x0 + 0.03, y, za), (x1 - 0.03, y, zb)) if i % 2 == 0 else ((x1 - 0.03, y, za), (x0 + 0.03, y, zb))
            blue.box_between(a, bb, 0.028, 0.02, up=(0, 1, 0), lm=LM_DETAIL)


def rack_beam_pair(k, x0, x1, y0, y1, z_top):
    orange = k.B("rack_orange", lm=LM_SMALL)
    for x in (x0, x1):
        # Step beam: 110 mm tall box with a 40 mm step on the inside top.
        orange.box((x, (y0 + y1) / 2, z_top - 0.055), (0.05, y1 - y0 - 0.1, 0.11))
        for yy in (y0 + 0.06, y1 - 0.06):  # connectors
            orange.box((x, yy, z_top - 0.08), (0.06, 0.012, 0.16), lm=LM_TINY)
    # Pallet support bars.
    for f in (0.3, 0.7):
        yy = y0 + (y1 - y0) * f
        k.B("galv").box((x0 + (x1 - x0) / 2, yy, z_top - 0.02), (x1 - x0 - 0.04, 0.04, 0.03), lm=LM_TINY)


# --------------------------------------------------------------------------------------
# Pallets and loads
# --------------------------------------------------------------------------------------

def pallet(k, pos, yaw, rng, broken=False):
    """GMA-style 1.22 x 1.02 m stringer pallet; returns the top height."""
    wood = k.B("wood", lm=LM_PROP)
    f = Frame(pos, yaw)
    L, W = 1.22, 1.02
    z = 0.0
    # Bottom boards (run across the stringers).
    for x in (-0.54, 0.0, 0.54):
        wood.box(f.p(x + jitter(rng, 0.01), 0, 0.009), (0.14, W, 0.018), f.r(rz=jitter(rng, 0.01)))
    # Stringers with fork notches (as two blocks + a thinner middle).
    for y in (-0.48, 0.0, 0.48):
        yy = y + jitter(rng, 0.006)
        wood.box(f.p(0, yy, 0.018 + 0.045), (L, 0.04, 0.09), f.r(rz=jitter(rng, 0.004)))
    # Top deck boards.
    n = 7
    for i in range(n):
        if broken and rng.random() < 0.25:
            continue
        x = -L / 2 + 0.07 + i * (L - 0.14) / (n - 1)
        wood.box(f.p(x + jitter(rng, 0.008), jitter(rng, 0.006), 0.108 + 0.009), (0.14 if i in (0, n - 1) else 0.1, W + jitter(rng, 0.01), 0.018),
                 f.r(rx=jitter(rng, 0.01), rz=jitter(rng, 0.012)))
    return pos[2] + 0.126


def pallet_stack(k, pos, yaw, count, rng):
    z = pos[2]
    for i in range(count):
        z = pallet(k, (pos[0] + jitter(rng, 0.03), pos[1] + jitter(rng, 0.03), z), yaw + jitter(rng, 0.04), rng, broken=rng.random() < 0.2)
    k.collide_box((pos[0], pos[1], pos[2] + (z - pos[2]) / 2), (1.22, 1.02, z - pos[2]), (0, 0, yaw), "wood")
    return z


def carton(k, center, size, yaw, rng, tape=True, label=True):
    cb = k.B("cardboard", lm=LM_PROP)
    f = Frame(center, yaw)
    sx, sy, sz = size
    # Slight bulge/sag via a two-box stack (bottom slightly wider).
    cb.box(f.p(0, 0, 0), (sx, sy, sz), f.r(rx=jitter(rng, 0.006), ry=jitter(rng, 0.006)))
    if tape:
        k.B("plastic_yellow" if rng.random() < 0.2 else "wrap", tag="tape", lm=LM_TINY).box(f.p(0, 0, sz / 2 + 0.0008), (sx + 0.002, 0.05, 0.0016), f.r())
    if label and rng.random() < 0.7:
        side = rng.choice([-1, 1])
        k.B("paper", lm=LM_TINY).box(f.p(side * (sx / 2 + 0.0008), jitter(rng, sy * 0.2), jitter(rng, sz * 0.15)), (0.0016, 0.1, 0.07), f.r())


def carton_load(k, pos, yaw, rng, layers=3, wrap=True, footprint=(1.2, 1.0), box=(0.4, 0.33, 0.3)):
    """Cartons stacked on a pallet at pos (pallet top)."""
    f = Frame(pos, yaw)
    nx = int(footprint[0] // box[0])
    ny = int(footprint[1] // box[1])
    top = pos[2]
    for lz in range(layers):
        for ix in range(nx):
            for iy in range(ny):
                if lz == layers - 1 and rng.random() < 0.25:
                    continue
                c = f.p(-footprint[0] / 2 + box[0] * (ix + 0.5) + jitter(rng, 0.01),
                        -footprint[1] / 2 + box[1] * (iy + 0.5) + jitter(rng, 0.01),
                        box[2] * (lz + 0.5))
                carton(k, c, box, yaw + jitter(rng, 0.03), rng)
        top = pos[2] + box[2] * (lz + 1)
    if wrap:
        # Stretch wrap: a thin shell around the lower layers, slightly bulged.
        wr = k.B("wrap", lm=LM_SMALL)
        hw = (nx * box[0]) / 2 + 0.01
        hd = (ny * box[1]) / 2 + 0.01
        hz = box[2] * min(layers, 2) + 0.05
        segs = []
        for (x, y) in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)):
            segs.append((x, y))
        for i in range(4):
            (xa, ya), (xb, yb) = segs[i], segs[(i + 1) % 4]
            a0, b0 = f.p(xa, ya, -0.06), f.p(xb, yb, -0.06)
            a1, b1 = f.p(xa, ya, hz), f.p(xb, yb, hz)
            wr.quad(a0, b0, b1, a1)
    k.collide_box(f.p(0, 0, (top - pos[2]) / 2 - 0.063), (footprint[0], footprint[1], top - pos[2] + 0.126), f.r(), "cardboard")
    return top


def drum(k, pos, mat, rng, lying=False, yaw=0.0, dented=False):
    """55 gallon steel drum with rolled chimes, rolling hoops and bungs."""
    b = k.B(mat, lm=LM_PROP, sharp=60)
    R = 0.286
    prof = [(0.0, 0.004), (R - 0.02, 0.004), (R - 0.012, 0.0), (R + 0.006, 0.006), (R + 0.006, 0.018), (R, 0.026),
            (R, 0.28), (R + 0.01, 0.292), (R + 0.01, 0.302), (R, 0.314),
            (R, 0.566), (R + 0.01, 0.578), (R + 0.01, 0.588), (R, 0.6),
            (R, 0.854), (R + 0.006, 0.862), (R + 0.006, 0.876), (R - 0.012, 0.882), (R - 0.02, 0.868), (0.0, 0.868)]
    if lying:
        M = Matrix.Translation(Vector(pos) + Vector((0, 0, R + 0.006))) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(math.pi / 2, 4, "Y") @ Matrix.Translation((0, 0, -0.44))
    else:
        M = Matrix.Translation(pos) @ Matrix.Rotation(yaw, 4, "Z")
    b.lathe(prof, M, segments=28, phase=rng.random())
    # Bungs on the lid.
    for (bx, by, br) in ((0.19, 0.0, 0.032), (-0.2, 0.05, 0.02)):
        c = M @ Vector((bx, by, 0.868))
        top = M @ Vector((bx, by, 0.878))
        b.rod(c, top, br, sides=10, caps=True, lm=LM_TINY)
    if lying:
        k.collide_box(M @ Vector((0, 0, 0.44)), (0.88, 0.58, 0.58), (0, 0, yaw), "metal")
    else:
        k.collide_cyl(Vector(pos) + Vector((0, 0, 0.44)), R + 0.01, 0.88, "metal")


def tyre(k, pos, rng, lying=True, yaw=0.0, tilt=0.0):
    rub = k.B("rubber", lm=LM_PROP, sharp=70)
    rim, Ro, w = 0.205, 0.335, 0.215
    prof = [(rim, -0.08), (rim + 0.02, -0.1), (Ro - 0.04, -w / 2), (Ro - 0.012, -w / 2 + 0.012),
            (Ro, -0.07), (Ro - 0.006, -0.05), (Ro, -0.035), (Ro, -0.012), (Ro - 0.006, 0.0), (Ro, 0.012), (Ro, 0.035),
            (Ro - 0.006, 0.05), (Ro, 0.07), (Ro - 0.012, w / 2 - 0.012), (Ro - 0.04, w / 2), (rim + 0.02, 0.1), (rim, 0.08)]
    if lying:
        M = Matrix.Translation(Vector(pos) + Vector((0, 0, w / 2))) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(tilt, 4, "X")
    else:
        M = Matrix.Translation(Vector(pos) + Vector((0, 0, Ro))) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(math.pi / 2, 4, "X") @ Matrix.Rotation(tilt, 4, "Y")
    rub.lathe(prof, M, segments=32, closed=True, phase=rng.random())
    return M


def tyre_stack(k, pos, count, rng):
    z = pos[2]
    for i in range(count):
        tyre(k, (pos[0] + jitter(rng, 0.03), pos[1] + jitter(rng, 0.03), z), rng, yaw=rng.random() * TAU, tilt=jitter(rng, 0.03))
        z += 0.215
    k.collide_cyl((pos[0], pos[1], pos[2] + (z - pos[2]) / 2), 0.34, z - pos[2], "rubber")


def pillow(sx, sy, sz, rng, nu=12, nv=7, e=0.3, e2=0.55):
    """Sandbag-like superellipsoid with sag and lumps. Returns verts, faces (local, centred)."""
    verts, faces = [], []

    def sp(v, ex):
        return math.copysign(abs(v) ** ex, v)

    rings = []
    for j in range(nv + 1):
        phi = -math.pi / 2 + math.pi * j / nv
        if j in (0, nv):
            rings.append([len(verts)])
            verts.append((0.0, 0.0, sp(math.sin(phi), e2) * sz / 2 * (0.9 if j == nv else 1.0)))
            continue
        ring = []
        for i in range(nu):
            th = TAU * i / nu
            cp = sp(math.cos(phi), e2)
            x = sx / 2 * cp * sp(math.cos(th), e)
            y = sy / 2 * cp * sp(math.sin(th), e)
            z = sz / 2 * sp(math.sin(phi), e2)
            # Top sags towards the ends, sides bulge, random lumps.
            ends = (abs(x) / (sx / 2)) ** 2
            if z > 0:
                z *= 1.0 - 0.35 * ends
            lump = 1.0 + jitter(rng, 0.04)
            ring.append(len(verts))
            verts.append((x * lump, y * lump, z + jitter(rng, 0.004)))
        rings.append(ring)
    for j in range(nv):
        ra, rb = rings[j], rings[j + 1]
        for i in range(nu):
            i2 = (i + 1) % nu
            if len(ra) == 1:
                faces.append((ra[0], rb[i2], rb[i]))
            elif len(rb) == 1:
                faces.append((ra[i], ra[i2], rb[0]))
            else:
                faces.append((ra[i], ra[i2], rb[i2], rb[i]))
    return verts, faces


def sandbag(k, pos, yaw, rng, size=(0.6, 0.34, 0.15)):
    v, f = pillow(size[0] * (1 + jitter(rng, 0.05)), size[1] * (1 + jitter(rng, 0.05)), size[2] * (1 + jitter(rng, 0.1)), rng)
    M = Matrix.Translation(pos) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(jitter(rng, 0.05), 4, "X")
    k.B("sandbag", lm=0.25, sharp=89).add_mesh(v, f, M)


# --------------------------------------------------------------------------------------
# Small props
# --------------------------------------------------------------------------------------

def traffic_cone(k, pos, rng, knocked=False, yaw=0.0):
    if knocked:
        M = Matrix.Translation(Vector(pos) + Vector((0, 0, 0.13))) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(1.2, 4, "Y") @ Matrix.Translation((0, 0, -0.2))
    else:
        M = Matrix.Translation(pos) @ Matrix.Rotation(yaw, 4, "Z")
    o = k.B("plastic_orange", lm=LM_SMALL, sharp=60)
    base = [(-0.18, -0.18), (0.18, -0.18), (0.18, 0.18), (-0.18, 0.18)]
    o.box(M @ Vector((0, 0, 0.015)), (0.36, 0.36, 0.03), M.to_3x3())
    o.lathe([(0.14, 0.03), (0.12, 0.2), (0.0, 0.2)], M, segments=20)  # hidden inner cap keeps it closed
    o.lathe([(0.135, 0.03), (0.105, 0.22)], M, segments=20)
    k.B("paint_white", lm=LM_TINY, sharp=60).lathe([(0.106, 0.22), (0.074, 0.38)], M, segments=20)
    o.lathe([(0.075, 0.38), (0.028, 0.7), (0.0, 0.7)], M, segments=20)


def bollard(k, pos, rng):
    y = k.B("paint_yellow", lm=LM_SMALL, sharp=60)
    x, yy, _ = pos
    y.lathe([(0.0, 0.0), (0.084, 0.0), (0.084, 1.05), (0.07, 1.1), (0.04, 1.125), (0.0, 1.13)], Matrix.Translation(pos), segments=20)
    k.collide_cyl((x, yy, 0.565), 0.09, 1.13, "metal")


def chain(k, p0, p1, pitch=0.034, r=0.004, mat="galv"):
    b = k.B(mat, tag="chain", lm=LM_TINY, sharp=89)
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    n = max(1, int(d.length / pitch))
    ax = d.normalized()
    base = ax.to_track_quat("Z", "Y").to_matrix().to_4x4()
    circle = [(0.012 + r * math.cos(t), r * math.sin(t)) for t in [TAU * i / 6 for i in range(6)]]
    for i in range(n):
        c = p0 + ax * (pitch * (i + 0.5))
        M = Matrix.Translation(c) @ base @ Matrix.Rotation(math.pi / 2 * (i % 2), 4, "Z") @ Matrix.Rotation(math.pi / 2, 4, "X") @ Matrix.Diagonal((1.0, 1.55, 1.0, 1.0))
        b.lathe(circle, M, segments=8, closed=True)


def pipe_run(k, pts, r, mat, sides=12, lm=LM_DETAIL, elbow=True):
    b = k.B(mat, lm=lm, sharp=60)
    pts = [Vector(p) for p in pts]
    for a, c in zip(pts, pts[1:]):
        b.rod(a, c, r, sides=sides)
    if elbow:
        for p in pts[1:-1]:
            b.lathe([(0.0, -r * 1.25), (r * 1.25, -r * 0.6), (r * 1.25, r * 0.6), (0.0, r * 1.25)], Matrix.Translation(p), segments=sides, lm=LM_TINY)
    for p in (pts[0], pts[-1]):
        pass


def lamp_dome(k, pos, hang_z, rng, chain_len=None, bulb="emit_lamp"):
    """Industrial aluminium high-bay dome, hung on a chain from the joists. pos = bulb position."""
    x, y, z = pos
    alu = k.B("paint_grey", lm=LM_SMALL, sharp=60)
    # Reflector dome: open at the bottom, rim lip.
    dome = [(0.3, -0.06), (0.29, -0.055), (0.27, 0.0), (0.22, 0.09), (0.15, 0.15), (0.1, 0.18), (0.095, 0.2)]
    alu.lathe(dome, Matrix.Translation((x, y, z)), segments=32)
    # Bulb (glows in the game; the spot light carries its light).
    k.B(bulb, kind="emissive", sharp=89).lathe([(0.0, -0.075), (0.045, -0.06), (0.062, -0.02), (0.058, 0.02), (0.035, 0.06), (0.03, 0.1), (0.0, 0.1)],
                                                     Matrix.Translation((x, y, z)), segments=16)
    # Inner face of the reflector (bright spun aluminium).
    inner = [(0.09, 0.19), (0.14, 0.14), (0.21, 0.08), (0.26, 0.0), (0.285, -0.05)]
    k.B("galv", lm=LM_SMALL, sharp=60).lathe(inner, Matrix.Translation((x, y, z)), segments=32)
    # Driver housing with fins on top.
    dk = k.B("paint_dark", lm=LM_DETAIL, sharp=60)
    dk.lathe([(0.0, 0.2), (0.12, 0.2), (0.12, 0.32), (0.06, 0.36), (0.0, 0.36)], Matrix.Translation((x, y, z)), segments=24)
    for i in range(12):
        t = TAU * i / 12
        c, s = math.cos(t), math.sin(t)
        dk.box((x + c * 0.125, y + s * 0.125, z + 0.26), (0.03, 0.006, 0.1), Euler((0, 0, t)).to_matrix(), lm=LM_TINY)
    # Hook, chain, and power cord.
    top = Vector((x, y, z + 0.37))
    k.B("steel").rod(top, top + Vector((0, 0, 0.05)), 0.008, sides=6, lm=LM_TINY)
    chain(k, top + Vector((0, 0, 0.05)), (x, y, hang_z))
    cord_pts = [Vector((x + 0.05, y, z + 0.33))]
    for i in range(1, 7):
        t = i / 6
        cord_pts.append(Vector((x + 0.05 + 0.02 * math.sin(t * 5), y + 0.02 * t, z + 0.33 + (hang_z - z - 0.33) * t)))
    pipe_run(k, cord_pts, 0.006, "plastic_black", sides=6, lm=LM_TINY, elbow=False)


def work_light(k, pos, target, rng):
    """Twin-head halogen work light on a tripod. Returns the head position (between lamps)."""
    x, y, _ = pos
    st = k.B("paint_yellow", lm=LM_DETAIL, sharp=60)
    dk = k.B("paint_dark", lm=LM_DETAIL, sharp=60)
    hub = Vector((x, y, 0.62))
    for i in range(3):
        t = TAU * i / 3 + 0.3
        foot = Vector((x + math.cos(t) * 0.55, y + math.sin(t) * 0.55, 0.0))
        st.rod(foot + Vector((0, 0, 0.012)), hub, 0.014, sides=8)
        k.B("rubber").lathe([(0.0, 0.0), (0.025, 0.0), (0.022, 0.025), (0.0, 0.025)], Matrix.Translation(foot), segments=10, lm=LM_TINY)
        mid = foot.lerp(hub, 0.45)
        st.rod(mid, Vector((x, y, 0.95)), 0.008, sides=6)
    st.rod(Vector((x, y, 0.5)), Vector((x, y, 1.95)), 0.018, sides=10, caps=True)
    dk.rod(Vector((x, y, 1.45)), Vector((x, y, 1.52)), 0.026, sides=10, caps=True)
    # T-bar and two heads aimed at the target.
    aim = (Vector(target) - Vector((x, y, 1.95))).normalized()
    side = Vector((0, 0, 1)).cross(aim).normalized()
    st.rod(Vector((x, y, 1.95)) - side * 0.32, Vector((x, y, 1.95)) + side * 0.32, 0.014, sides=8, caps=True)
    heads = []
    for s in (-1, 1):
        c = Vector((x, y, 1.95)) + side * (s * 0.24) + Vector((0, 0, 0.06))
        R = aim.to_track_quat("Y", "Z").to_matrix()
        dk.box(c, (0.2, 0.09, 0.16), R, lm=LM_DETAIL)
        # Cage bars on the front.
        front = c + aim * 0.047
        for i in range(-2, 3):
            p0 = front + (R @ Vector((i * 0.035, 0, -0.075)))
            p1 = front + (R @ Vector((i * 0.035, 0, 0.075)))
            dk.rod(p0 + aim * 0.012, p1 + aim * 0.012, 0.003, sides=5, lm=LM_TINY)
        # Glass front (emissive in the game, not baked; the spot light carries the light).
        k.B("emit_worklight", kind="emissive").box(front, (0.17, 0.004, 0.13), R)
        heads.append(c)
    # Power cord across the floor to the wall.
    k.collide_box((x, y, 0.9), (0.4, 0.4, 1.8), (0, 0, 0), "metal")
    return (heads[0] + heads[1]) / 2 + aim * 0.06, aim


def barricade(k, pos, yaw, rng, width=1.22, height=2.2, port=None):
    """Plywood tactical barricade on 2x4 framing with braced feet. `port` = (x, z, w, h) cut-out."""
    f = Frame(pos, yaw)
    ply = k.B("plywood", lm=LM_PROP)
    wood = k.B("wood", lm=LM_SMALL)
    t = 0.018
    # Plywood face with an optional port (built as up to four panels around the hole).
    if port:
        px, pz, pw, ph = port
        panels = [(-width / 2, px - pw / 2, 0.05, height), (px + pw / 2, width / 2, 0.05, height),
                  (px - pw / 2, px + pw / 2, 0.05, pz - ph / 2), (px - pw / 2, px + pw / 2, pz + ph / 2, height)]
    else:
        panels = [(-width / 2, width / 2, 0.05, height)]
    for (x0, x1, z0, z1) in panels:
        if x1 - x0 < 0.01 or z1 - z0 < 0.01:
            continue
        ply.box(f.p((x0 + x1) / 2, 0, (z0 + z1) / 2), (x1 - x0, t, z1 - z0), f.r())
    # Studs behind (2x4 = 38 x 89 mm).
    yb = t / 2 + 0.0445
    for x in (-width / 2 + 0.019, width / 2 - 0.019):
        wood.box(f.p(x, yb, height / 2 + 0.025), (0.038, 0.089, height - 0.05), f.r())
    for z in (0.07, height - 0.02) + ((port[1] - port[3] / 2 - 0.019, port[1] + port[3] / 2 + 0.019) if port else ()):
        wood.box(f.p(0, yb, z), (width - 0.076, 0.089, 0.038), f.r())
    # Feet and diagonal braces.
    for x in (-width / 2 + 0.1, width / 2 - 0.1):
        wood.box(f.p(x, 0.25, 0.019), (0.089, 0.9, 0.038), f.r())
        a, b = f.p(x, 0.65, 0.04), f.p(x, yb + 0.03, 1.2)
        wood.box_between(a, b, 0.038, 0.089, up=tuple(f.R @ Vector((0, 1, 0))))
    k.collide_box(f.p(0, 0.0, height / 2), (width, 0.05, height), f.r(), "wood")


def workbench(k, pos, yaw, rng, length=2.4, depth=0.76, height=0.92):
    f = Frame(pos, yaw)
    st = k.B("paint_grey", lm=LM_SMALL)
    top = k.B("plywood", lm=LM_PROP)
    top.box(f.p(0, 0, height - 0.02), (length, depth, 0.04), f.r())
    for sx in (-1, 1):
        for sy in (-1, 1):
            st.box(f.p(sx * (length / 2 - 0.05), sy * (depth / 2 - 0.05), (height - 0.04) / 2), (0.05, 0.05, height - 0.04), f.r())
    for sy in (-1, 1):
        st.box(f.p(0, sy * (depth / 2 - 0.05), 0.15), (length - 0.1, 0.04, 0.04), f.r())
    top.box(f.p(0, 0, 0.19), (length - 0.1, depth - 0.1, 0.018), f.r())  # lower shelf
    # Vise at one end.
    dk = k.B("paint_blue", lm=LM_DETAIL, sharp=60)
    vx = length / 2 - 0.25
    dk.box(f.p(vx, depth / 2 - 0.1, height + 0.05), (0.16, 0.2, 0.1), f.r())
    dk.box(f.p(vx, depth / 2 + 0.02, height + 0.07), (0.16, 0.05, 0.1), f.r())
    k.B("steel").rod(f.p(vx, depth / 2 + 0.05, height + 0.06), f.p(vx, depth / 2 + 0.2, height + 0.06), 0.009, sides=8, caps=True, lm=LM_TINY)
    k.B("steel").rod(f.p(vx - 0.08, depth / 2 + 0.18, height + 0.06), f.p(vx + 0.08, depth / 2 + 0.18, height + 0.06), 0.006, sides=6, caps=True, lm=LM_TINY)
    # Clutter: ammo cans, a rag (paper), a toolbox.
    ammo_can(k, f.p(-length / 2 + 0.3, -0.1, height), yaw + 0.1, rng)
    toolbox = k.B("paint_red", lm=LM_DETAIL)
    toolbox.box(f.p(-0.2, 0.18, height + 0.09), (0.5, 0.22, 0.18), f.r(rz=0.08))
    k.B("paint_dark", lm=LM_TINY).box(f.p(-0.2, 0.18, height + 0.19), (0.3, 0.03, 0.02), f.r(rz=0.08))
    k.collide_box(f.p(0, 0, height / 2), (length, depth, height), f.r(), "wood")
    return height


def ammo_can(k, pos, yaw, rng):
    f = Frame(pos, yaw)
    g = k.B("paint_green", lm=LM_DETAIL)
    g.box(f.p(0, 0, 0.09), (0.28, 0.15, 0.18), f.r())
    g.box(f.p(0, 0, 0.187), (0.29, 0.16, 0.014), f.r())
    k.B("steel", lm=LM_TINY).box(f.p(0, 0, 0.2), (0.12, 0.02, 0.012), f.r())


def electrical_panel(k, pos, yaw, rng, size=(0.6, 0.2, 0.9)):
    f = Frame(pos, yaw)
    g = k.B("paint_grey", lm=LM_SMALL)
    g.box(f.p(0, 0, 0), size, f.r())
    g.box(f.p(0, -size[1] / 2 - 0.006, 0), (size[0] - 0.04, 0.012, size[2] - 0.04), f.r())
    k.B("paint_dark", lm=LM_TINY).box(f.p(size[0] / 2 - 0.08, -size[1] / 2 - 0.015, 0), (0.02, 0.015, 0.1), f.r())
    k.B("paint_yellow", lm=LM_TINY).box(f.p(0, -size[1] / 2 - 0.0125, size[2] / 2 - 0.12), (0.12, 0.001, 0.08), f.r())


def fire_extinguisher(k, pos, yaw, rng):
    f = Frame(pos, yaw)
    red = k.B("paint_red", lm=LM_DETAIL, sharp=60)
    M = f.m4()
    red.lathe([(0.0, 0.0), (0.075, 0.0), (0.08, 0.02), (0.08, 0.42), (0.06, 0.48), (0.02, 0.5), (0.0, 0.5)], M, segments=20)
    dk = k.B("paint_dark", lm=LM_TINY, sharp=60)
    dk.rod(f.p(0, 0, 0.5), f.p(0, 0, 0.56), 0.018, sides=8, caps=True)
    dk.box(f.p(0.03, 0, 0.58), (0.12, 0.02, 0.015), f.r())
    pipe_run(k, [f.p(-0.02, 0, 0.54), f.p(-0.09, 0, 0.5), f.p(-0.1, 0, 0.3), f.p(-0.085, 0, 0.12)], 0.009, "plastic_black", sides=6, lm=LM_TINY, elbow=False)
    k.B("paint_red", lm=LM_TINY).box(f.p(0, 0.1, 0.9), (0.2, 0.004, 0.25), f.r())


def floor_line(k, p0, p1, w=0.1):
    p0, p1 = Vector(p0), Vector(p1)
    k.B("floor_paint", lm=LM_BIG, sharp=89).box_between(Vector((p0.x, p0.y, 0.0012)), Vector((p1.x, p1.y, 0.0012)), w, 0.0024, faces=("+z",))


def paper_sheet(k, pos, yaw, rng, size=(0.21, 0.297)):
    f = Frame(pos, yaw)
    b = k.B("paper", lm=LM_TINY, sharp=89)
    w, h = size
    # Slightly curled sheet (3 x 2 grid).
    verts = []
    for j in range(3):
        for i in range(4):
            u, v = i / 3, j / 2
            z = 0.002 + 0.01 * (abs(u - 0.5) * 2) ** 2 * rng.random()
            verts.append(tuple(f.p((u - 0.5) * w, (v - 0.5) * h, z)))
    faces = []
    for j in range(2):
        for i in range(3):
            a = j * 4 + i
            faces.append((a, a + 1, a + 5, a + 4))
    fuv = [[((a % 4) / 3 * w, (a // 4) / 2 * h) for a in fc] for fc in faces]
    b.add_mesh(verts, faces, fuv=fuv)
