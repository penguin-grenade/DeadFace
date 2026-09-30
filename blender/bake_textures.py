"""Bake seamless PBR texture sets from procedural Blender materials.

Outputs <out>/<name>_{albedo,normal,roughness}.jpg for each recipe below.
The game loads these instead of its CPU-generated fallbacks when they are
listed in public/assets.json (build_all.py does that).

Tiling trick: UV (u, v) is mapped onto a 4D torus
    (r_u cos 2πu, r_u sin 2πu, r_v cos 2πv, r_v sin 2πv)
and fed to Blender's 4D Noise/Voronoi, so every texture wraps seamlessly.
The torus radii r_u, r_v set the feature frequency along each axis, which
also gives anisotropy (wood grain, rust streaks, tyre marks).

Patterns that are periodic by construction (block courses, weave, joints)
are built from floor/fract on the raw UV so they also wrap exactly.

Each recipe documents the real-world size one texture tile represents; the
level builder writes matching UVs.

Swap in scanned materials (e.g. CC0 sets from ambientCG / Poly Haven) by
dropping JPGs with the same names into public/textures.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import common as c  # noqa: E402

TAU = 2 * math.pi


class G:
    """Tiny node-graph builder. Methods accept sockets or plain numbers."""

    def __init__(self, mat):
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.out = self.nt.nodes.new("ShaderNodeOutputMaterial")
        self.bsdf = self.nt.nodes.new("ShaderNodeBsdfPrincipled")
        self.link(self.bsdf.outputs["BSDF"], self.out.inputs["Surface"])
        tc = self.nt.nodes.new("ShaderNodeTexCoord")
        sep = self.nt.nodes.new("ShaderNodeSeparateXYZ")
        self.link(tc.outputs["UV"], sep.inputs[0])
        self.u = sep.outputs["X"]
        self.v = sep.outputs["Y"]
        self.color = None

    # -- plumbing ---------------------------------------------------------------
    def link(self, a, b):
        self.nt.links.new(a, b)

    def _set(self, sock, x):
        if isinstance(x, (int, float)):
            sock.default_value = x
        elif isinstance(x, tuple):
            sock.default_value = (*x, 1.0) if len(x) == 3 else x
        else:
            self.link(x, sock)

    def math(self, op, a, b=None, clamp=False):
        n = self.nt.nodes.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for i, x in enumerate((a, b)):
            if x is not None:
                self._set(n.inputs[i], x)
        return n.outputs[0]

    def add(self, a, b): return self.math("ADD", a, b)
    def sub(self, a, b): return self.math("SUBTRACT", a, b)
    def mul(self, a, b): return self.math("MULTIPLY", a, b)
    def mn(self, a, b): return self.math("MINIMUM", a, b)
    def mx(self, a, b): return self.math("MAXIMUM", a, b)
    def floor(self, a): return self.math("FLOOR", a)
    def fract(self, a): return self.math("FRACT", a)
    def absf(self, a): return self.math("ABSOLUTE", a)
    def clamp01(self, a): return self.math("ADD", a, 0.0, clamp=True)

    def mr(self, x, a, b, lo=0.0, hi=1.0, smooth=True):
        """Map range [a, b] -> [lo, hi] (smoothstep by default), clamped."""
        n = self.nt.nodes.new("ShaderNodeMapRange")
        n.interpolation_type = "SMOOTHSTEP" if smooth else "LINEAR"
        n.clamp = True
        self._set(n.inputs["Value"], x)
        self._set(n.inputs["From Min"], a)
        self._set(n.inputs["From Max"], b)
        self._set(n.inputs["To Min"], lo)
        self._set(n.inputs["To Max"], hi)
        return n.outputs["Result"]

    def combine(self, x, y, z=0.0):
        n = self.nt.nodes.new("ShaderNodeCombineXYZ")
        for i, s in enumerate((x, y, z)):
            self._set(n.inputs[i], s)
        return n.outputs[0]

    # -- tileable noise ---------------------------------------------------------
    def torus(self, ru=1.0, rv=1.0, u=None, v=None):
        u = self.u if u is None else u
        v = self.v if v is None else v
        au = self.mul(u, TAU)
        av = self.mul(v, TAU)
        vec = self.combine(self.mul(self.math("COSINE", au), ru), self.mul(self.math("SINE", au), ru), self.mul(self.math("COSINE", av), rv))
        w = self.mul(self.math("SINE", av), rv)
        return vec, w

    def noise(self, scale, detail=6.0, rough=0.55, ru=1.0, rv=1.0, distortion=0.0, u=None, v=None, lac=2.0):
        vec, w = self.torus(ru, rv, u, v)
        n = self.nt.nodes.new("ShaderNodeTexNoise")
        n.noise_dimensions = "4D"
        self.link(vec, n.inputs["Vector"])
        self.link(w, n.inputs["W"])
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = distortion
        n.inputs["Lacunarity"].default_value = lac
        return n.outputs["Fac"]

    def voronoi(self, scale, feature="F1", ru=1.0, rv=1.0, out="Distance", rand=1.0):
        vec, w = self.torus(ru, rv)
        n = self.nt.nodes.new("ShaderNodeTexVoronoi")
        n.voronoi_dimensions = "4D"
        n.feature = feature
        self.link(vec, n.inputs["Vector"])
        self.link(w, n.inputs["W"])
        n.inputs["Scale"].default_value = scale
        n.inputs["Randomness"].default_value = rand
        return n.outputs[out]

    def white(self, x, y, z=0.0):
        n = self.nt.nodes.new("ShaderNodeTexWhiteNoise")
        n.noise_dimensions = "3D"
        self.link(self.combine(x, y, z), n.inputs["Vector"])
        return n.outputs["Value"]

    # -- colour -------------------------------------------------------------------
    def mix(self, a, b, fac, blend="MIX"):
        n = self.nt.nodes.new("ShaderNodeMix")
        n.data_type = "RGBA"
        n.blend_type = blend
        n.clamp_result = True
        self._set(n.inputs[0], fac)
        self._set(n.inputs[6], a)
        self._set(n.inputs[7], b)
        return n.outputs[2]

    def shade(self, col, f):
        """Multiply a colour by a scalar socket/number."""
        return self.mix(col, f, 1.0, "MULTIPLY")

    def ramp(self, fac, stops):
        n = self.nt.nodes.new("ShaderNodeValToRGB")
        self._set(n.inputs["Fac"], fac)
        els = n.color_ramp.elements
        while len(els) > 1:
            els.remove(els[-1])
        for i, (pos, col) in enumerate(stops):
            e = els[0] if i == 0 else els.new(pos)
            e.position = pos
            e.color = (*col, 1) if isinstance(col, tuple) else (col, col, col, 1)
        return n.outputs["Color"]

    # -- outputs ------------------------------------------------------------------
    def bump(self, height, strength=0.5, distance=0.02):
        n = self.nt.nodes.new("ShaderNodeBump")
        n.inputs["Strength"].default_value = strength
        n.inputs["Distance"].default_value = distance
        self._set(n.inputs["Height"], height)
        self.link(n.outputs["Normal"], self.bsdf.inputs["Normal"])

    def finish(self, color, rough, metallic=0.0):
        self.color = color
        self._set(self.bsdf.inputs["Base Color"], color)
        self._set(self.bsdf.inputs["Roughness"], rough)
        self.bsdf.inputs["Metallic"].default_value = metallic


# ------------------------------------------------------------------------------------
# Recipes. Comment = real-world size of one tile.
# ------------------------------------------------------------------------------------

def floor_concrete(g):
    """4 m x 4 m warehouse slab with saw-cut control joints on the tile edges."""
    mottling = g.noise(2.0, 5, 0.55)
    broad = g.noise(0.7, 3, 0.5, distortion=0.4)
    aggregate = g.voronoi(260, "F1")
    fine = g.noise(90, 6, 0.6)
    trowel = g.noise(6, 4, 0.6, distortion=3.0)
    tyre = g.noise(3, 3, 0.5, ru=0.12, rv=3.0)
    cracks = g.voronoi(3.5, "DISTANCE_TO_EDGE")
    crack_mask = g.mr(g.noise(1.2, 3, 0.5), 0.52, 0.6)
    crack = g.mul(g.mr(cracks, 0.004, 0.0, 0.0, 1.0), crack_mask)
    # Joints: 6 mm wide at u=0 and v=0 (tile edges).
    ju = g.mn(g.u, g.sub(1.0, g.u))
    jv = g.mn(g.v, g.sub(1.0, g.v))
    joint = g.mr(g.mn(ju, jv), 0.0016, 0.0008, 0.0, 1.0)

    base = g.ramp(mottling, [(0.3, (0.33, 0.325, 0.31)), (0.7, (0.45, 0.44, 0.42))])
    base = g.shade(base, g.mr(broad, 0.3, 0.75, 0.82, 1.08))
    speck = g.mr(aggregate, 0.0, 0.25, 0.0, 1.0)
    base = g.mix(base, (0.26, 0.25, 0.24), g.mul(g.sub(1.0, speck), 0.35))
    stains = g.mr(g.noise(1.4, 4, 0.6, distortion=0.8), 0.58, 0.72)
    base = g.mix(base, (0.2, 0.19, 0.18), g.mul(stains, 0.55))
    base = g.mix(base, (0.16, 0.155, 0.15), g.mul(g.mr(tyre, 0.62, 0.7), 0.35))
    base = g.mix(base, (0.1, 0.1, 0.1), g.mx(crack, g.mul(joint, 0.85)))

    rough = g.add(g.mr(trowel, 0.3, 0.7, 0.5, 0.72), g.mul(stains, -0.18))
    rough = g.add(rough, g.mul(g.mx(joint, crack), 0.25))
    height = g.add(g.mul(fine, 0.25), g.mul(speck, 0.2))
    height = g.sub(height, g.mul(g.mx(joint, crack), 1.2))
    g.bump(height, 0.8, 0.03)
    g.finish(base, rough)


def cmu_block(g):
    """1.6 m x 1.6 m of painted 400 x 200 mm concrete block, running bond."""
    rows = 8.0
    cols = 4.0
    row = g.floor(g.mul(g.v, rows))
    offset = g.mul(g.math("MODULO", row, 2.0), 0.5)
    colpos = g.add(g.mul(g.u, cols), offset)
    col = g.math("MODULO", g.floor(colpos), cols)
    fx = g.fract(colpos)
    fy = g.fract(g.mul(g.v, rows))
    # Mortar joints (~10 mm) with a concave profile.
    mx_ = g.mn(fx, g.sub(1.0, fx))
    my_ = g.mn(fy, g.sub(1.0, fy))
    joint_x = g.mr(mx_, 0.0, 0.03, 1.0, 0.0)
    joint_y = g.mr(my_, 0.0, 0.06, 1.0, 0.0)
    joint = g.mx(joint_x, joint_y)
    block_id = g.white(col, row, 3.0)
    pores = g.voronoi(140, "F1")
    pore = g.mr(pores, 0.0, 0.18, 1.0, 0.0)
    fine = g.noise(70, 6, 0.65)
    paint_patch = g.noise(2.2, 4, 0.5)

    paint = g.ramp(paint_patch, [(0.35, (0.56, 0.555, 0.52)), (0.7, (0.63, 0.62, 0.58))])
    paint = g.shade(paint, g.mr(block_id, 0.0, 1.0, 0.93, 1.05, smooth=False))
    col_ = g.mix(paint, (0.36, 0.35, 0.33), g.mul(pore, 0.5))
    col_ = g.mix(col_, (0.4, 0.39, 0.36), g.mul(joint, 0.55))
    dirt = g.mr(g.noise(1.6, 5, 0.6, distortion=0.5), 0.55, 0.75)
    col_ = g.mix(col_, (0.3, 0.28, 0.25), g.mul(dirt, 0.35))
    height = g.sub(g.add(g.mul(fine, 0.2), g.mul(pore, -0.35)), g.mul(joint, 0.9))
    g.bump(height, 0.9, 0.035)
    g.finish(col_, g.add(0.82, g.mul(pore, 0.1)))


def corrugated(g):
    """2 m x 2 m of galvanised sheet (the waves are real geometry); u runs along the wall."""
    spangle = g.voronoi(55, "F1", out="Color")
    spangle_v = g.math("POWER", g.mr(g.voronoi(55, "F1", out="Distance"), 0.0, 0.5, 0.0, 1.0), 0.5)
    white_rust = g.mr(g.noise(2.5, 6, 0.6, distortion=0.6), 0.6, 0.72)
    streaks = g.noise(4.0, 5, 0.55, ru=0.35, rv=5.0)
    streak_mask = g.mr(g.noise(1.2, 3, 0.5), 0.42, 0.6)
    rust = g.mul(g.mr(streaks, 0.55, 0.7), streak_mask)
    dirt = g.mr(g.noise(3.0, 4, 0.5), 0.45, 0.8)

    base = g.mix((0.56, 0.57, 0.58), spangle, 0.12)
    base = g.shade(base, g.mr(spangle_v, 0.0, 1.0, 0.9, 1.05))
    base = g.mix(base, (0.7, 0.7, 0.68), g.mul(white_rust, 0.6))
    base = g.mix(base, (0.26, 0.13, 0.06), g.mul(rust, 1.0))
    base = g.mix(base, (0.25, 0.24, 0.22), g.mul(dirt, 0.3))
    rough = g.add(g.mr(spangle_v, 0.0, 1.0, 0.28, 0.42), g.mul(g.mx(white_rust, rust), 0.45))
    g.bump(g.add(g.mul(rust, 0.3), g.mul(white_rust, 0.2)), 0.5, 0.012)
    g.finish(base, rough, metallic=1.0)


def painted_metal(g, paint, gloss=0.38, chip_scale=5.0, rusty=True):
    """1 m x 1 m of enamel-painted steel with chips, scuffs and edge rust."""
    chips = g.noise(chip_scale, 7, 0.65, distortion=0.3)
    chip = g.mr(chips, 0.66, 0.69)
    scuff = g.mr(g.noise(8, 3, 0.5, ru=0.3, rv=3.0), 0.6, 0.75)
    orange_peel = g.noise(120, 3, 0.5)
    tone = g.noise(1.5, 3, 0.5)
    base = g.shade(paint, g.mr(tone, 0.3, 0.7, 0.9, 1.06))
    base = g.mix(base, (0.62, 0.62, 0.6), g.mul(scuff, 0.25))
    metal = g.mix((0.35, 0.34, 0.33), (0.3, 0.14, 0.06), g.mr(g.noise(20, 4, 0.5), 0.4, 0.7) if rusty else 0.0)
    base = g.mix(base, metal, chip)
    rough = g.add(gloss, g.mul(chip, 0.3))
    rough = g.add(rough, g.mul(scuff, 0.12))
    g.bump(g.sub(g.mul(orange_peel, 0.08), g.mul(chip, 0.5)), 0.6, 0.012)
    g.finish(base, rough)


def structural_steel(g):
    """1.5 m of primed structural steel: dark grey primer, mill scale, dust."""
    painted_metal(g, (0.16, 0.165, 0.17), gloss=0.5, chip_scale=3.0)


def wood_pallet(g):
    """1.2 m x 1.2 m of rough-sawn pine; grain along u."""
    grain = g.noise(2.0, 7, 0.62, ru=0.25, rv=9.0, distortion=1.2)
    rings = g.fract(g.mul(g.add(g.mul(grain, 6.0), g.mul(g.v, 18.0)), 1.0))
    saw = g.math("SINE", g.mul(g.u, TAU * 38))
    weather = g.mul(g.mr(g.noise(1.1, 3, 0.5), 0.5, 0.75), 0.45)
    knots = g.mr(g.voronoi(4.0, "F1"), 0.05, 0.0, 0.0, 1.0)
    dirt = g.mr(g.noise(3.5, 5, 0.55), 0.55, 0.8)

    fresh = g.ramp(rings, [(0.0, (0.62, 0.5, 0.34)), (0.7, (0.55, 0.43, 0.28)), (1.0, (0.44, 0.33, 0.2))])
    aged = g.ramp(rings, [(0.0, (0.5, 0.47, 0.42)), (1.0, (0.38, 0.35, 0.31))])
    base = g.mix(fresh, aged, weather)
    base = g.mix(base, (0.18, 0.12, 0.07), g.mul(knots, 0.8))
    base = g.mix(base, (0.2, 0.18, 0.15), g.mul(dirt, 0.4))
    height = g.add(g.mul(rings, 0.15), g.mul(saw, 0.05))
    height = g.sub(height, g.mul(knots, 0.2))
    g.bump(height, 0.8, 0.02)
    g.finish(base, g.add(0.72, g.mul(weather, 0.12)))


def _spots(g, nu, nv, a, b, prob, seed, pointed=False):
    """One optional feature per cell of an nu x nv grid on the raw UV (so it
    tiles exactly): returns (d, present) where d is the normalised distance
    to a jittered centre, 1 on an ellipse of half-axes a x b (in cell units),
    or on a pointed "football" outline when pointed=True."""
    cu, cv = g.floor(g.mul(g.u, nu)), g.floor(g.mul(g.v, nv))
    fu, fv = g.fract(g.mul(g.u, nu)), g.fract(g.mul(g.v, nv))
    present = g.mr(g.white(cu, cv, seed), 1.0 - prob, 1.0 - prob + 0.001, 0.0, 1.0, smooth=False)
    ou = g.add(a, g.mul(g.white(cu, cv, seed + 1.0), 1.0 - 2.0 * a))
    ov = g.add(b, g.mul(g.white(cu, cv, seed + 2.0), 1.0 - 2.0 * b))
    x = g.absf(g.mul(g.sub(fu, ou), 1.0 / a))
    y = g.mul(g.sub(fv, ov), 1.0 / b)
    if pointed:
        d = g.add(x, g.mul(y, y))
    else:
        d = g.math("SQRT", g.add(g.mul(x, x), g.mul(y, y)))
    return d, present


def plywood(g):
    """1.2 m x 1.2 m of sanded softwood plywood face veneer; grain along u.

    Rotary-cut veneer shows growth rings as contour lines of a smooth surface
    sliced at a shallow angle, so the figure is fract() of a low-frequency
    height field: wide nested bands with abrupt latewood edges. Every feature
    stays several texels wide so the texture does not sparkle at a distance.
    """
    # Knots (~20 x 12 mm) and boat-shaped repair patches (~110 x 45 mm).
    kd, kp = _spots(g, 3, 4, 0.012 * 3, 0.01 * 4, 0.45, 11.0)
    knot = g.mul(g.mr(kd, 1.0, 0.75), kp)
    swirl = g.mul(g.mr(kd, 3.5, 0.0), kp)
    pd, pp = _spots(g, 2, 3, 0.05 * 2, 0.02 * 3, 0.35, 23.0, pointed=True)
    patch = g.mul(g.mr(pd, 1.0, 0.96), pp)
    patch_edge = g.mul(g.mr(g.absf(g.sub(pd, 1.0)), 0.06, 0.0), pp)

    warp = g.noise(0.9, 3, 0.45, ru=0.22, rv=1.3, distortion=0.35)
    # Grain wraps around knots; patches are cut from a different part of the log.
    figure = g.add(g.add(g.mul(warp, 13.0), g.mul(g.v, 9.0)), g.mul(swirl, 1.6))
    figure = g.add(figure, g.mul(patch, g.add(0.4, g.mul(g.u, 30.0))))
    rings = g.fract(figure)
    fibre = g.noise(6.0, 2, 0.4, ru=0.04, rv=4.0)
    tone = g.noise(0.6, 2, 0.5, ru=0.3, rv=0.8)
    # Veneer is jointed from strips 600 mm wide.
    jv = g.fract(g.add(g.mul(g.v, 2.0), 0.23))
    joint = g.mr(g.mn(jv, g.sub(1.0, jv)), 0.0035, 0.0015, 0.0, 1.0)
    strip_tint = g.white(g.floor(g.add(g.mul(g.v, 2.0), 0.23)), 5.0, 31.0)
    grime = g.mr(g.noise(1.4, 4, 0.55), 0.55, 0.8)

    base = g.ramp(rings, [
        (0.0, (0.6, 0.45, 0.27)), (0.55, (0.52, 0.37, 0.2)), (0.82, (0.36, 0.21, 0.09)),
        (0.95, (0.3, 0.17, 0.07)), (1.0, (0.44, 0.3, 0.16)),
    ])
    base = g.mix(base, (0.64, 0.5, 0.31), g.mul(tone, 0.3))
    base = g.mix(base, (0.47, 0.33, 0.18), g.mul(strip_tint, 0.25))
    base = g.mix(base, (0.43, 0.3, 0.16), g.mul(fibre, 0.3))
    base = g.mix(base, (0.17, 0.09, 0.04), g.mul(knot, 0.95))
    base = g.mix(base, (0.22, 0.13, 0.06), g.mx(g.mul(patch_edge, 0.8), g.mul(joint, 0.7)))
    base = g.mix(base, (0.3, 0.26, 0.21), g.mul(grime, 0.35))
    late = g.mr(rings, 0.7, 0.95)
    height = g.add(g.mul(late, 0.06), g.mul(fibre, 0.04))
    height = g.sub(height, g.add(g.mul(g.mx(patch_edge, joint), 0.12), g.mul(knot, 0.05)))
    g.bump(height, 0.6, 0.02)
    g.finish(base, g.add(0.72, g.add(g.mul(late, -0.08), g.add(g.mul(grime, 0.08), g.mul(knot, -0.15)))))


def drywall(g):
    """2.4 m x 2.4 m of painted gypsum board: sheet seams, screw heads, scuffs."""
    peel = g.noise(160, 3, 0.5)
    seam = g.mr(g.absf(g.sub(g.fract(g.mul(g.u, 2.0)), 0.5)), 0.5, 0.497, 0.0, 1.0)
    # Screw heads every 300 mm on 400 mm studs.
    sx = g.fract(g.mul(g.u, 6.0))
    sy = g.fract(g.mul(g.v, 8.0))
    screws = g.mr(g.math("SQRT", g.add(g.mul(g.sub(sx, 0.5), g.sub(sx, 0.5)), g.mul(g.sub(sy, 0.5), g.sub(sy, 0.5)))), 0.03, 0.015, 0.0, 1.0)
    scuff = g.mr(g.noise(6, 4, 0.5, ru=0.5, rv=2.0), 0.64, 0.75)
    patch = g.mr(g.noise(1.2, 2, 0.5), 0.62, 0.64)
    base = g.mix((0.74, 0.73, 0.69), (0.7, 0.7, 0.67), patch)
    base = g.mix(base, (0.55, 0.53, 0.5), g.mul(scuff, 0.4))
    base = g.mix(base, (0.66, 0.65, 0.62), g.mul(g.mx(seam, screws), 0.3))
    height = g.add(g.mul(peel, 0.05), g.mul(g.mx(seam, screws), 0.12))
    g.bump(height, 0.5, 0.012)
    g.finish(base, g.add(0.86, g.mul(scuff, -0.1)))


def sandbag(g):
    """0.5 m x 0.5 m of woven polypropylene sandbag fabric."""
    n = 160.0
    warp = g.math("SINE", g.mul(g.u, TAU * n))
    weft = g.math("SINE", g.mul(g.v, TAU * n))
    weave = g.mul(g.add(g.mul(warp, weft), 1.0), 0.5)
    dirt = g.mul(g.mr(g.noise(2.0, 4, 0.55), 0.5, 0.8), 0.6)
    stain = g.mr(g.noise(1.4, 4, 0.5, distortion=0.6), 0.6, 0.72)
    base = g.mix((0.6, 0.54, 0.4), (0.5, 0.45, 0.32), weave)
    base = g.mix(base, (0.3, 0.26, 0.19), g.mul(dirt, 0.5))
    base = g.mix(base, (0.22, 0.2, 0.15), g.mul(stain, 0.5))
    g.bump(g.add(g.mul(weave, 0.3), g.mul(dirt, 0.1)), 0.8, 0.008)
    g.finish(base, 0.88)


def rubber(g):
    """0.5 m x 0.5 m of worn black rubber (tyres, mats)."""
    fine = g.noise(90, 5, 0.6)
    wear = g.mul(g.mr(g.noise(2, 3, 0.5), 0.5, 0.8), 0.5)
    base = g.mix((0.028, 0.028, 0.03), (0.07, 0.068, 0.065), wear)
    g.bump(g.mul(fine, 0.2), 0.5, 0.01)
    g.finish(base, g.add(0.68, g.mul(wear, 0.15)))


def cardboard(g):
    """0.6 m x 0.6 m of corrugated kraft board."""
    fine = g.noise(40, 5, 0.5)
    flutes = g.math("SINE", g.mul(g.v, TAU * 70))
    stain = g.mr(g.noise(2.0, 4, 0.55), 0.6, 0.72)
    base = g.ramp(fine, [(0.3, (0.44, 0.32, 0.19)), (0.7, (0.52, 0.39, 0.24))])
    base = g.mix(base, (0.3, 0.22, 0.13), g.mul(stain, 0.5))
    g.bump(g.add(g.mul(flutes, 0.04), g.mul(fine, 0.1)), 0.5, 0.012)
    g.finish(base, 0.9)


def painted_generic(g):
    """1 m x 1 m of light neutral paint on steel, tinted per object in the game."""
    painted_metal(g, (0.78, 0.78, 0.77), gloss=0.42, chip_scale=4.0)


def paper(g):
    """0.5 m x 0.5 m of printer paper (scattered sheets, notices)."""
    fine = g.noise(80, 3, 0.5)
    lines = g.mr(g.absf(g.sub(g.fract(g.mul(g.v, 30.0)), 0.5)), 0.47, 0.5, 0.0, 1.0)
    base = g.mix((0.86, 0.85, 0.82), (0.35, 0.35, 0.37), g.mul(lines, 0.35))
    g.bump(g.mul(fine, 0.1), 0.3, 0.006)
    g.finish(base, 0.8)


RECIPES = {
    # name: (recipe, size)
    "floor_concrete": (floor_concrete, 2048),
    "cmu_block": (cmu_block, 1024),
    "corrugated": (corrugated, 1024),
    "steel": (structural_steel, 1024),
    "rack_blue": (lambda g: painted_metal(g, (0.035, 0.1, 0.3), gloss=0.34), 1024),
    "rack_orange": (lambda g: painted_metal(g, (0.72, 0.24, 0.025), gloss=0.34), 1024),
    "wood": (wood_pallet, 1024),
    "plywood": (plywood, 1024),
    "drywall": (drywall, 1024),
    "sandbag": (sandbag, 1024),
    "rubber": (rubber, 512),
    "cardboard": (cardboard, 1024),
    "painted": (painted_generic, 1024),
    "paper": (paper, 512),
}


def bake(name, recipe, out, size, samples):
    c.reset_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    bpy.ops.mesh.primitive_plane_add(size=2)
    plane = bpy.context.active_object
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    plane.data.materials.append(mat)
    g = G(mat)
    recipe(g)

    tex = g.nt.nodes.new("ShaderNodeTexImage")
    g.nt.nodes.active = tex
    emit = g.nt.nodes.new("ShaderNodeEmission")
    g.link(g.color, emit.inputs["Color"])
    for kind, bake_type, colorspace in (("albedo", "EMIT", "sRGB"), ("roughness", "ROUGHNESS", "Non-Color"), ("normal", "NORMAL", "Non-Color")):
        # Albedo comes through an emission shader so metals keep their colour.
        src = emit.outputs["Emission"] if kind == "albedo" else g.bsdf.outputs["BSDF"]
        g.link(src, g.out.inputs["Surface"])
        img = bpy.data.images.new(f"{name}_{kind}", size, size)
        img.colorspace_settings.name = colorspace
        tex.image = img
        kwargs = {"type": bake_type, "margin": 0}
        if bake_type == "NORMAL":
            kwargs["normal_space"] = "TANGENT"
        bpy.ops.object.bake(**kwargs)
        path = os.path.join(out, f"{name}_{kind}.jpg")
        img.filepath_raw = path
        img.file_format = "JPEG"
        scene.render.image_settings.quality = 92
        img.save()
        print("baked", path, flush=True)


def main():
    def extra(p):
        p.add_argument("--samples", type=int, default=4)
        p.add_argument("--only", nargs="*")
        p.add_argument("--scale", type=float, default=1.0, help="multiply every texture size (0.5 for quick tests)")

    args = c.parse_args(extra)
    for name, (recipe, size) in RECIPES.items():
        if args.only and name not in args.only:
            continue
        bake(name, recipe, args.out, max(64, int(size * args.scale)), args.samples)


if __name__ == "__main__":
    main()
