"""Surface detail for texture composites: projector decals, seams and stitching,
bullet holes, tape and marker, all evaluated at baked texel positions.

Everything here is a function of object-space position P (N, 3) and the surface
normal N (N, 3), so it can be sampled on the texel grid (see texbake's position
bake) and differentiated for normal maps: a pattern's height h(P) becomes a
normal perturbation through `perturb`, the way a texturing tool turns a height
layer into normals, but without UV seams because the pattern lives in 3D.
"""
from __future__ import annotations

import math

import numpy as np


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def perturb(P, N, height, eps=3e-4):
    """Normals of the surface displaced by height(P) (metres) along N."""
    g = np.zeros_like(P)
    for a in range(3):
        e = np.zeros(3)
        e[a] = eps
        g[:, a] = (height(P + e) - height(P - e)) / (2 * eps)
    g -= N * np.sum(g * N, 1, keepdims=True)
    n = N - g
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


# ----------------------------------------------------------------------------- lines
def ridge(d, width):
    """Soft line profile: 1 on the line (d = 0) falling to 0 at |d| = width."""
    t = np.clip(1.0 - np.abs(d) / width, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def dashes(along, pitch, duty=0.7, soft=0.12):
    """1 on the stitches of a dashed line, 0 in the gaps between them."""
    f = np.mod(along / pitch, 1.0)
    return smoothstep(0.0, soft, f) * (1.0 - smoothstep(duty - soft, duty, f))


def stitch(d, along, pitch=0.0036, width=0.0007):
    """A row of thread stitches along a line: returns (thread cover, height in metres)."""
    t = ridge(d, width) * dashes(along, pitch)
    return t, t * 0.00022


def seam(d, width=0.0022, depth=0.0005):
    """Height of a sewn seam: a groove on the line with the fabric puffed either side."""
    return -depth * ridge(d, width * 0.5) + depth * 0.4 * ridge(np.abs(d) - width * 0.9, width * 0.7)


# ----------------------------------------------------------------------------- decals
class Decal:
    """A projector: centre c on the surface, normal n, u axis t. `local` maps texels to
    decal coordinates, masked to the side of the surface the decal faces."""

    def __init__(self, c, n, t, reach=0.06, depth=0.03, facing=0.25):
        self.c = np.asarray(c, float)
        self.n = unit(n)
        t = np.asarray(t, float)
        self.t = unit(t - self.n * (t @ self.n))
        self.b = np.cross(self.n, self.t)
        self.reach, self.depth, self.facing = reach, depth, facing

    def local(self, P, N):
        q = P - self.c
        u, v, w = q @ self.t, q @ self.b, q @ self.n
        ok = (np.abs(u) < self.reach) & (np.abs(v) < self.reach) & (np.abs(w) < self.depth) & (N @ self.n > self.facing)
        return u, v, ok


def hash01(*ints):
    h = 2166136261
    for i in ints:
        h = ((h ^ (int(i) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return h / 4294967295.0


def angular_noise(theta, seed, octaves=3):
    """Smooth periodic noise of an angle, for ragged hole and tear outlines."""
    out = np.zeros_like(theta)
    amp = 1.0
    for o in range(octaves):
        k = 3 * (2 ** o)
        for j in range(2):
            ph = hash01(seed, o, j) * 2 * math.pi
            out += amp * 0.5 * np.sin(theta * (k + j * 2) + ph)
        amp *= 0.5
    return out


def bullet_hole(u, v, seed, r=None, brittle=True):
    """A 9 mm hole seen in decal coordinates. Returns (hole core, damage ring, cracks,
    bullet wipe, height in metres)."""
    if r is None:
        r = 0.0046 if brittle else 0.0036     # fibreglass shatters wider than cloth tears
    rr = np.hypot(u, v)
    th = np.arctan2(v, u)
    ragged = r * (1 + 0.18 * angular_noise(th, seed))
    core = 1 - smoothstep(ragged * 0.85, ragged * 1.02, rr)
    wipe = ridge(rr - ragged * 1.25, ragged * 0.45) * (1 - core)
    if brittle:
        ring_r = ragged * (1.7 + 0.5 * angular_noise(th, seed + 7))
        ring = (1 - smoothstep(ring_r * 0.92, ring_r, rr)) * (1 - core)
        cracks = np.zeros_like(rr)
        n = 5 + int(hash01(seed, 3) * 4)
        for k in range(n):
            a = (k + 0.3 * hash01(seed, k)) * 2 * math.pi / n
            length = r * (2.5 + 2.5 * hash01(seed, k, 1))
            # a crack wanders a little as it runs out from the hole
            wob = 0.18 * np.sin(rr * 900 + k) * rr
            d = np.abs(np.sin(th - a) * rr + wob)
            along = np.cos(th - a) * rr
            fade = 1 - smoothstep(length * 0.5, length, along)
            cracks = np.maximum(cracks, ridge(d, 0.00045) * (along > r * 0.8) * fade)
        h = -0.004 * core - 0.00035 * ring - 0.0002 * cracks
    else:
        ring = ridge(rr - ragged * 1.05, ragged * 0.35) * (1 - core)     # frayed lip
        cracks = np.zeros_like(rr)
        h = -0.0025 * core + 0.0003 * ring
    return core, ring, cracks, wipe, h


def tape_strip(u, v, length, width, seed):
    """Duct tape: mask, edge (for dirt and the lifted-edge shadow) and the scrim weave."""
    tear = 0.0025 * (angular_noise(v * 900.0, seed) + 0.6 * np.sin(v * 3400 + seed))
    du = length * 0.5 + tear - np.abs(u)
    dv = width * 0.5 - np.abs(v)
    inside = np.minimum(du, dv)
    mask = smoothstep(-0.0003, 0.0003, inside)
    edge = ridge(inside - 0.0006, 0.0012)
    weave = 0.5 + 0.25 * (np.sin(u * 2 * math.pi / 0.0011) + np.sin(v * 2 * math.pi / 0.0013))
    return mask, edge, weave


def marker_strokes(u, v, segments, width=0.0011, seed=0):
    """Felt-tip strokes given as 2D segments ((u0, v0), (u1, v1)) in decal coordinates."""
    out = np.zeros_like(u)
    for i, ((u0, v0), (u1, v1)) in enumerate(segments):
        a = np.array((u0, v0))
        e = np.array((u1, v1)) - a
        L2 = float(e @ e)
        t = np.clip(((u - u0) * e[0] + (v - v0) * e[1]) / L2, 0, 1)
        d = np.hypot(u - (u0 + t * e[0]), v - (v0 + t * e[1]))
        w = width * (0.8 + 0.3 * np.sin(t * 7 + i + seed))       # pressure varies along a stroke
        out = np.maximum(out, 1 - smoothstep(w * 0.6, w, d))
    return out


def marker_circle(u, v, radius, width=0.0012, seed=0, gap=0.5):
    """A hand-drawn ring: wobbly radius, overshooting ends."""
    th = np.arctan2(v, u)
    r = radius * (1 + 0.06 * angular_noise(th, seed, 2))
    d = np.abs(np.hypot(u, v) - r)
    start = hash01(seed, 9) * 2 * math.pi
    lapse = np.mod(th - start, 2 * math.pi)
    gap_m = smoothstep(0.0, gap, lapse) * smoothstep(0.0, gap * 0.5, 2 * math.pi - lapse + gap * 0.3)
    return (1 - smoothstep(width * 0.6, width, d)) * np.maximum(gap_m, 0.6)


def text_image(text, size_px=96, font_paths=(), pad=8, stretch=1.0):
    """Render text to a float mask (rows top-down), for stencils and name tapes."""
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for p in font_paths:
        try:
            font = ImageFont.truetype(p, size_px)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default(size=size_px)
    box = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), text, fill=255, font=font)
    if stretch != 1.0:
        img = img.resize((max(1, int(w * stretch)), h), Image.LANCZOS)
    return np.asarray(img, np.float32) / 255.0


def sample_image(img, x, y):
    """Bilinear lookup of a (h, w) mask at normalised x (left..right) and y (top..bottom) in
    [0, 1]; outside reads 0."""
    h, w = img.shape
    fx, fy = x * (w - 1), y * (h - 1)
    ok = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, h - 2)
    tx, ty = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
    top = img[y0, x0] * (1 - tx) + img[y0, x0 + 1] * tx
    bot = img[y0 + 1, x0] * (1 - tx) + img[y0 + 1, x0 + 1] * tx
    return np.where(ok, top * (1 - ty) + bot * ty, 0.0)
