"""Signed distance fields on a voxel grid and a vectorised Surface Nets mesher.

Organic shapes (the training mannequin) are modelled as smooth unions of
ellipsoids and tapered capsules, the way a sculptor blocks out forms, then
polygonised. Only numpy is needed: no marching-cubes library.

    field = Field(bmin, bmax, voxel=0.004)
    field.add(ellipsoid(...), blend=0.03)
    field.cut(ellipsoid(...), blend=0.01)
    verts, quads = field.mesh()

Every primitive is evaluated only inside its bounding box grown by the blend
radius, so a figure built from ~100 parts polygonises in seconds.
"""
from __future__ import annotations

import numpy as np

BIG = 1e3


def _rot(R):
    return np.eye(3) if R is None else np.asarray(R, float)


class Prim:
    """A primitive: signed distance over (N, 3) points and a world AABB."""

    def __init__(self, fn, lo, hi):
        self.fn, self.lo, self.hi = fn, np.asarray(lo, float), np.asarray(hi, float)


def ellipsoid(c, r, R=None):
    """Ellipsoid with semi-axes r, rotated by R (columns = local axes in world space)."""
    c, r, R = np.asarray(c, float), np.asarray(r, float), _rot(R)
    ext = np.abs(R) @ r

    def fn(p):
        q = (p - c) @ R / r
        k0 = np.linalg.norm(q, axis=-1)
        k1 = np.linalg.norm(q / r, axis=-1)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

    return Prim(fn, c - ext, c + ext)


def cone(a, b, ra, rb):
    """Capsule whose radius tapers from ra at a to rb at b (exact, after Inigo Quilez)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba = b - a
    l2 = float(ba @ ba)
    rr = ra - rb
    a2 = l2 - rr * rr
    il2 = 1.0 / l2

    def fn(p):
        pa = p - a
        y = pa @ ba
        z = y - l2
        x2 = np.sum((pa * l2 - np.outer(y, ba)) ** 2, axis=-1)
        y2 = y * y * l2
        z2 = z * z * l2
        k = np.sign(rr) * rr * rr * x2
        d = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - ra
        d = np.where(np.sign(y) * a2 * y2 < k, np.sqrt(x2 + y2) * il2 - ra, d)
        d = np.where(np.sign(z) * a2 * z2 > k, np.sqrt(x2 + z2) * il2 - rb, d)
        return d

    r = max(ra, rb)
    return Prim(fn, np.minimum(a, b) - r, np.maximum(a, b) + r)


def rbox(c, half, radius, R=None):
    """Box with rounded edges."""
    c, half, R = np.asarray(c, float), np.asarray(half, float), _rot(R)
    ext = np.abs(R) @ half

    def fn(p):
        q = np.abs((p - c) @ R) - half + radius
        return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - radius

    return Prim(fn, c - ext, c + ext)


def halfspace(c, n, lo, hi):
    """Solid on the side opposite n (distance = dot(p - c, n)), evaluated within [lo, hi]."""
    c = np.asarray(c, float)
    n = np.asarray(n, float) / np.linalg.norm(n)
    return Prim(lambda p: (p - c) @ n, lo, hi)


def custom(fn, lo, hi):
    return Prim(fn, lo, hi)


def spline(x, xs, ys):
    """Catmull-Rom interpolation through (xs, ys), clamped at the ends (smooth profile tables)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    x = np.clip(x, xs[0], xs[-1])
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    x0, x1 = xs[i], xs[i + 1]
    t = (x - x0) / (x1 - x0)
    y0, y1 = ys[i], ys[i + 1]
    ym = ys[np.maximum(i - 1, 0)]
    yp = ys[np.minimum(i + 2, len(ys) - 1)]
    xm = xs[np.maximum(i - 1, 0)]
    xp = xs[np.minimum(i + 2, len(xs) - 1)]
    m0 = (y1 - ym) / np.maximum(x1 - xm, 1e-9) * (x1 - x0)
    m1 = (yp - y0) / np.maximum(xp - x0, 1e-9) * (x1 - x0)
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * m1


def loft(table, yc, back, n=2.5, round_bottom=0.0):
    """Solid whose horizontal sections are superellipses: `table` rows are (z, half width,
    front y); the section runs from the front y back to yc + back. For faces and similar
    profiles that are easiest to give as measurements. `round_bottom` rounds the edge of
    the bottom cap (a sharp one shows up as a crease in the detail normals)."""
    T = np.asarray(table, float)

    def fn(p):
        a = np.maximum(spline(p[:, 2], T[:, 0], T[:, 1]), 1e-4)
        yf = spline(p[:, 2], T[:, 0], T[:, 2])
        dy = p[:, 1] - yc
        b = np.where(dy < 0, yc - yf, back)
        q = (np.abs(p[:, 0]) / a) ** n + (np.abs(dy) / b) ** n
        d = (q ** (1.0 / n) - 1.0) * np.minimum(a, b)
        bottom = T[0, 0] - p[:, 2]
        d = round_inter(d, bottom, round_bottom) if round_bottom > 0 else np.maximum(d, bottom)
        return np.maximum(d, p[:, 2] - T[-1, 0])

    w = T[:, 1].max()
    return Prim(fn, (-w, T[:, 2].min(), T[0, 0]), (w, yc + back, T[-1, 0]))


def both(a, b, blend=0.0):
    """Intersection of two primitives."""
    return Prim(lambda p: smax(a.fn(p), b.fn(p), blend), np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi))


def poly2d(pts):
    """Signed distance to a closed 2D polygon (negative inside), for (N, 2) queries."""
    V = np.asarray(pts, float)

    def fn(q):
        d = np.sum((q - V[0]) ** 2, -1)
        s = np.ones(len(q))
        j = len(V) - 1
        for i in range(len(V)):
            e = V[j] - V[i]
            w = q - V[i]
            b = w - e * np.clip((w @ e) / (e @ e), 0, 1)[:, None]
            d = np.minimum(d, np.sum(b * b, -1))
            c1 = q[:, 1] >= V[i, 1]
            c2 = q[:, 1] < V[j, 1]
            c3 = e[0] * w[:, 1] > e[1] * w[:, 0]
            flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
            s = np.where(flip, -s, s)
            j = i
        return s * np.sqrt(d)

    return fn


def round_inter(d1, d2, r):
    """Intersection of two distance fields with the shared edge rounded by r."""
    a, b = d1 + r, d2 + r
    return np.sqrt(np.maximum(a, 0) ** 2 + np.maximum(b, 0) ** 2) + np.minimum(np.maximum(a, b), 0) - r


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


class Field:
    def __init__(self, bmin, bmax, voxel):
        self.h = float(voxel)
        self.o = np.asarray(bmin, float)
        self.n = np.ceil((np.asarray(bmax, float) - self.o) / self.h).astype(int) + 1
        self.F = np.full(tuple(self.n), BIG, np.float32)
        self.ops = []

    def _block(self, prim, pad):
        i0 = np.clip(np.floor((prim.lo - pad - self.o) / self.h).astype(int), 0, self.n - 1)
        i1 = np.clip(np.ceil((prim.hi + pad - self.o) / self.h).astype(int) + 1, 1, self.n)
        axes = [self.o[a] + self.h * np.arange(i0[a], i1[a]) for a in range(3)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
        sl = tuple(slice(i0[a], i1[a]) for a in range(3))
        return sl, P.reshape(-1, 3), P.shape[:3]

    def add(self, prim, blend=0.0, detail=False):
        """Smooth union. `detail` parts are too fine for the grid: they are only recorded, for
        `exact` (and so for the normal map)."""
        self.ops.append((True, prim, blend))
        if detail:
            return
        sl, P, shape = self._block(prim, blend + 2 * self.h)
        d = prim.fn(P).reshape(shape)
        self.F[sl] = smin(self.F[sl], d, blend)

    def cut(self, prim, blend=0.0, detail=False):
        self.ops.append((False, prim, blend))
        if detail:
            return
        sl, P, shape = self._block(prim, blend + 2 * self.h)
        d = prim.fn(P).reshape(shape)
        self.F[sl] = smax(self.F[sl], -d, blend)

    def exact(self, P):
        """Every recorded add/cut, detail parts included, evaluated at arbitrary points (N, 3):
        the 'high poly' that the mesh from the grid is the low-poly cage of."""
        P = np.asarray(P, float)
        d = np.full(len(P), BIG)
        for is_add, prim, blend in self.ops:
            pad = blend + 2 * self.h
            m = np.all((P >= prim.lo - pad) & (P <= prim.hi + pad), axis=1)
            if not m.any():
                continue
            v = prim.fn(P[m])
            d[m] = smin(d[m], v, blend) if is_add else smax(d[m], -v, blend)
        return d

    def exact_normal(self, P, eps=4e-4):
        """Gradient of `exact` by central differences, normalised."""
        P = np.asarray(P, float)
        g = np.zeros_like(P)
        for a in range(3):
            e = np.zeros(3)
            e[a] = eps
            g[:, a] = self.exact(P + e) - self.exact(P - e)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)

    def project(self, P, N, reach=0.006, step=0.00025):
        """Where rays from P + reach * N travelling back along -N first enter the exact
        surface: a baker's cage rays, from the mesh (low poly) onto `exact` (high poly).
        Returns the points and which rays found the surface (the rest come back unchanged)."""
        P, N = np.asarray(P, float), np.asarray(N, float)
        out = P.copy()
        found = np.zeros(len(P), bool)
        prev = self.exact(P + N * reach)
        t_prev = reach
        for t in np.arange(reach - step, -reach - step / 2, -step):
            live = ~found
            if not live.any():
                break
            cur = prev.copy()
            cur[live] = self.exact(P[live] + N[live] * t)
            hit = live & (prev > 0) & (cur <= 0)
            if hit.any():
                s = prev[hit] / (prev[hit] - cur[hit])
                out[hit] = P[hit] + N[hit] * (t_prev + (t - t_prev) * s)[:, None]
                found |= hit
            prev, t_prev = cur, t
        return out, found

    def points(self):
        """World positions of every grid sample, shape (nx, ny, nz, 3)."""
        axes = [self.o[a] + self.h * np.arange(self.n[a]) for a in range(3)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), -1)

    def like(self, F=None):
        """A field on the same grid (a copy of F, or empty)."""
        g = Field.__new__(Field)
        g.h, g.o, g.n = self.h, self.o.copy(), self.n.copy()
        g.F = np.full(tuple(self.n), BIG, np.float32) if F is None else np.array(F, np.float32)
        g.ops = []          # derived fields live on the grid only
        return g

    def sample(self, P):
        """Trilinear lookup of F at world points P (N, 3); outside the grid reads BIG."""
        q = (np.asarray(P, float) - self.o) / self.h
        i = np.floor(q).astype(int)
        t = q - i
        out = np.zeros(len(q))
        ok = np.all((i >= 0) & (i < self.n - 1), axis=1)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dz else 1 - t[:, 2])
                    ii = np.clip(i + (dx, dy, dz), 0, self.n - 1)
                    out += w * self.F[ii[:, 0], ii[:, 1], ii[:, 2]]
        return np.where(ok, out, BIG)

    def intersect(self, prim, blend=0.0, everywhere=False):
        """Keep only the part inside prim."""
        if everywhere:
            P = self.points().reshape(-1, 3)
            d = prim.fn(P).reshape(self.F.shape)
            self.F = smax(self.F, d, blend).astype(np.float32)
            return
        sl, P, shape = self._block(prim, blend + 2 * self.h)
        d = prim.fn(P).reshape(shape)
        outside = np.full(self.F.shape, True)
        outside[sl] = False
        self.F[outside] = np.maximum(self.F[outside], BIG)
        self.F[sl] = smax(self.F[sl], d, blend)

    def mesh(self):
        """Surface Nets: one vertex per surface-crossing cell (the mean of its edge
        crossings), one quad per sign-changing grid edge."""
        F = self.F
        inside = F < 0
        nx, ny, nz = F.shape
        acc = np.zeros((nx - 1, ny - 1, nz - 1, 3), np.float64)
        cnt = np.zeros((nx - 1, ny - 1, nz - 1), np.int32)
        for axis in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[axis] = slice(0, -1)
            b[axis] = slice(1, None)
            f0, f1 = F[tuple(a)], F[tuple(b)]
            cross = inside[tuple(a)] != inside[tuple(b)]
            t = np.where(cross, f0 / np.where(cross, f0 - f1, 1.0), 0.0)
            # each grid edge along `axis` borders up to 4 cells
            idx = np.nonzero(cross)
            base = np.stack(idx, -1).astype(np.float64)
            pos = base.copy()
            pos[:, axis] += t[idx]
            others = [o for o in range(3) if o != axis]
            for du in (0, 1):
                for dv in (0, 1):
                    ci = [idx[0].copy(), idx[1].copy(), idx[2].copy()]
                    ci[others[0]] -= du
                    ci[others[1]] -= dv
                    ok = np.ones(len(ci[0]), bool)
                    for o in range(3):
                        ok &= (ci[o] >= 0) & (ci[o] < F.shape[o] - 1)
                    sel = tuple(c[ok] for c in ci)
                    np.add.at(acc, sel, pos[ok])
                    np.add.at(cnt, sel, 1)
        active = cnt > 0
        vid = np.full(cnt.shape, -1, np.int64)
        vid[active] = np.arange(int(active.sum()))
        verts = acc[active] / cnt[active][:, None]
        verts = self.o + verts * self.h

        quads = []
        for axis in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[axis] = slice(0, -1)
            b[axis] = slice(1, None)
            ia, ib = inside[tuple(a)], inside[tuple(b)]
            cross = ia != ib
            others = [o for o in range(3) if o != axis]
            idx = np.nonzero(cross)
            # need the 4 cells around the edge to exist
            ok = np.ones(len(idx[0]), bool)
            for o in others:
                ok &= (idx[o] >= 1) & (idx[o] < F.shape[o] - 1)
            idx = tuple(i[ok] for i in idx)
            flip = ia[idx]          # inside -> outside along +axis
            u, v = others

            def cell(du, dv):
                ci = [idx[0].copy(), idx[1].copy(), idx[2].copy()]
                ci[u] -= du
                ci[v] -= dv
                return vid[tuple(ci)]

            q = np.stack([cell(1, 1), cell(0, 1), cell(0, 0), cell(1, 0)], -1)
            # wind so the normal points from inside to outside ((u, v, axis) is cyclic
            # except for axis y, whose (x, z) pair is anti-cyclic)
            out_pos = ~flip if axis == 1 else flip
            q = np.where(out_pos[:, None], q, q[:, ::-1])
            quads.append(q)
        quads = np.concatenate(quads)
        quads = quads[(quads >= 0).all(1)]
        return verts.astype(np.float32), quads
