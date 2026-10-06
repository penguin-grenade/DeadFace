"""Texture-atlas baking for hero assets (weapon, hands, mannequins).

The asset scripts model with plain materials that only say *what* a surface is
(slide steel, grip polymer, ...). This module turns that into a game-ready PBR
atlas the way a texturing tool would:

  1. `atlas_uvs` packs every object into one UV atlas with xatlas.
  2. Cycles bakes the utility maps the compositor needs:
       ids       material id per texel (exact, 1 sample)
       ao_edge   R = ambient occlusion, G = edge (bevel-shader normal vs. surface
                 normal), B = cavity (short-range AO)
       pos/nrm   object-space position and normal, so noise and grime are 3D
                 and never show UV seams
       normal    tangent-space normal including each material's bump detail
  3. The asset script mixes those in numpy into baseColor / ORM / normal maps
     (see `pbr_material` for the glTF-friendly material that uses them).
"""
from __future__ import annotations

import os
import time

import bpy
import numpy as np

UV = "UVMap"


# ----------------------------------------------------------------------------- UVs
def triangulate(obj):
    """Triangulate while keeping the custom (bevel-hardened) normals."""
    mod = obj.modifiers.new("Tri", "TRIANGULATE")
    mod.quad_method = "BEAUTY"
    mod.ngon_method = "BEAUTY"
    mod.keep_custom_normals = True
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    bpy.data.meshes.remove(old)


def atlas_uvs(objs, resolution=2048, padding=6, density=None, name=UV):
    """One xatlas atlas across all objects. `density` maps object name -> texel density
    multiplier (parts seen up close get more texels). Returns xatlas stats."""
    import xatlas

    atlas = xatlas.Atlas()
    for o in objs:
        triangulate(o)
        me = o.data
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        d = (density or {}).get(o.name, 1.0)
        if d != 1.0:
            c = co.mean(0)
            co = c + (co - c) * d
        lv = np.empty(len(me.loops), np.int64)
        me.loops.foreach_get("vertex_index", lv)
        atlas.add_mesh(co, lv.reshape(-1, 3).astype(np.uint32))
    copt = xatlas.ChartOptions()
    copt.max_iterations = 4
    copt.normal_deviation_weight = 2.0
    copt.max_cost = 2.0
    popt = xatlas.PackOptions()
    popt.resolution = resolution
    popt.padding = padding
    popt.bilinear = True
    popt.rotate_charts = True
    popt.bruteForce = False
    popt.blockAlign = False
    atlas.generate(copt, popt, verbose=False)
    for i, o in enumerate(objs):
        _, idx, uvs = atlas[i]
        me = o.data
        for layer in list(me.uv_layers):
            me.uv_layers.remove(layer)
        layer = me.uv_layers.new(name=name)
        layer.data.foreach_set("uv", uvs[idx.reshape(-1)].astype(np.float32).ravel())
    util = np.atleast_1d(atlas.utilization)
    return dict(charts=atlas.chart_count, atlases=atlas.atlas_count, utilization=round(float(util[0]), 3))


# ----------------------------------------------------------------------------- baking
def setup(samples=16, threads=0):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.use_denoising = False
    if threads:
        sc.render.threads_mode = "FIXED"
        sc.render.threads = threads
    if sc.world is None:
        sc.world = bpy.data.worlds.new("BakeWorld")
    return sc


def new_image(name, size, float_buffer=True):
    img = bpy.data.images.get(name)
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name, size, size, alpha=True, float_buffer=float_buffer)
    img.colorspace_settings.name = "Non-Color"
    img.generated_color = (0, 0, 0, 0)
    return img


def image_array(img):
    w, h = img.size
    arr = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(arr)
    return arr.reshape(h, w, 4)


def _materials(objs):
    seen = []
    for o in objs:
        for s in o.material_slots:
            if s.material is not None and s.material not in seen:
                seen.append(s.material)
    return seen


class _Swap:
    """Temporarily replace every material with a bake material built by `build(nt, mat)`,
    which must return the shader socket to plug into the output."""

    def __init__(self, objs, build):
        self.objs, self.build = objs, build

    def __enter__(self):
        self.saved = [(o, [s.material for s in o.material_slots]) for o in self.objs]
        cache = {}
        for o in self.objs:
            for s in o.material_slots:
                m = s.material
                if m is None:
                    continue
                if m.name not in cache:
                    t = bpy.data.materials.new(f"_bake_{m.name}")
                    t.use_nodes = True
                    nt = t.node_tree
                    nt.nodes.clear()
                    out = nt.nodes.new("ShaderNodeOutputMaterial")
                    nt.links.new(self.build(nt, m), out.inputs["Surface"])
                    cache[m.name] = t
                s.material = cache[m.name]
        self.temp = list(cache.values())
        return self

    def __exit__(self, *exc):
        for o, mats in self.saved:
            for s, m in zip(o.material_slots, mats):
                s.material = m
        for t in self.temp:
            bpy.data.materials.remove(t)


def _attach(objs, img):
    for m in _materials(objs):
        nt = m.node_tree
        n = nt.nodes.get("BakeTarget") or nt.nodes.new("ShaderNodeTexImage")
        n.name = "BakeTarget"
        n.image = img
        n.interpolation = "Closest"
        nt.nodes.active = n


def _detach(objs):
    for m in _materials(objs):
        n = m.node_tree.nodes.get("BakeTarget")
        if n is not None:
            m.node_tree.nodes.remove(n)


def bake(objs, img, kind, samples=None, margin=8, **kw):
    sc = bpy.context.scene
    if samples is not None:
        sc.cycles.samples = samples
    _attach(objs, img)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    t = time.time()
    bpy.ops.object.bake(type=kind, uv_layer=UV, margin=margin, margin_type="EXTEND", use_clear=True,
                        target="IMAGE_TEXTURES", save_mode="INTERNAL", **kw)
    print(f"  bake {kind:7s} {img.size[0]}px {sc.cycles.samples:3d} spp  {time.time() - t:6.1f}s", flush=True)
    _detach(objs)
    return image_array(img)


def _emission(nt, color_socket=None, color=None):
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    if color_socket is not None:
        nt.links.new(color_socket, em.inputs["Color"])
    elif color is not None:
        em.inputs["Color"].default_value = (*color, 1.0)
    return em.outputs["Emission"]


def bake_ids(objs, size, ids):
    """ids: material name -> small int. Returns (id map int32 [-1 = empty], coverage bool)."""
    img = new_image("bake_ids", size)
    with _Swap(objs, lambda nt, m: _emission(nt, color=(ids.get(m.name, 0) / 64.0, 1.0, 0.0))):
        a = bake(objs, img, "EMIT", samples=1, margin=0)
    cover = a[..., 1] > 0.5
    idmap = np.where(cover, np.rint(a[..., 0] * 64.0).astype(np.int32), -1)
    return idmap, cover


def bake_ao_edge(objs, size, samples=16, ao_dist=0.02, cavity_dist=0.004, bevel_radius=0.0009):
    """R = AO, G = edge amount (0 flat .. 1 sharp edge), B = cavity AO."""

    def build(nt, _m):
        ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
        ao.samples = 8
        ao.inputs["Distance"].default_value = ao_dist
        cav = nt.nodes.new("ShaderNodeAmbientOcclusion")
        cav.samples = 8
        cav.inputs["Distance"].default_value = cavity_dist
        bev = nt.nodes.new("ShaderNodeBevel")
        bev.samples = 8
        bev.inputs["Radius"].default_value = bevel_radius
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        dot = nt.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        nt.links.new(bev.outputs["Normal"], dot.inputs[0])
        nt.links.new(geo.outputs["Normal"], dot.inputs[1])
        inv = nt.nodes.new("ShaderNodeMath")
        inv.operation = "SUBTRACT"
        inv.inputs[0].default_value = 1.0
        nt.links.new(dot.outputs["Value"], inv.inputs[1])
        comb = nt.nodes.new("ShaderNodeCombineColor")
        nt.links.new(ao.outputs["AO"], comb.inputs[0])
        nt.links.new(inv.outputs["Value"], comb.inputs[1])
        nt.links.new(cav.outputs["AO"], comb.inputs[2])
        return _emission(nt, comb.outputs["Color"])

    img = new_image("bake_ao", size)
    with _Swap(objs, build):
        a = bake(objs, img, "EMIT", samples=samples)
    return a[..., :3]


def bake_position_normal(objs, size, bmin, bmax):
    """Object-space position (normalised to the bounds) and geometric normal."""
    bmin, bmax = np.asarray(bmin, np.float32), np.asarray(bmax, np.float32)
    ext = bmax - bmin

    def pos(nt, _m):
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sub = nt.nodes.new("ShaderNodeVectorMath")
        sub.operation = "SUBTRACT"
        sub.inputs[1].default_value = tuple(bmin)
        div = nt.nodes.new("ShaderNodeVectorMath")
        div.operation = "DIVIDE"
        div.inputs[1].default_value = tuple(ext)
        nt.links.new(tc.outputs["Object"], sub.inputs[0])
        nt.links.new(sub.outputs["Vector"], div.inputs[0])
        return _emission(nt, div.outputs["Vector"])

    def nrm(nt, _m):
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        mad = nt.nodes.new("ShaderNodeVectorMath")
        mad.operation = "MULTIPLY_ADD"
        mad.inputs[1].default_value = (0.5, 0.5, 0.5)
        mad.inputs[2].default_value = (0.5, 0.5, 0.5)
        nt.links.new(geo.outputs["Normal"], mad.inputs[0])
        return _emission(nt, mad.outputs["Vector"])

    img = new_image("bake_pos", size)
    with _Swap(objs, pos):
        p = bake(objs, img, "EMIT", samples=1)[..., :3] * ext + bmin
    img = new_image("bake_nrm", size)
    with _Swap(objs, nrm):
        n = bake(objs, img, "EMIT", samples=1)[..., :3] * 2 - 1
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    return p, n


def bake_attribute(objs, size, name):
    """A colour attribute (e.g. region fields painted on the mesh) as a float map."""

    def build(nt, _m):
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_name = name
        return _emission(nt, at.outputs["Color"])

    img = new_image(f"bake_attr_{name}", size)
    with _Swap(objs, build):
        return bake(objs, img, "EMIT", samples=1)[..., :3]


def bake_tangent_normal(objs, size, samples=4):
    """Tangent-space normal map (OpenGL / glTF convention) with the materials' bump detail."""
    img = new_image("bake_tnrm", size)
    return bake(objs, img, "NORMAL", samples=samples, normal_space="TANGENT")[..., :3]


# ----------------------------------------------------------------------------- numpy helpers
def dilate(arr, mask, iterations=8):
    """Grow covered texels outwards so bilinear filtering and mips never pull in the background."""
    out = arr.copy()
    m = mask.astype(np.float32)
    squeeze = out.ndim == 2
    if squeeze:
        out = out[..., None]
    for _ in range(iterations):
        acc = np.zeros_like(out)
        cnt = np.zeros(m.shape, np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cnt += np.roll(np.roll(m, dy, 0), dx, 1)
            acc += np.roll(np.roll(out * m[..., None], dy, 0), dx, 1)
        fill = (m == 0) & (cnt > 0)
        out[fill] = acc[fill] / cnt[fill][:, None]
        m = np.where(fill, 1.0, m)
    return out[..., 0] if squeeze else out


def upsample(a, size):
    """Bilinear resize of a (h, w, c) or (h, w) array to size x size."""
    h, w = a.shape[:2]
    if h == size:
        return a
    ys = (np.arange(size) + 0.5) * h / size - 0.5
    xs = (np.arange(size) + 0.5) * w / size - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = np.clip(ys - y0, 0, 1)[:, None]
    fx = np.clip(xs - x0, 0, 1)[None, :]
    if a.ndim == 3:
        fy, fx = fy[..., None], fx[..., None]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def blur(a, radius=1):
    """Cheap box blur (separable), used to soften baked masks."""
    out = a.astype(np.float32)
    for axis in (0, 1):
        acc = np.zeros_like(out)
        for d in range(-radius, radius + 1):
            acc += np.roll(out, d, axis)
        out = acc / (2 * radius + 1)
    return out


def _hash3(i, j, k, seed):
    h = (i * 73856093) ^ (j * 19349663) ^ (k * 83492791) ^ (seed * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def value_noise(p, freq, seed=0):
    """Trilinear value noise at points p (..., 3) in [0, 1]."""
    q = p * freq
    i = np.floor(q).astype(np.int64)
    f = q - i
    f = f * f * (3 - 2 * f)
    out = 0.0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1]) * (f[..., 2] if dz else 1 - f[..., 2])
                out = out + w * _hash3(i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz, seed)
    return out


def fbm(p, freq, octaves=4, seed=0, gain=0.5):
    out = np.zeros(p.shape[:-1], np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        out += amp * value_noise(p, freq * (2 ** o), seed + o * 17)
        norm += amp
        amp *= gain
    return out / norm


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save(arr01, path, quality=92):
    """Save an (h, w, 3) array in [0, 1]; Blender images are bottom-up, files are top-down."""
    from PIL import Image
    img = Image.fromarray((np.clip(arr01, 0, 1) * 255 + 0.5).astype(np.uint8)[::-1], "RGB")
    if path.endswith(".png"):
        img.save(path, optimize=True)
    else:
        img.save(path, quality=quality, subsampling=0, optimize=True)
    print(f"  wrote {os.path.basename(path)} ({os.path.getsize(path) / 1024:.0f} KB)")


# ----------------------------------------------------------------------------- final material
def _occlusion_group():
    """The node group the glTF exporter reads ambient occlusion from."""
    ng = bpy.data.node_groups.get("glTF Material Output")
    if ng is None:
        ng = bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")
        ng.interface.new_socket("Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
        ng.interface.new_socket("Thickness", in_out="INPUT", socket_type="NodeSocketFloat")
    return ng


def pbr_material(name, base_path, orm_path, normal_path, normal_strength=1.0):
    """Principled material wired the way the glTF exporter expects:
    baseColor (sRGB), ORM (R occlusion, G roughness, B metallic), tangent normal."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    def tex(path, non_color):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(path, check_existing=True)
        if non_color:
            n.image.colorspace_settings.name = "Non-Color"
        return n

    base = tex(base_path, False)
    nt.links.new(base.outputs["Color"], bsdf.inputs["Base Color"])
    orm = tex(orm_path, True)
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(orm.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    grp = nt.nodes.new("ShaderNodeGroup")
    grp.node_tree = _occlusion_group()
    nt.links.new(sep.outputs["Red"], grp.inputs["Occlusion"])
    nrm = tex(normal_path, True)
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.uv_map = UV
    nm.inputs["Strength"].default_value = normal_strength
    nt.links.new(nrm.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat
