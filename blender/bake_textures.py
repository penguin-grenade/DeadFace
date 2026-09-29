"""Bake seamless PBR texture sets from procedural Blender materials.

Outputs <out>/<name>_{albedo,normal,roughness}.jpg for each material below.
The game loads these instead of its CPU-generated fallbacks when they are
listed in public/assets.json (build_all.py does that).

Tiling trick: UV (u, v) is mapped onto a 4D torus
    (r_u cos 2πu, r_u sin 2πu, r_v cos 2πv, r_v sin 2πv)
and fed to Blender's 4D Noise/Voronoi, so every texture wraps seamlessly.
The torus radii r_u, r_v set the feature frequency along each axis, which
also gives anisotropy for wood grain.

Swap in scanned materials (e.g. CC0 sets from ambientCG / Poly Haven) by
dropping JPGs with the same names into public/textures and listing them
in assets.json; no code changes needed.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import common as c  # noqa: E402

TAU = 2 * math.pi


class G:
    """Tiny node-graph builder."""

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

    def link(self, a, b):
        self.nt.links.new(a, b)

    def math(self, op, a, b=None, clamp=False):
        n = self.nt.nodes.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for i, x in enumerate((a, b)):
            if x is None:
                continue
            if isinstance(x, (int, float)):
                n.inputs[i].default_value = x
            else:
                self.link(x, n.inputs[i])
        return n.outputs[0]

    def torus(self, ru=1.0, rv=1.0, u_offset=None):
        u = self.u if u_offset is None else self.math("ADD", self.u, u_offset)
        au = self.math("MULTIPLY", u, TAU)
        av = self.math("MULTIPLY", self.v, TAU)
        comb = self.nt.nodes.new("ShaderNodeCombineXYZ")
        self.link(self.math("MULTIPLY", self.math("COSINE", au), ru), comb.inputs[0])
        self.link(self.math("MULTIPLY", self.math("SINE", au), ru), comb.inputs[1])
        self.link(self.math("MULTIPLY", self.math("COSINE", av), rv), comb.inputs[2])
        w = self.math("MULTIPLY", self.math("SINE", av), rv)
        return comb.outputs[0], w

    def noise(self, scale, detail=6.0, rough=0.55, ru=1.0, rv=1.0, u_offset=None, distortion=0.0):
        vec, w = self.torus(ru, rv, u_offset)
        n = self.nt.nodes.new("ShaderNodeTexNoise")
        n.noise_dimensions = "4D"
        self.link(vec, n.inputs["Vector"])
        self.link(w, n.inputs["W"])
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = distortion
        return n.outputs["Fac"]

    def voronoi(self, scale, feature="F1", ru=1.0, rv=1.0):
        vec, w = self.torus(ru, rv)
        n = self.nt.nodes.new("ShaderNodeTexVoronoi")
        n.voronoi_dimensions = "4D"
        n.feature = feature
        self.link(vec, n.inputs["Vector"])
        self.link(w, n.inputs["W"])
        n.inputs["Scale"].default_value = scale
        return n.outputs["Distance"]

    def ramp(self, fac, stops):
        """stops: [(pos, (r,g,b) or float)]"""
        n = self.nt.nodes.new("ShaderNodeValToRGB")
        self.link(fac, n.inputs["Fac"])
        els = n.color_ramp.elements
        while len(els) > 1:
            els.remove(els[-1])
        for i, (pos, col) in enumerate(stops):
            e = els[0] if i == 0 else els.new(pos)
            e.position = pos
            e.color = (*col, 1) if isinstance(col, tuple) else (col, col, col, 1)
        return n.outputs["Color"]

    def mix(self, a, b, fac, blend="MIX"):
        n = self.nt.nodes.new("ShaderNodeMix")
        n.data_type = "RGBA"
        n.blend_type = blend
        for sock, x in ((n.inputs[0], fac), (n.inputs[6], a), (n.inputs[7], b)):
            if isinstance(x, (int, float)):
                sock.default_value = x
            elif isinstance(x, tuple):
                sock.default_value = (*x, 1)
            else:
                self.link(x, sock)
        return n.outputs[2]

    def bump(self, height, strength=0.5, distance=0.02):
        n = self.nt.nodes.new("ShaderNodeBump")
        n.inputs["Strength"].default_value = strength
        n.inputs["Distance"].default_value = distance
        self.link(height, n.inputs["Height"])
        self.link(n.outputs["Normal"], self.bsdf.inputs["Normal"])

    def finish(self, color, rough, metallic=0.0):
        self.link(color, self.bsdf.inputs["Base Color"])
        if isinstance(rough, (int, float)):
            self.bsdf.inputs["Roughness"].default_value = rough
        else:
            self.link(rough, self.bsdf.inputs["Roughness"])
        self.bsdf.inputs["Metallic"].default_value = metallic


def concrete(g, seed):
    big = g.noise(1.2 + seed * 0.1, 5, 0.6)
    fine = g.noise(30, 8, 0.6)
    pores = g.voronoi(60)
    stains = g.noise(0.8 + seed * 0.05, 3, 0.6, distortion=0.5)
    base = g.ramp(big, [(0.3, (0.3, 0.3, 0.29)), (0.7, (0.46, 0.45, 0.43))])
    col = g.mix(base, g.ramp(fine, [(0.3, 0.7), (0.7, 1.0)]), 1.0, "MULTIPLY")
    col = g.mix(col, g.ramp(stains, [(0.55, 1.0), (0.75, 0.55)]), 1.0, "MULTIPLY")
    col = g.mix(col, g.ramp(pores, [(0.0, 0.25), (0.06, 1.0)]), 1.0, "MULTIPLY")
    rough = g.ramp(stains, [(0.5, 0.92), (0.8, 0.62)])
    height = g.math("ADD", g.math("MULTIPLY", fine, 0.6), g.math("MULTIPLY", pores, 0.8))
    g.bump(height, 0.9, 0.05)
    g.finish(col, rough)


def plaster(g, seed):
    fine = g.noise(12, 6, 0.55)
    peel = g.noise(1.6, 5, 0.6, distortion=0.3)
    dirt = g.noise(2.5, 4, 0.5)
    paint = g.ramp(fine, [(0.3, (0.52, 0.55, 0.49)), (0.7, (0.6, 0.62, 0.56))])
    under = (0.45, 0.43, 0.39)
    peel_mask = g.ramp(peel, [(0.61, 0.0), (0.63, 1.0)])
    col = g.mix(paint, under, peel_mask)
    # v=0 is the bottom of the wall: grime rising from the floor.
    floor_grime = g.math("POWER", g.math("SUBTRACT", 1.0, g.v, clamp=True), 5.0)
    grime = g.math("ADD", g.math("MULTIPLY", floor_grime, 0.45), g.math("MULTIPLY", dirt, 0.25))
    col = g.mix(col, (0.12, 0.1, 0.08), grime)
    rough = g.math("ADD", 0.72, g.math("MULTIPLY", peel_mask, 0.22))
    g.bump(g.math("ADD", g.math("MULTIPLY", fine, 0.2), g.math("MULTIPLY", peel_mask, -0.6)), 0.4)
    g.finish(col, rough)


def wood(g, seed):
    planks = 4
    plank_id = g.math("FLOOR", g.math("MULTIPLY", g.v, planks))
    offset = g.math("MULTIPLY", plank_id, 0.37)
    grain = g.noise(3.0, 6, 0.6, ru=0.35, rv=6.0, u_offset=offset, distortion=1.5)
    tone = g.math("FRACT", g.math("MULTIPLY", plank_id, 0.618))
    local_v = g.math("FRACT", g.math("MULTIPLY", g.v, planks))
    seam = g.math("MULTIPLY", g.math("LESS_THAN", local_v, 0.03), 1.0)
    seam = g.math("MAXIMUM", seam, g.math("GREATER_THAN", local_v, 0.97))
    base = g.ramp(grain, [(0.35, (0.2, 0.13, 0.07)), (0.65, (0.42, 0.29, 0.17))])
    col = g.mix(base, g.ramp(tone, [(0.0, 0.8), (1.0, 1.1)]), 1.0, "MULTIPLY")
    col = g.mix(col, (0.05, 0.04, 0.03), g.math("MULTIPLY", seam, 0.85))
    g.bump(g.math("SUBTRACT", g.math("MULTIPLY", grain, 0.3), seam), 0.5)
    g.finish(col, g.ramp(grain, [(0.3, 0.82), (0.7, 0.66)]))


def painted_metal(g, seed, paint):
    rust = g.noise(2.2 + seed * 0.1, 7, 0.6, distortion=0.4)
    scratches = g.noise(40, 2, 0.5, ru=0.2, rv=4.0)
    mask = g.ramp(rust, [(0.6, 0.0), (0.66, 1.0)])
    col = g.mix(paint, g.ramp(rust, [(0.6, (0.3, 0.12, 0.05)), (0.9, (0.12, 0.05, 0.02))]), mask)
    scratch = g.ramp(scratches, [(0.495, 0.0), (0.5, 1.0), (0.505, 0.0)])
    col = g.mix(col, (0.45, 0.45, 0.45), scratch)
    rough = g.math("ADD", g.math("MULTIPLY", mask, 0.45), 0.45)
    rough = g.math("SUBTRACT", rough, g.math("MULTIPLY", scratch, 0.25))
    g.bump(g.math("MULTIPLY", mask, rust), 0.6)
    g.finish(col, rough)


def cardboard(g, seed):
    fine = g.noise(18, 5, 0.5)
    flutes = g.math("SINE", g.math("MULTIPLY", g.v, TAU * 90))
    col = g.ramp(fine, [(0.3, (0.3, 0.2, 0.1)), (0.7, (0.4, 0.28, 0.15))])
    g.bump(g.math("ADD", g.math("MULTIPLY", flutes, 0.05), g.math("MULTIPLY", fine, 0.2)), 0.3)
    g.finish(col, 0.9)


RECIPES = {
    "concrete_wall": lambda g: concrete(g, 1),
    "concrete_floor": lambda g: concrete(g, 4),
    "plaster": lambda g: plaster(g, 2),
    "wood": lambda g: wood(g, 3),
    "steel": lambda g: painted_metal(g, 4, (0.045, 0.05, 0.055)),
    "rusty": lambda g: painted_metal(g, 9, (0.3, 0.06, 0.04)),
    "cardboard": lambda g: cardboard(g, 5),
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
    for kind, bake_type, colorspace in (("albedo", "DIFFUSE", "sRGB"), ("roughness", "ROUGHNESS", "Non-Color"), ("normal", "NORMAL", "Non-Color")):
        img = bpy.data.images.new(f"{name}_{kind}", size, size)
        img.colorspace_settings.name = colorspace
        tex.image = img
        kwargs = {"type": bake_type, "margin": 0}
        if bake_type == "DIFFUSE":
            kwargs["pass_filter"] = {"COLOR"}
        if bake_type == "NORMAL":
            kwargs["normal_space"] = "TANGENT"
        bpy.ops.object.bake(**kwargs)
        path = os.path.join(out, f"{name}_{kind}.jpg")
        img.filepath_raw = path
        img.file_format = "JPEG"
        scene.render.image_settings.quality = 90
        img.save()
        print("baked", path)


def main():
    def extra(p):
        p.add_argument("--size", type=int, default=1024)
        p.add_argument("--samples", type=int, default=4)
        p.add_argument("--only", nargs="*")

    args = c.parse_args(extra)
    for name, recipe in RECIPES.items():
        if args.only and name not in args.only:
            continue
        bake(name, recipe, args.out, args.size, args.samples)


if __name__ == "__main__":
    main()
