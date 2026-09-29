"""Shared helpers for the Blender asset scripts.

Every script works both inside Blender and with the standalone `bpy` module:

    blender -b -P blender/make_pistol.py -- --out public/models
    python3 blender/make_pistol.py --out public/models      # pip install bpy (Python 3.11)

Conventions (so the game can find things by name):
  * Units are metres. Blender +Y is "forward" and +Z is "up"; the glTF exporter
    converts that to three.js -Z forward / +Y up.
  * Animated parts are separate objects whose origin sits at the weapon origin
    (e.g. "Slide"), so the game can offset them from zero.
  * Sockets are empties: "Muzzle", "Ejection", "LightMount".
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
import bmesh
from mathutils import Vector


def parse_args(extra=None):
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "public", "models"))
    if extra:
        extra(p)
    args, _ = p.parse_known_args(argv)
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    return args


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    return scene


def material(name, color=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0, emission=None, emission_strength=0.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def image_material(name, albedo_path, roughness=0.8, metallic=0.0, normal_path=None):
    """Material using baked textures, so they survive glTF export."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if albedo_path and os.path.exists(albedo_path):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(albedo_path)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if normal_path and os.path.exists(normal_path):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(normal_path)
        tex.image.colorspace_settings.name = "Non-Color"
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(tex.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def box(name, size, location=(0, 0, 0), mat=None, bevel=0.0015, segments=3, rotation=(0, 0, 0)):
    """Axis-aligned box built with bmesh, optional bevel modifier for soft highlights."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = rotation
    if mat:
        obj.data.materials.append(mat)
    if bevel > 0:
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = segments
        mod.limit_method = "ANGLE"
        mod.harden_normals = True
    smooth(obj)
    return obj


def cylinder(name, radius, depth, location=(0, 0, 0), rotation=(0, 0, 0), mat=None, verts=32, bevel=0.0):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=verts, radius1=radius, radius2=radius, depth=depth)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = rotation
    if mat:
        obj.data.materials.append(mat)
    if bevel > 0:
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 2
        mod.limit_method = "ANGLE"
    smooth(obj)
    return obj


def smooth(obj, angle=35):
    """Smooth shading with hard edges above `angle` (Blender 4.1+ API)."""
    obj.data.shade_smooth()
    obj.data.set_sharp_from_angle(angle=math.radians(angle))


def empty(name, location, parent=None):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_size = 0.01
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    if parent:
        obj.parent = parent
    return obj


def boolean_cut(target, cutter):
    mod = target.modifiers.new("Cut", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = cutter
    mod.solver = "EXACT"
    # Move the boolean before bevel so the cut edges get bevelled too.
    target.modifiers.move(len(target.modifiers) - 1, 0)
    cutter.hide_render = True
    cutter.hide_viewport = True
    cutter.display_type = "WIRE"
    return mod


def join(name, objs):
    """Bake modifiers and merge objects into one mesh whose origin is the world origin."""
    dg = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    mats = []
    for o in objs:
        ev = o.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev)
        me.transform(o.matrix_world)
        remap = []
        for m in me.materials:
            if m not in mats:
                mats.append(m)
            remap.append(mats.index(m))
        for p in me.polygons:
            if remap:
                p.material_index = remap[p.material_index]
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for m in mats:
        mesh.materials.append(m)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for o in objs:
        bpy.data.objects.remove(o, do_unlink=True)
    return obj


def export_glb(path, objects=None):
    if objects is not None:
        bpy.ops.object.select_all(action="DESELECT")
        for o in objects:
            o.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=objects is not None,
        export_apply=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_cameras=False,
        export_lights=False,
    )
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")
