"""Render a quick Cycles preview of a .glb (for reviewing pipeline output headless).

    python3 blender/preview.py --glb public/models/pistol.glb --png /tmp/pistol.png [--dist 0.5]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
import common as c  # noqa: E402


def main():
    def extra(p):
        p.add_argument("--glb", required=True)
        p.add_argument("--png", required=True)
        p.add_argument("--yaw", type=float, default=35)
        p.add_argument("--pitch", type=float, default=15)
        p.add_argument("--samples", type=int, default=32)

    args = c.parse_args(extra)
    scene = c.reset_scene()
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(args.glb))
    objs = [o for o in scene.objects if o.type == "MESH"]
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        for corner in o.bound_box:
            w = o.matrix_world @ Vector(corner)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    center = (lo + hi) / 2
    radius = (hi - lo).length / 2

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    yaw, pitch = math.radians(args.yaw), math.radians(args.pitch)
    d = radius * 4.2
    cam.location = center + Vector((math.sin(yaw) * math.cos(pitch) * d, -math.cos(yaw) * math.cos(pitch) * d, math.sin(pitch) * d))
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam

    for loc, energy, size in [((2, -2, 3), 800, 2), ((-3, -1, 1), 250, 3), ((0, 3, 2), 400, 1)]:
        ld = bpy.data.lights.new("l", "AREA")
        ld.energy = energy * radius * radius
        ld.size = size * radius
        lo_ = bpy.data.objects.new("l", ld)
        lo_.location = center + Vector(loc) * radius * 2
        lo_.rotation_euler = (center - lo_.location).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(lo_)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.055, 1)
    scene.world = world

    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.device = "CPU"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 600
    scene.render.filepath = os.path.abspath(args.png)
    bpy.ops.render.render(write_still=True)
    print("wrote", args.png)


if __name__ == "__main__":
    main()
