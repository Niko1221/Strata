import bpy
import bmesh
import math

for i in range(8):
    angle = (2.0 * math.pi * i) / 8.0
    x = 3.0 * math.cos(angle)
    y = 3.0 * math.sin(angle)

    name = f"Cilindro.{i + 1:03d}"
    mesh = bpy.data.meshes.new(name)

    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=32,
        radius1=0.2,
        radius2=0.2,
        depth=1.0,
    )
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (x, y, 0.5)
