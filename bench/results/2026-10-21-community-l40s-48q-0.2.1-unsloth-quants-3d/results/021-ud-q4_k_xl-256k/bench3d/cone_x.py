import bpy
import bmesh
import math

mesh = bpy.data.meshes.new("Cone")
bm = bmesh.new()

bmesh.ops.create_cone(
    bm,
    cap_ends=True,
    segments=32,
    radius1=0.5,
    radius2=0.0,
    depth=2.0,
)

# Move the mesh so the base center is at the object origin.
for vert in bm.verts:
    vert.co.z += 1.0

bm.to_mesh(mesh)
bm.free()

cone = bpy.data.objects.new("Cone", mesh)
cone.location = (0.0, 0.0, 0.0)
cone.rotation_euler = (0.0, math.pi / 2.0, 0.0)

bpy.context.scene.collection.objects.link(cone)
bpy.context.view_layer.objects.active = cone
