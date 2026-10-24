import bpy
import bmesh
import math
from mathutils import Matrix

mesh = bpy.data.meshes.new("Cone")
obj = bpy.data.objects.new("Cone", mesh)
bpy.context.scene.collection.objects.link(obj)

bm = bmesh.new()
bmesh.ops.create_cone(
    bm,
    cap_ends=True,
    segments=32,
    radius1=0.5,
    radius2=0.0,
    depth=2.0,
)

# Rotate the cone so its axis points along +X, then move it so the base center is at the origin.
bm.transform(Matrix.Rotation(math.pi / 2.0, 4, 'Y'))
bm.transform(Matrix.Translation((1.0, 0.0, 0.0)))

bm.to_mesh(mesh)
bm.free()
