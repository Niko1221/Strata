import bpy

for x in (-2.0, 0.0, 2.0):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.5,
        location=(x, 0.0, 0.5)
    )
