import bpy
import math

bpy.ops.mesh.primitive_cylinder_add(
    vertices=64,
    radius=0.25,
    depth=4.0,
    location=(0.0, 0.0, 0.0),
    rotation=(0.0, math.pi / 2.0, 0.0)
)

obj = bpy.context.active_object
obj.name = "Cilindro_X"
