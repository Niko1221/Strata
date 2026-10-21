import bpy
import math

radius = 0.2
height = 1.0
circle_radius = 3.0
count = 8

for i in range(count):
    angle = 2.0 * math.pi * i / count
    x = circle_radius * math.cos(angle)
    y = circle_radius * math.sin(angle)

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32,
        radius=radius,
        depth=height,
        location=(x, y, height / 2.0),
    )

    obj = bpy.context.active_object
    if obj is not None:
        obj.name = f"Cilindro_{i + 1:02d}"
