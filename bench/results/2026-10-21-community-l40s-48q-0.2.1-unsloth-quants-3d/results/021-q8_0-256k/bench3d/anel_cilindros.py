import bpy
import math

# Garante que a cena contenha apenas os objetos criados aqui
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

count = 8
circle_radius = 3.0
cylinder_radius = 0.2
cylinder_height = 1.0

for i in range(count):
    angle = 2.0 * math.pi * i / count
    x = circle_radius * math.cos(angle)
    y = circle_radius * math.sin(angle)
    z = cylinder_height / 2.0  # apoia o cilindro no chão (z = 0)

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32,
        radius=cylinder_radius,
        depth=cylinder_height,
        location=(x, y, z)
    )

    obj = bpy.context.active_object
    obj.name = f"Cylinder_{i + 1:02d}"
