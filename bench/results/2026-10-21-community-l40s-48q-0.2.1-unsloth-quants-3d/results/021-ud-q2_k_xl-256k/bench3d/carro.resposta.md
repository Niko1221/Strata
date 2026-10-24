```python
import bpy
import math
import bmesh

collection = bpy.context.scene.collection

# Empty 'Carro' na origem
empty = bpy.data.objects.new("Carro", None)
collection.objects.link(empty)
empty.location = (0.0, 0.0, 0.0)
empty.empty_display_type = "PLAIN_AXES"

bpy.context.view_layer.update()

def create_cylinder_mesh(name, radius, depth, vertices=32):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=vertices,
        radius1=radius,
        radius2=radius,
        depth=depth,
    )
    bm.to_mesh(mesh)
    bm.free()
    return mesh

# Malha comum para as rodas: raio 0.3 m, largura/profundidade 0.2 m
wheel_mesh = create_cylinder_mesh("RodaMesh", 0.3, 0.2, 32)

# Quatro rodas filhas do Empty, com eixo paralelo ao eixo Y
for x in (1.2, -1.2):
    for y in (0.8, -0.8):
        wheel = bpy.data.objects.new(f"Roda_{x}_{y}", wheel_mesh)
        wheel.location = (x, y, 0.3)
        wheel.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
        wheel.parent = empty
        wheel.matrix_parent_inverse = empty.matrix_world.inverted()
        collection.objects.link(wheel)
```