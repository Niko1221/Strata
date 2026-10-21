```python
import bpy
import math
import bmesh

scene = bpy.context.scene
collection = scene.collection

empty = bpy.data.objects.new("Carro", None)
collection.objects.link(empty)
empty.location = (0.0, 0.0, 0.0)

bpy.context.view_layer.update()


def add_wheel(name, location):
    mesh = bpy.data.meshes.new(name)

    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=32,
        radius1=0.3,
        radius2=0.3,
        depth=0.2,
    )
    bm.to_mesh(mesh)
    bm.free()

    wheel = bpy.data.objects.new(name, mesh)
    collection.objects.link(wheel)

    wheel.location = location
    wheel.rotation_euler = (math.pi / 2.0, 0.0, 0.0)

    wheel.parent = empty
    wheel.matrix_parent_inverse = empty.matrix_world.inverted()

    return wheel


positions = [
    (1.2, 0.8),
    (1.2, -0.8),
    (-1.2, 0.8),
    (-1.2, -0.8),
]

for i, (x, y) in enumerate(positions, start=1):
    add_wheel(f"Roda_{i}", (x, y, 0.3))
```