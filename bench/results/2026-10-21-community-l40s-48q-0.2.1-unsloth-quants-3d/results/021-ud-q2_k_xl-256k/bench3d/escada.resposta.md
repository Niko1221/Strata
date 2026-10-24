```python
import bpy

scene = bpy.context.scene
collection = scene.collection

vertices = [
    (-0.5, -0.15, -0.09),
    (0.5, -0.15, -0.09),
    (0.5, 0.15, -0.09),
    (-0.5, 0.15, -0.09),
    (-0.5, -0.15, 0.09),
    (0.5, -0.15, 0.09),
    (0.5, 0.15, 0.09),
    (-0.5, 0.15, 0.09),
]

faces = [
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (3, 7, 6, 2),
    (0, 4, 7, 3),
    (1, 2, 6, 5),
]

for n in range(1, 11):
    mesh = bpy.data.meshes.new(f"Step {n:02d} Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(f"Step {n:02d}", mesh)
    collection.objects.link(obj)

    obj.location = (
        0.0,
        (n - 0.5) * 0.30,
        (n - 0.5) * 0.18,
    )
```