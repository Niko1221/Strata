```python
import bpy

for n in range(1, 11):
    mesh = bpy.data.meshes.new(f"Degrau_{n:02d}")

    x = 0.5
    y = 0.15
    z = 0.09

    vertices = [
        (-x, -y, -z),
        ( x, -y, -z),
        ( x,  y, -z),
        (-x,  y, -z),
        (-x, -y,  z),
        ( x, -y,  z),
        ( x,  y,  z),
        (-x,  y,  z),
    ]

    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
    ]

    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(f"Degrau_{n:02d}", mesh)
    obj.location = (
        0.0,
        (n - 0.5) * 0.30,
        (n - 0.5) * 0.18,
    )

    bpy.context.collection.objects.link(obj)
```