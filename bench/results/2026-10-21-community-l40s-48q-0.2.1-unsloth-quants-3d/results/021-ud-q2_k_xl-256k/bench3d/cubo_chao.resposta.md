```python
import bpy

mesh = bpy.data.meshes.new("Cube")

vertices = [
    (-1.0, -1.0, 0.0),
    ( 1.0, -1.0, 0.0),
    ( 1.0,  1.0, 0.0),
    (-1.0,  1.0, 0.0),
    (-1.0, -1.0, 2.0),
    ( 1.0, -1.0, 2.0),
    ( 1.0,  1.0, 2.0),
    (-1.0,  1.0, 2.0),
]

faces = [
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
]

mesh.from_pydata(vertices, [], faces)
mesh.update()

cube = bpy.data.objects.new("Cube", mesh)
bpy.context.collection.objects.link(cube)
```