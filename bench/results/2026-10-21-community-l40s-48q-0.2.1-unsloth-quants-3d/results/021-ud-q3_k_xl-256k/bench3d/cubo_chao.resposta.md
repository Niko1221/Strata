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

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

faces = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]

mesh.from_pydata(vertices, edges, faces)
mesh.update()

cube = bpy.data.objects.new("Cube", mesh)
bpy.context.collection.objects.link(cube)
```