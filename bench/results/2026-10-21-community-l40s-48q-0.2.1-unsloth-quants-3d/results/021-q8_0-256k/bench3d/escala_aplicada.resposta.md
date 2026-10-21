```python
import bpy

# Mesh vertices for a 3 m × 2 m × 1 m box resting on the XY plane (z = 0)
vertices = [
    (-1.5, -1.0, 0.0),
    ( 1.5, -1.0, 0.0),
    ( 1.5,  1.0, 0.0),
    (-1.5,  1.0, 0.0),
    (-1.5, -1.0, 1.0),
    ( 1.5, -1.0, 1.0),
    ( 1.5,  1.0, 1.0),
    (-1.5,  1.0, 1.0),
]

faces = [
    (0, 3, 2, 1),  # bottom
    (4, 5, 6, 7),  # top
    (0, 1, 5, 4),  # front (y = -1)
    (3, 7, 6, 2),  # back  (y = +1)
    (1, 2, 6, 5),  # right (x = +1)
    (0, 4, 7, 3),  # left  (x = -1)
]

mesh = bpy.data.meshes.new("Box")
mesh.from_pydata(vertices, [], faces)
mesh.update()

obj = bpy.data.objects.new("Box", mesh)
bpy.context.scene.collection.objects.link(obj)

# Ensure scale is applied / identity
obj.scale = (1.0, 1.0, 1.0)
```