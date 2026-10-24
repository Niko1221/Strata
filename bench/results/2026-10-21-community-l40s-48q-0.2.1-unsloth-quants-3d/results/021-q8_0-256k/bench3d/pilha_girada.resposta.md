```python
import bpy
import math


def create_cube(name, z_center, rotation_z):
    verts = [
        (-0.5, -0.5, -0.5),
        ( 0.5, -0.5, -0.5),
        ( 0.5,  0.5, -0.5),
        (-0.5,  0.5, -0.5),
        (-0.5, -0.5,  0.5),
        ( 0.5, -0.5,  0.5),
        ( 0.5,  0.5,  0.5),
        (-0.5,  0.5,  0.5),
    ]

    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    obj.location = (0.0, 0.0, z_center)
    obj.rotation_euler = (0.0, 0.0, rotation_z)

    return obj


create_cube("Cube", 0.5, 0.0)
create_cube("Cube.001", 1.5, math.radians(15.0))
create_cube("Cube.002", 2.5, math.radians(30.0))
```