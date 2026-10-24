```python
import bpy

def add_box(name, dimensions, location):
    dx, dy, dz = dimensions
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0

    verts = [
        (-hx, -hy, -hz),
        ( hx, -hy, -hz),
        ( hx,  hy, -hz),
        (-hx,  hy, -hz),
        (-hx, -hy,  hz),
        ( hx, -hy,  hz),
        ( hx,  hy,  hz),
        (-hx,  hy,  hz),
    ]

    faces = [
        (3, 2, 1, 0),
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
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)

    return obj

table_size_x = 1.6
table_size_y = 0.9
table_size_z = 0.75

chair_size = 0.45
gap = 0.2

table = add_box(
    "Mesa",
    (table_size_x, table_size_y, table_size_z),
    (0.0, 0.0, table_size_z / 2.0),
)

chair_center_y = table_size_y / 2.0 + gap + chair_size / 2.0

chair_1 = add_box(
    "Cadeira_1",
    (chair_size, chair_size, chair_size),
    (0.0, chair_center_y, chair_size / 2.0),
)

chair_2 = add_box(
    "Cadeira_2",
    (chair_size, chair_size, chair_size),
    (0.0, -chair_center_y, chair_size / 2.0),
)

bpy.context.view_layer.update()
```