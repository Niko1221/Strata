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
    obj.location = location
    return obj


TOP_X = 2.0
TOP_Y = 1.0
TOP_Z = 0.05
TOP_TOP_Z = 0.75

LEG_X = 0.06
LEG_Y = 0.06

top_center_z = TOP_TOP_Z - TOP_Z / 2.0
leg_height = TOP_TOP_Z - TOP_Z
leg_center_z = leg_height / 2.0

add_box(
    "Mesa_Tampo",
    (TOP_X, TOP_Y, TOP_Z),
    (0.0, 0.0, top_center_z),
)

leg_offset_x = TOP_X / 2.0 - LEG_X / 2.0
leg_offset_y = TOP_Y / 2.0 - LEG_Y / 2.0

leg_positions = [
    ( leg_offset_x,  leg_offset_y),
    (-leg_offset_x,  leg_offset_y),
    ( leg_offset_x, -leg_offset_y),
    (-leg_offset_x, -leg_offset_y),
]

for i, (x, y) in enumerate(leg_positions, start=1):
    add_box(
        f"Mesa_Perna_{i}",
        (LEG_X, LEG_Y, leg_height),
        (x, y, leg_center_z),
    )
```