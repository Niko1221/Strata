import bpy


def add_box(name, location, dimensions):
    mesh = bpy.data.meshes.new(name)

    dx, dy, dz = dimensions
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0

    vertices = [
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
        (3, 7, 6, 2),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
    ]

    mesh.from_pydata(vertices, [], faces)
    mesh.validate()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location

    return obj


table_dims = (1.6, 0.9, 0.75)
chair_side = 0.45
clearance = 0.2

add_box(
    "Mesa",
    (0.0, 0.0, table_dims[2] / 2.0),
    table_dims,
)

chair_y = table_dims[1] / 2.0 + clearance + chair_side / 2.0
chair_z = chair_side / 2.0

add_box(
    "Cadeira_1",
    (0.0, chair_y, chair_z),
    (chair_side, chair_side, chair_side),
)

add_box(
    "Cadeira_2",
    (0.0, -chair_y, chair_z),
    (chair_side, chair_side, chair_side),
)
