import bpy


def create_box(name, size_x, size_y, size_z, location):
    hx = size_x / 2.0
    hy = size_y / 2.0
    hz = size_z / 2.0

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
        (0, 4, 7, 3),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location

    return obj


# Tampo: 2.0 m (X) x 1.0 m (Y) x 0.05 m (Z)
# Face superior em z = 0.75 m, logo o centro fica em z = 0.725 m.
create_box("Tampo", 2.0, 1.0, 0.05, (0.0, 0.0, 0.725))

# Pernas: seção 0.06 m x 0.06 m, do chão até a face inferior do tampo.
# Face inferior do tampo: z = 0.75 - 0.05 = 0.70 m.
leg_size = 0.06
leg_height = 0.70
leg_z = leg_height / 2.0

# Bordas externas do tampo: x = ±1.0, y = ±0.5.
# As faces externas das pernas ficam alinhadas a essas bordas.
x_edge = 2.0 / 2.0
y_edge = 1.0 / 2.0
inset = leg_size / 2.0

x_pos = x_edge - inset
y_pos = y_edge - inset

corners = [
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
]

for i, (sx, sy) in enumerate(corners, start=1):
    create_box(
        f"Perna_{i}",
        leg_size,
        leg_size,
        leg_height,
        (sx * x_pos, sy * y_pos, leg_z),
    )
