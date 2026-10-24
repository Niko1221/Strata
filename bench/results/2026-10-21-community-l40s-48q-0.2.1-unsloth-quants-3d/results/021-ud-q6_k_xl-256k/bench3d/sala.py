import bpy


def add_box(name, dimensions, center):
    dx, dy, dz = dimensions
    cx, cy, cz = center

    hx = dx / 2.0
    hy = dy / 2.0
    hz = dz / 2.0

    vertices = [
        (cx - hx, cy - hy, cz - hz),  # 0
        (cx + hx, cy - hy, cz - hz),  # 1
        (cx + hx, cy + hy, cz - hz),  # 2
        (cx - hx, cy + hy, cz - hz),  # 3
        (cx - hx, cy - hy, cz + hz),  # 4
        (cx + hx, cy - hy, cz + hz),  # 5
        (cx + hx, cy + hy, cz + hz),  # 6
        (cx - hx, cy + hy, cz + hz),  # 7
    ]

    faces = [
        (0, 3, 2, 1),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),  # -Y
        (2, 3, 7, 6),  # +Y
        (1, 2, 6, 5),  # +X
        (3, 0, 4, 7),  # -X
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    return obj


# Piso: 4 m em X, 5 m em Y, 0.1 m de espessura, face superior em z = 0
add_box("Piso", (4.0, 5.0, 0.1), (0.0, 0.0, -0.05))

# Paredes: 0.1 m de espessura, 2.5 m de altura, de z = 0 a z = 2.5
# Faces externas alinhadas às bordas externas do piso.
add_box("Parede_N", (4.0, 0.1, 2.5), (0.0, 2.45, 1.25))
add_box("Parede_S", (4.0, 0.1, 2.5), (0.0, -2.45, 1.25))
add_box("Parede_L", (0.1, 4.8, 2.5), (1.95, 0.0, 1.25))
add_box("Parede_O", (0.1, 4.8, 2.5), (-1.95, 0.0, 1.25))
