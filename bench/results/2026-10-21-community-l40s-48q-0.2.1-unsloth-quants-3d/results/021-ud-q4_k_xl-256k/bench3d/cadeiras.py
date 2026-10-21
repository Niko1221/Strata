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
        (0, 3, 2, 1),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),  # -Y
        (3, 7, 6, 2),  # +Y
        (0, 4, 7, 3),  # -X
        (1, 2, 6, 5),  # +X
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)

    return obj


# Mesa: caixa 1.6 m (X) x 0.9 m (Y) x 0.75 m (Z), apoiada no chão e centrada em XY na origem.
table_dims = (1.6, 0.9, 0.75)
table_center = (0.0, 0.0, table_dims[2] / 2.0)

# Cadeiras: caixas de 0.45 m de lado, apoiadas no chão, centradas em x = 0.
chair_side = 0.45
chair_dims = (chair_side, chair_side, chair_side)
chair_center_z = chair_side / 2.0

# Folga de 0.2 m entre a face da mesa e a face da cadeira no eixo Y.
gap = 0.2
chair_center_y = table_dims[1] / 2.0 + gap + chair_side / 2.0

add_box("Mesa", table_dims, table_center)
add_box("Cadeira_1", chair_dims, (0.0,  chair_center_y, chair_center_z))
add_box("Cadeira_2", chair_dims, (0.0, -chair_center_y, chair_center_z))
