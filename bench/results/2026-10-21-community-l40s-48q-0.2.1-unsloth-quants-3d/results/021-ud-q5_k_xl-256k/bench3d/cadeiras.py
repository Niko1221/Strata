import bpy

# Garante que apenas os objetos pedidos existam na cena.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

for mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(mesh)


def add_box(name, dimensions, location):
    dx, dy, dz = [d / 2.0 for d in dimensions]

    vertices = [
        (-dx, -dy, -dz),
        ( dx, -dy, -dz),
        ( dx,  dy, -dz),
        (-dx,  dy, -dz),
        (-dx, -dy,  dz),
        ( dx, -dy,  dz),
        ( dx,  dy,  dz),
        (-dx,  dy,  dz),
    ]

    faces = [
        (0, 3, 2, 1),  # base
        (4, 5, 6, 7),  # topo
        (0, 1, 5, 4),  # -Y
        (3, 2, 6, 7),  # +Y
        (0, 4, 7, 3),  # -X
        (1, 2, 6, 5),  # +X
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location

    return obj


# Mesa: 1.6 m (X) × 0.9 m (Y) × 0.75 m (Z), apoiada no chão e centrada em XY na origem.
table_dims = (1.6, 0.9, 0.75)
table_loc = (0.0, 0.0, table_dims[2] / 2.0)
add_box("Mesa", table_dims, table_loc)

# Cadeiras: cubos de 0.45 m de lado, apoiadas no chão, centradas em x = 0,
# uma em cada lado da mesa no eixo Y, com 0.2 m de folga entre as faces.
chair_side = 0.45
gap = 0.2

chair_y = table_dims[1] / 2.0 + gap + chair_side / 2.0
chair_z = chair_side / 2.0

add_box("Cadeira_1", (chair_side, chair_side, chair_side), (0.0,  chair_y, chair_z))
add_box("Cadeira_2", (chair_side, chair_side, chair_side), (0.0, -chair_y, chair_z))
