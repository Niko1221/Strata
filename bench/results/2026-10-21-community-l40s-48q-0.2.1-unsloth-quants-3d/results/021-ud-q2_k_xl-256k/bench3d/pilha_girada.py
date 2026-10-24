import bpy
import math

# Remove qualquer objeto existente para garantir que apenas os cubos pedidos sejam criados.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

def add_cube(name, z, rot_z_deg):
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
        (0, 3, 2, 1),  # face inferior
        (4, 5, 6, 7),  # face superior
        (0, 1, 5, 4),  # face frontal
        (1, 2, 6, 5),  # face direita
        (2, 3, 7, 6),  # face traseira
        (3, 0, 4, 7),  # face esquerda
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = (0.0, 0.0, z)
    obj.rotation_euler = (0.0, 0.0, math.radians(rot_z_deg))

    bpy.context.scene.collection.objects.link(obj)
    return obj

# Cubo 1: apoiado no chão, sem rotação.
add_cube("Cube_1", 0.5, 0.0)

# Cubo 2: apoiado sobre o primeiro, girado 15° em torno de Z.
add_cube("Cube_2", 1.5, 15.0)

# Cubo 3: apoiado sobre o segundo, girado 30° em torno de Z.
add_cube("Cube_3", 2.5, 30.0)
