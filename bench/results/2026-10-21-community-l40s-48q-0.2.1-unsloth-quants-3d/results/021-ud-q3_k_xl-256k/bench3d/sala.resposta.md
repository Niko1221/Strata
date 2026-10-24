```python
import bpy


def add_box(name, center, dimensions):
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
        (0, 3, 2, 1),  # -Z
        (4, 5, 6, 7),  # +Z
        (0, 1, 5, 4),  # -Y
        (2, 3, 7, 6),  # +Y
        (1, 2, 6, 5),  # +X
        (3, 0, 4, 7),  # -X
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    bpy.context.scene.collection.objects.link(obj)
    return obj


names = [
    "Piso",
    "Parede_Frente",
    "Parede_Fundo",
    "Parede_Esquerda",
    "Parede_Direita",
]

for name in names:
    obj = bpy.data.objects.get(name)
    if obj is not None:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


# Piso: 4 m (X) x 5 m (Y) x 0.1 m, face superior em z = 0
add_box("Piso", (0.0, 0.0, -0.05), (4.0, 5.0, 0.1))

# Paredes: 0.1 m de espessura, de z = 0 a z = 2.5
# Faces externas alinhadas às bordas externas do piso.
add_box("Parede_Frente",   ( 0.0, -2.45, 1.25), (4.0, 0.1, 2.5))
add_box("Parede_Fundo",    ( 0.0,  2.45, 1.25), (4.0, 0.1, 2.5))
add_box("Parede_Esquerda", (-1.95,  0.0, 1.25), (0.1, 4.8, 2.5))
add_box("Parede_Direita",  ( 1.95,  0.0, 1.25), (0.1, 4.8, 2.5))
```