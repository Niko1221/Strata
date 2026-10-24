import bpy

# Garanta que apenas os objetos pedidos existam na cena.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# Configure unidades em metros.
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.length_unit = 'METERS'
scene.unit_settings.scale_length = 1.0


def add_box(name, location, dimensions):
    """
    Cria um cubo/paralelepípedo com origem no centro e dimensões exatas.
    location: centro do objeto em metros.
    dimensions: (dx, dy, dz) em metros.
    """
    mesh = bpy.data.meshes.new(name)

    hx = dimensions[0] / 2.0
    hy = dimensions[1] / 2.0
    hz = dimensions[2] / 2.0

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
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location

    bpy.context.scene.collection.objects.link(obj)
    return obj


# Piso: 4 m em X, 5 m em Y, 0.1 m de espessura.
# Face superior em z = 0, portanto centro em z = -0.05.
add_box(
    name="Floor",
    location=(0.0, 0.0, -0.05),
    dimensions=(4.0, 5.0, 0.1),
)

# Paredes: 0.1 m de espessura, 2.5 m de altura, de z = 0 a z = 2.5.
# Centro em z = 1.25.
#
# As faces externas das paredes estão alinhadas às bordas externas do piso:
# piso em X: -2 a +2
# piso em Y: -2.5 a +2.5
#
# Paredes Norte/Sul ocupam toda a extensão em X.
# Paredes Leste/Oeste ocupam o vão interno em Y para evitar sobreposição nos cantos.

add_box(
    name="Wall_North",
    location=(0.0, 2.45, 1.25),
    dimensions=(4.0, 0.1, 2.5),
)

add_box(
    name="Wall_South",
    location=(0.0, -2.45, 1.25),
    dimensions=(4.0, 0.1, 2.5),
)

add_box(
    name="Wall_East",
    location=(1.95, 0.0, 1.25),
    dimensions=(0.1, 4.8, 2.5),
)

add_box(
    name="Wall_West",
    location=(-1.95, 0.0, 1.25),
    dimensions=(0.1, 4.8, 2.5),
)
