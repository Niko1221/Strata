```python
import bpy

scene = bpy.context.scene
target_names = ("Mesa", "Cadeira", "Armario")

# Remove objetos existentes com os nomes exatos solicitados.
for name in target_names:
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)

# Obtém ou cria a coleção 'Moveis'.
collection = bpy.data.collections.get("Moveis")
if collection is None:
    collection = bpy.data.collections.new("Moveis")

# Garante que a coleção esteja ligada à cena.
linked = False
for child in scene.collection.children:
    if child == collection:
        linked = True
        break

if not linked:
    scene.collection.children.link(collection)

# Limpa objetos diretos da coleção para que ela contenha exatamente os três pedidos.
for obj in list(collection.objects):
    bpy.data.objects.remove(obj, do_unlink=True)


def make_cube_mesh(mesh_name, dimensions):
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
        (1, 5, 4, 0),
        (2, 6, 5, 1),
        (3, 7, 6, 2),
        (0, 4, 7, 3),
    ]

    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


specs = [
    ("Mesa", (1.6, 0.8, 0.75)),
    ("Cadeira", (0.45, 0.45, 0.9)),
    ("Armario", (1.0, 0.5, 2.0)),
]

for name, dims in specs:
    mesh = make_cube_mesh(f"{name}_mesh", dims)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = (0.0, 0.0, dims[2] / 2.0)
    collection.objects.link(obj)
```