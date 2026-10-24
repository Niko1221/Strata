```python
import bpy

scene = bpy.context.scene

# Garante que a coleção 'Moveis' exista e esteja ligada à cena.
collection = bpy.data.collections.get("Moveis")
if collection is None:
    collection = bpy.data.collections.new("Moveis")

if collection.name not in [child.name for child in scene.collection.children]:
    scene.collection.children.link(collection)

# Remove objetos já presentes na coleção para que ela contenha exatamente os pedidos.
for obj in list(collection.objects):
    collection.objects.unlink(obj)


def create_cube_object(name, size):
    half = size / 2.0

    vertices = [
        (-half, -half, -half),
        ( half, -half, -half),
        ( half,  half, -half),
        (-half,  half, -half),
        (-half, -half,  half),
        ( half, -half,  half),
        ( half,  half,  half),
        (-half,  half,  half),
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
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = (0.0, 0.0, half)
    return obj


# Remove objetos existentes com os nomes pedidos para evitar duplicações.
for name in ("Mesa", "Cadeira", "Armario"):
    existing = bpy.data.objects.get(name)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)

# Cria os três cubos de tamanhos diferentes e os coloca na coleção.
for name, size in (
    ("Mesa", 1.5),
    ("Cadeira", 0.5),
    ("Armario", 2.0),
):
    obj = create_cube_object(name, size)
    collection.objects.link(obj)
```