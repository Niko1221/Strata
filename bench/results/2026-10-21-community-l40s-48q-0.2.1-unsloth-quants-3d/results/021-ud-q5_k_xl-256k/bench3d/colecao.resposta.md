```python
import bpy

scene = bpy.context.scene
collection_name = "Moveis"

collection = bpy.data.collections.get(collection_name)
if collection is None:
    collection = bpy.data.collections.new(collection_name)
    scene.collection.children.link(collection)
else:
    if collection not in scene.collection.children:
        scene.collection.children.link(collection)

    # Garante que a coleção contenha exatamente os três objetos pedidos.
    for obj in list(collection.objects):
        collection.objects.unlink(obj)
        if not obj.users_collection:
            bpy.data.objects.remove(obj, do_unlink=True)

# Evita sufixos automáticos caso já existam objetos com os nomes exatos.
for name in ("Mesa", "Cadeira", "Armario"):
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)


def create_box(name, dimensions):
    x, y, z = dimensions

    verts = [
        (-x / 2.0, -y / 2.0, -z / 2.0),
        ( x / 2.0, -y / 2.0, -z / 2.0),
        ( x / 2.0,  y / 2.0, -z / 2.0),
        (-x / 2.0,  y / 2.0, -z / 2.0),
        (-x / 2.0, -y / 2.0,  z / 2.0),
        ( x / 2.0, -y / 2.0,  z / 2.0),
        ( x / 2.0,  y / 2.0,  z / 2.0),
        (-x / 2.0,  y / 2.0,  z / 2.0),
    ]

    faces = [
        (0, 3, 2, 1),  # base
        (4, 5, 6, 7),  # topo
        (0, 1, 5, 4),  # frente
        (1, 2, 6, 5),  # direita
        (2, 3, 7, 6),  # trás
        (3, 0, 4, 7),  # esquerda
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    # Apoia o objeto no plano XY (z = 0).
    obj.location = (0.0, 0.0, z / 2.0)

    return obj


create_box("Mesa", (1.2, 0.8, 0.75))
create_box("Cadeira", (0.45, 0.45, 0.9))
create_box("Armario", (1.0, 0.5, 2.0))
```