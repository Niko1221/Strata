```python
import bpy
import bmesh

scene = bpy.context.scene
collection_name = "Moveis"

# Cria ou reutiliza a coleção 'Moveis'
collection = bpy.data.collections.get(collection_name)
if collection is None:
    collection = bpy.data.collections.new(collection_name)

# Garante que a coleção está ligada à cena
if collection not in scene.collection.children:
    scene.collection.children.link(collection)

# Remove objetos existentes dentro da coleção para garantir exatamente três
for obj in list(collection.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# Remove objetos com os nomes solicitados, se existirem fora da coleção
requested_names = ("Mesa", "Cadeira", "Armario")
for name in requested_names:
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)


def create_cube_object(name, dimensions, location=(0.0, 0.0, 0.0)):
    dx, dy, dz = dimensions

    x = dx / 2.0
    y = dy / 2.0
    z = dz / 2.0

    vertices = [
        (-x, -y, -z),
        ( x, -y, -z),
        ( x,  y, -z),
        (-x,  y, -z),
        (-x, -y,  z),
        ( x, -y,  z),
        ( x,  y,  z),
        (-x,  y,  z),
    ]

    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Recalcula normais para consistência
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    collection.objects.link(obj)
    return obj


# Cria os três objetos de malha
create_cube_object("Mesa", (1.2, 0.8, 0.75), (0.0, 0.0, 0.375))
create_cube_object("Cadeira", (0.45, 0.45, 0.9), (0.0, 0.0, 0.45))
create_cube_object("Armario", (1.0, 0.5, 1.8), (0.0, 0.0, 0.9))
```