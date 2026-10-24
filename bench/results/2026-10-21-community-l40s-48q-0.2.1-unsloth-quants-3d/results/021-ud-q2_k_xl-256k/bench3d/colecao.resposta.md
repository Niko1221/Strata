```python
import bpy
import bmesh

scene = bpy.context.scene

# Remove uma coleção existente chamada 'Moveis', se houver.
existing_collection = bpy.data.collections.get("Moveis")
if existing_collection:
    for obj in list(existing_collection.objects):
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data and data.users == 0 and isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)
    bpy.data.collections.remove(existing_collection)

# Remove objetos existentes com os nomes solicitados, para evitar sufixos.
for name in ("Mesa", "Cadeira", "Armario"):
    obj = bpy.data.objects.get(name)
    if obj:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data and data.users == 0 and isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)

# Cria a coleção e liga à cena.
moveis = bpy.data.collections.new("Moveis")
scene.collection.children.link(moveis)


def add_cube(name, size_x, size_y, size_z, location):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)

    for vert in bm.verts:
        vert.co.x *= size_x
        vert.co.y *= size_y
        vert.co.z *= size_z

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    moveis.objects.link(obj)


# Cubos de tamanhos diferentes, com a base apoiada no plano XY (z = 0).
add_cube("Mesa", 1.2, 0.8, 0.75, (0.0, 0.0, 0.375))
add_cube("Cadeira", 0.45, 0.45, 0.9, (1.5, 0.0, 0.45))
add_cube("Armario", 1.0, 0.5, 1.8, (3.0, 0.0, 0.9))
```