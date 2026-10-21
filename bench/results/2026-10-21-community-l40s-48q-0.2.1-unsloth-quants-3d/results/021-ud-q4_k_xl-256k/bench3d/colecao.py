import bpy

scene = bpy.context.scene

collection = bpy.data.collections.get("Moveis")
if collection is None:
    collection = bpy.data.collections.new("Moveis")

if collection.name not in [child.name for child in scene.collection.children]:
    scene.collection.children.link(collection)

for obj in list(collection.objects):
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh and mesh.users == 0:
        bpy.data.meshes.remove(mesh)

for name in ("Mesa", "Cadeira", "Armario"):
    obj = bpy.data.objects.get(name)
    if obj is not None:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def create_cube(name, size_x, size_y, size_z):
    mesh = bpy.data.meshes.new(name)

    hx = size_x / 2.0
    hy = size_y / 2.0
    hz = size_z / 2.0

    vertices = [
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
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


create_cube("Mesa", 1.2, 0.8, 0.75)
create_cube("Cadeira", 0.45, 0.45, 0.9)
create_cube("Armario", 0.8, 0.5, 1.8)
