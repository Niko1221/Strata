import bpy

mesh = bpy.data.meshes.new("Box")

verts = [
    (-1.5, -1.0, 0.0),
    ( 1.5, -1.0, 0.0),
    ( 1.5,  1.0, 0.0),
    (-1.5,  1.0, 0.0),
    (-1.5, -1.0, 1.0),
    ( 1.5, -1.0, 1.0),
    ( 1.5,  1.0, 1.0),
    (-1.5,  1.0, 1.0),
]

faces = [
    (0, 3, 2, 1),  # base, normal -Z
    (4, 5, 6, 7),  # topo, normal +Z
    (0, 1, 5, 4),  # frente, normal -Y
    (1, 2, 6, 5),  # direita, normal +X
    (2, 3, 7, 6),  # trás, normal +Y
    (3, 0, 4, 7),  # esquerda, normal -X
]

mesh.from_pydata(verts, [], faces)
mesh.update()

obj = bpy.data.objects.new("Box", mesh)
bpy.context.scene.collection.objects.link(obj)

obj.location = (0.0, 0.0, 0.0)
obj.scale = (1.0, 1.0, 1.0)
