import bpy

mesh = bpy.data.meshes.new("Box")

vertices = [
    (-1.5, -1.0, -0.5),
    ( 1.5, -1.0, -0.5),
    ( 1.5,  1.0, -0.5),
    (-1.5,  1.0, -0.5),
    (-1.5, -1.0,  0.5),
    ( 1.5, -1.0,  0.5),
    ( 1.5,  1.0,  0.5),
    (-1.5,  1.0,  0.5),
]

faces = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]

mesh.from_pydata(vertices, [], faces)
mesh.update()

obj = bpy.data.objects.new("Box", mesh)
bpy.context.collection.objects.link(obj)

obj.location = (0.0, 0.0, 0.5)
obj.scale = (1.0, 1.0, 1.0)
