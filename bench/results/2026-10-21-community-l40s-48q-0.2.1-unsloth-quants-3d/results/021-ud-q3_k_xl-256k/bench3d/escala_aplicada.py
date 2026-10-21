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
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (3, 7, 6, 2),
    (0, 4, 7, 3),
    (1, 2, 6, 5),
]

mesh.from_pydata(vertices, [], faces)
mesh.update()

obj = bpy.data.objects.new("Box", mesh)
bpy.context.scene.collection.objects.link(obj)

obj.location = (0.0, 0.0, 0.5)
obj.rotation_euler = (0.0, 0.0, 0.0)
obj.scale = (1.0, 1.0, 1.0)

bpy.context.view_layer.update()
