import bpy

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 50

mesh = bpy.data.meshes.new("Movel")

verts = [
    (-0.5, -0.5, -0.5),
    ( 0.5, -0.5, -0.5),
    ( 0.5,  0.5, -0.5),
    (-0.5,  0.5, -0.5),
    (-0.5, -0.5,  0.5),
    ( 0.5, -0.5,  0.5),
    ( 0.5,  0.5,  0.5),
    (-0.5,  0.5,  0.5),
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

obj = bpy.data.objects.new("Movel", mesh)
scene.collection.objects.link(obj)

obj.location = (0.0, 0.0, 0.0)
obj.keyframe_insert(data_path="location", frame=1)

obj.location = (5.0, 0.0, 0.0)
obj.keyframe_insert(data_path="location", frame=50)

scene.frame_current = 1
