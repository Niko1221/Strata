import bpy

scene = bpy.context.scene

mesh = bpy.data.meshes.new("Movel")

vertices = [
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
]

faces = [
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
]

mesh.from_pydata(vertices, [], faces)
mesh.update()

cube = bpy.data.objects.new("Movel", mesh)
scene.collection.objects.link(cube)

scene.frame_start = 1
scene.frame_end = 50

cube.location = (0.0, 0.0, 0.0)
cube.keyframe_insert(data_path="location", frame=1)

cube.location = (5.0, 0.0, 0.0)
cube.keyframe_insert(data_path="location", frame=50)

if cube.animation_data and cube.animation_data.action:
    for fcurve in cube.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"

scene.frame_set(1)
