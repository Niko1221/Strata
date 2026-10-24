```python
import bpy

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 50

mesh = bpy.data.meshes.new("Movel")

vertices = [
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
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 4, 7, 3),
    (1, 2, 6, 5),
    (0, 1, 5, 4),
    (3, 7, 6, 2),
]

mesh.from_pydata(vertices, [], faces)
mesh.update()

obj = bpy.data.objects.new("Movel", mesh)
scene.collection.objects.link(obj)

obj.location = (0.0, 0.0, 0.0)
obj.keyframe_insert(data_path="location", frame=1)

obj.location = (5.0, 0.0, 0.0)
obj.keyframe_insert(data_path="location", frame=50)

scene.frame_set(1)
```