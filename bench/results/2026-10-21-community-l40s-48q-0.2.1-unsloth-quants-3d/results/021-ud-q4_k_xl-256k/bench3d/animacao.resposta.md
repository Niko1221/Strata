```python
import bpy

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 50

obj = bpy.data.objects.get("Movel")
if obj is None:
    mesh = bpy.data.meshes.new("Movel")
    verts = [
        (-1.0, -1.0, -1.0),
        ( 1.0, -1.0, -1.0),
        ( 1.0,  1.0, -1.0),
        (-1.0,  1.0, -1.0),
        (-1.0, -1.0,  1.0),
        ( 1.0, -1.0,  1.0),
        ( 1.0,  1.0,  1.0),
        (-1.0,  1.0,  1.0),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
        (3, 7, 6, 2),
        (0, 1, 5, 4),
    ]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("Movel", mesh)
    scene.collection.objects.link(obj)

scene.frame_set(1)
obj.location = (0.0, 0.0, 0.0)
obj.keyframe_insert("location")

scene.frame_set(50)
obj.location = (5.0, 0.0, 0.0)
obj.keyframe_insert("location")
```