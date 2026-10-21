```python
import bpy
import bmesh

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 50

mesh = bpy.data.meshes.new("Movel")
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=1.0)
bm.to_mesh(mesh)
bm.free()

obj = bpy.data.objects.new("Movel", mesh)
scene.collection.objects.link(obj)

obj.location = (0.0, 0.0, 0.0)
obj.keyframe_insert(data_path="location", frame=1)

obj.location = (5.0, 0.0, 0.0)
obj.keyframe_insert(data_path="location", frame=50)
```