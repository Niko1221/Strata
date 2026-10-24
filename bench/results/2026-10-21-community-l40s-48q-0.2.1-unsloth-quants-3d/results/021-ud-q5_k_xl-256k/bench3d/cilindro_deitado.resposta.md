```python
import bpy
import math
import bmesh

mesh = bpy.data.meshes.new("Cilindro")
bm = bmesh.new()

bmesh.ops.create_cone(
    bm,
    cap_ends=True,
    segments=32,
    radius1=0.25,
    radius2=0.25,
    depth=4.0,
)

bm.to_mesh(mesh)
bm.free()

obj = bpy.data.objects.new("Cilindro", mesh)
bpy.context.collection.objects.link(obj)

obj.location = (0.0, 0.0, 0.0)
obj.rotation_euler = (0.0, math.pi / 2.0, 0.0)
```