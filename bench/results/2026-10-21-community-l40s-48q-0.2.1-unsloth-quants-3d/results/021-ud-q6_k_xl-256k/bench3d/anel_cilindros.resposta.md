```python
import bpy
import math
import bmesh

collection = bpy.context.scene.collection

for i in range(8):
    angle = 2.0 * math.pi * i / 8.0
    x = 3.0 * math.cos(angle)
    y = 3.0 * math.sin(angle)

    mesh = bpy.data.meshes.new(f"Cilindro_{i + 1}")
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=32,
        radius1=0.2,
        radius2=0.2,
        depth=1.0,
    )
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(f"Cilindro_{i + 1}", mesh)
    obj.location = (x, y, 0.5)
    collection.objects.link(obj)
```