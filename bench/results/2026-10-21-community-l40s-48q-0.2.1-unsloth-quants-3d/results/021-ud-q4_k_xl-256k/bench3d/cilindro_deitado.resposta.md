```python
import bpy
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

for v in bm.verts:
    x, y, z = v.co
    v.co = (z, y, -x)

bm.to_mesh(mesh)
bm.free()

obj = bpy.data.objects.new("Cilindro", mesh)
bpy.context.scene.collection.objects.link(obj)
```