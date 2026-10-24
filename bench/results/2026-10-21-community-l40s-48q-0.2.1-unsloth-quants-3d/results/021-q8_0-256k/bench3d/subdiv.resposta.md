```python
import bpy
import bmesh

mesh = bpy.data.meshes.new("UV Sphere")
bm = bmesh.new()
bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
bm.to_mesh(mesh)
bm.free()

obj = bpy.data.objects.new("UV Sphere", mesh)
bpy.context.scene.collection.objects.link(obj)

for poly in mesh.polygons:
    poly.use_smooth = True

mesh.update()

subsurf = obj.modifiers.new(name="Subdivision Surface", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 2
```