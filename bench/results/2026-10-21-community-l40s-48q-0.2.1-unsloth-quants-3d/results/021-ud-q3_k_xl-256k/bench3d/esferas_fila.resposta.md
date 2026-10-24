```python
import bpy
import bmesh

scene = bpy.context.scene

for i, x in enumerate((-2.0, 0.0, 2.0), start=1):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(
        bm,
        u_segments=32,
        v_segments=16,
        radius=0.5,
    )

    mesh = bpy.data.meshes.new(f"Sphere_{i:02d}")
    bm.to_mesh(mesh)
    bm.free()

    for polygon in mesh.polygons:
        polygon.use_smooth = True

    obj = bpy.data.objects.new(f"Sphere_{i:02d}", mesh)
    obj.location = (x, 0.0, 0.5)

    scene.collection.objects.link(obj)
```