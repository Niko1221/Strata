```python
import bpy
import bmesh

for n in range(1, 11):
    mesh = bpy.data.meshes.new(f"Degrau_{n:02d}")
    bm = bmesh.new()

    verts = [
        bm.verts.new((-0.5, -0.5, -0.5)),
        bm.verts.new(( 0.5, -0.5, -0.5)),
        bm.verts.new(( 0.5,  0.5, -0.5)),
        bm.verts.new((-0.5,  0.5, -0.5)),
        bm.verts.new((-0.5, -0.5,  0.5)),
        bm.verts.new(( 0.5, -0.5,  0.5)),
        bm.verts.new(( 0.5,  0.5,  0.5)),
        bm.verts.new((-0.5,  0.5,  0.5)),
    ]

    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]

    for face in faces:
        bm.faces.new([verts[i] for i in face])

    bmesh.ops.scale(
        bm,
        verts=bm.verts,
        vec=(1.0, 0.30, 0.18),
    )

    bmesh.ops.translate(
        bm,
        verts=bm.verts,
        vec=(0.0, (n - 0.5) * 0.30, (n - 0.5) * 0.18),
    )

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(f"Degrau_{n:02d}", mesh)
    bpy.context.scene.collection.objects.link(obj)
```