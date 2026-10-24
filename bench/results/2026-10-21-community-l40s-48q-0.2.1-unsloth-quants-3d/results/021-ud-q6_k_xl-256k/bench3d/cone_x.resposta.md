```python
import bpy
import bmesh
import math

mesh = bpy.data.meshes.new("Cone")
bm = bmesh.new()

segments = 32
radius = 0.5
height = 2.0

base_verts = []
for i in range(segments):
    angle = 2.0 * math.pi * i / segments
    y = radius * math.cos(angle)
    z = radius * math.sin(angle)
    base_verts.append(bm.verts.new((0.0, y, z)))

tip = bm.verts.new((height, 0.0, 0.0))

for i in range(segments):
    j = (i + 1) % segments
    bm.faces.new((base_verts[i], base_verts[j], tip))

bm.faces.new(tuple(reversed(base_verts)))

bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

bm.to_mesh(mesh)
bm.free()

obj = bpy.data.objects.new("Cone", mesh)
bpy.context.scene.collection.objects.link(obj)
```