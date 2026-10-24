import bpy
import bmesh
import math

scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0

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
    bm.faces.new((base_verts[i], base_verts[(i + 1) % segments], tip))

bm.faces.new(base_verts)

bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

bm.to_mesh(mesh)
bm.free()

cone = bpy.data.objects.new("Cone", mesh)
scene.collection.objects.link(cone)
