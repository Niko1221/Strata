import bpy
import bmesh

mesh = bpy.data.meshes.new("UVSphere")
bm = bmesh.new()

bmesh.ops.create_uvsphere(
    bm,
    u_segments=32,
    v_segments=16,
    radius=1.0
)

bm.to_mesh(mesh)
bm.free()

for polygon in mesh.polygons:
    polygon.use_smooth = True

obj = bpy.data.objects.new("UVSphere", mesh)
bpy.context.scene.collection.objects.link(obj)

subsurf = obj.modifiers.new(name="Subdivision Surface", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 2
