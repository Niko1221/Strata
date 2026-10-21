import bpy

bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0)

obj = bpy.context.active_object
obj.name = "UV_Sphere"

for polygon in obj.data.polygons:
    polygon.use_smooth = True

subsurf = obj.modifiers.new(name="Subdivision Surface", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 2
