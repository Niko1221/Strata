import bpy

mesh = bpy.data.meshes.new("Cube")

verts = [
    (-0.5, -0.5, -0.5),
    ( 0.5, -0.5, -0.5),
    ( 0.5,  0.5, -0.5),
    (-0.5,  0.5, -0.5),
    (-0.5, -0.5,  0.5),
    ( 0.5, -0.5,  0.5),
    ( 0.5,  0.5,  0.5),
    (-0.5,  0.5,  0.5),
]

faces = [
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 4, 7, 3),
    (1, 2, 6, 5),
    (0, 1, 5, 4),
    (3, 7, 6, 2),
]

mesh.from_pydata(verts, [], faces)
mesh.update()

obj = bpy.data.objects.new("Cube", mesh)
bpy.context.scene.collection.objects.link(obj)

array_mod = obj.modifiers.new(name="Array", type='ARRAY')
array_mod.count = 5
array_mod.use_relative_offset = True
array_mod.relative_offset_displace = (1.5, 0.0, 0.0)
array_mod.use_constant_offset = False
