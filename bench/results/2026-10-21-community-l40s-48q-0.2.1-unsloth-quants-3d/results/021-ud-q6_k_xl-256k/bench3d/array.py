import bpy

# Cria a malha de um cubo de 1 m centrado na origem
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
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]

mesh.from_pydata(verts, [], faces)
mesh.validate()
mesh.update()

# Cria o objeto e o adiciona à cena
obj = bpy.data.objects.new("Cube", mesh)
bpy.context.collection.objects.link(obj)

# Adiciona o modificador Array não aplicado
array_mod = obj.modifiers.new(name="Array", type='ARRAY')
array_mod.count = 5
array_mod.use_relative_offset = True
array_mod.use_constant_offset = False
array_mod.use_object_offset = False
array_mod.relative_offset_displace = (1.5, 0.0, 0.0)
