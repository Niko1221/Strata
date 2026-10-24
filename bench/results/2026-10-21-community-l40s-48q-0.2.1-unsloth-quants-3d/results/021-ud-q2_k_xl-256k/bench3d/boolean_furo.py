import bpy

# Garanta que a cena comece sem objetos, caso o Blender não esteja totalmente vazio.
for obj in list(bpy.data.objects):
    obj_data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if obj_data and obj_data.users == 0 and isinstance(obj_data, bpy.types.Mesh):
        bpy.data.meshes.remove(obj_data)

# Cubo de 2 m centrado na origem.
bpy.ops.mesh.primitive_cube_add(size=2.0)
cube = bpy.context.active_object
cube.name = "Cube"

# Cilindro cortador: raio 0,5 m, atravessando completamente o cubo no eixo Z.
bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2.2, vertices=64)
cutter = bpy.context.active_object
cutter.name = "Cylinder"

# Operação booleana de diferença.
modifier = cube.modifiers.new("Boolean", 'BOOLEAN')
modifier.operation = 'DIFFERENCE'
modifier.object = cutter
modifier.solver = 'EXACT'

# Aplique o modificador no cubo.
bpy.ops.object.select_all(action='DESELECT')
cube.select_set(True)
bpy.context.view_layer.objects.active = cube
bpy.ops.object.convert(target='MESH')

# Remova o cilindro e sua malha, se ficar sem uso.
cutter_mesh = cutter.data
bpy.data.objects.remove(cutter, do_unlink=True)
if cutter_mesh and cutter_mesh.users == 0:
    bpy.data.meshes.remove(cutter_mesh)

# Garanta que apenas o cubo furado permaneça na cena.
for obj in list(bpy.data.objects):
    if obj is not cube:
        obj_data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if obj_data and obj_data.users == 0 and isinstance(obj_data, bpy.types.Mesh):
            bpy.data.meshes.remove(obj_data)
