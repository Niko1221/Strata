import bpy

# Criar o cubo de 2 m centrado na origem
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
cube = bpy.context.active_object

# Criar o cilindro cortador (raio 0.5 m, atravessando o cubo no eixo Z)
bpy.ops.mesh.primitive_cylinder_add(
    radius=0.5,
    depth=4.0,
    location=(0.0, 0.0, 0.0),
    vertices=128
)
cylinder = bpy.context.active_object

# Adicionar modificador booleano de diferença no cubo
modifier = cube.modifiers.new(name="Hole", type='BOOLEAN')
modifier.operation = 'DIFFERENCE'
modifier.object = cylinder

try:
    modifier.solver = 'EXACT'
except Exception:
    pass

# Aplicar o modificador booleano
bpy.ops.object.select_all(action='DESELECT')
cube.select_set(True)
bpy.context.view_layer.objects.active = cube
bpy.context.view_layer.update()

try:
    bpy.ops.object.modifier_apply(modifier=modifier.name)
except RuntimeError:
    # Fallback caso o operador não funcione em algum contexto headless
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_cube = cube.evaluated_get(depsgraph)
    new_mesh = bpy.data.meshes.new_from_object(evaluated_cube)

    old_mesh = cube.data
    cube.modifiers.clear()
    cube.data = new_mesh

    if old_mesh and old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)

# Remover o cilindro cortador
cylinder_mesh = cylinder.data
bpy.data.objects.remove(cylinder, do_unlink=True)

if cylinder_mesh and cylinder_mesh.users == 0:
    bpy.data.meshes.remove(cylinder_mesh)

# Limpar seleção final
bpy.ops.object.select_all(action='DESELECT')
