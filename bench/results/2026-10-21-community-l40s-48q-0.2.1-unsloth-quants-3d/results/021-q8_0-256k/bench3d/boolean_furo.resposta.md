```python
import bpy

# Criar o cubo de 2 m centrado na origem
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
cube = bpy.context.active_object

# Criar o cilindro cortador (um pouco mais alto que o cubo para atravessá-lo)
bpy.ops.mesh.primitive_cylinder_add(
    vertices=64,
    radius=0.5,
    depth=2.2,
    location=(0, 0, 0)
)
cylinder = bpy.context.active_object

# Adicionar modificador booleano de diferença no cubo
modifier = cube.modifiers.new(name="Hole", type='BOOLEAN')
modifier.operation = 'DIFFERENCE'
modifier.object = cylinder
modifier.solver = 'EXACT'

# Aplicar o booleano avaliando o objeto
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_cube = cube.evaluated_get(depsgraph)
new_mesh = bpy.data.meshes.new_from_object(evaluated_cube)

old_mesh = cube.data
cube.modifiers.clear()
cube.data = new_mesh

# Remover o cilindro cortador
cylinder_mesh = cylinder.data
bpy.data.objects.remove(cylinder, do_unlink=True)

# Limpar malhas órfãs
if old_mesh and old_mesh.users == 0:
    bpy.data.meshes.remove(old_mesh)
if cylinder_mesh and cylinder_mesh.users == 0:
    bpy.data.meshes.remove(cylinder_mesh)
```