```python
import bpy

# Criar o cubo de 2 m centrado na origem
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
cube = bpy.context.active_object
cube.name = "Cube"

# Criar o cilindro cortador (maior que o cubo para atravessar de cima a baixo)
bpy.ops.mesh.primitive_cylinder_add(
    vertices=64,
    radius=0.5,
    depth=4.0,
    location=(0.0, 0.0, 0.0)
)
cutter = bpy.context.active_object
cutter.name = "Cylinder"

# Adicionar modificador booleano de diferença no cubo
mod = cube.modifiers.new(name="Hole", type='BOOLEAN')
mod.operation = 'DIFFERENCE'
mod.object = cutter
mod.solver = 'EXACT'

# Aplicar o modificador booleano
for obj in bpy.context.selected_objects:
    obj.select_set(False)

cube.select_set(True)
bpy.context.view_layer.objects.active = cube
bpy.context.view_layer.update()

bpy.ops.object.modifier_apply(modifier=mod.name)

# Remover o cilindro cortador
cutter_mesh = cutter.data
bpy.data.objects.remove(cutter, do_unlink=True)

if cutter_mesh and cutter_mesh.users == 0:
    bpy.data.meshes.remove(cutter_mesh)
```