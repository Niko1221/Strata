```python
import bpy

# Criar o cubo de 2 m centrado na origem
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
cube = bpy.context.active_object
cube.name = "Cube"

# Criar o cilindro cortador (raio 0.5 m, atravessando todo o cubo)
bpy.ops.mesh.primitive_cylinder_add(
    radius=0.5,
    depth=4.0,
    location=(0.0, 0.0, 0.0),
    vertices=64
)
cylinder = bpy.context.active_object
cylinder.name = "HoleCutter"

# Adicionar modificador booleano de diferença no cubo
modifier = cube.modifiers.new(name="Hole", type='BOOLEAN')
modifier.operation = 'DIFFERENCE'
modifier.object = cylinder
modifier.solver = 'EXACT'

# Aplicar o modificador booleano
for obj in bpy.context.view_layer.objects:
    obj.select_set(False)

cube.select_set(True)
bpy.context.view_layer.objects.active = cube
bpy.ops.object.modifier_apply(modifier=modifier.name)

# Remover o cilindro cortador
bpy.data.objects.remove(cylinder, do_unlink=True)
```