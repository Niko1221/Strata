```python
import bpy

# Garante que a cena fique apenas com o objeto pedido.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# Cria o cubo de 2 m centrado na origem.
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
cube = bpy.context.active_object
cube.name = "Cubo_Furado"

# Cria o cilindro cortante: raio 0,5 m, atravessando o cubo no eixo Z.
bpy.ops.mesh.primitive_cylinder_add(
    radius=0.5,
    depth=4.0,
    location=(0.0, 0.0, 0.0),
    vertices=64,
)
cutter = bpy.context.active_object
cutter.name = "Cilindro_Furo"

# Adiciona o modificador booleano de diferença.
boolean = cube.modifiers.new(name="Furo", type="BOOLEAN")
boolean.operation = "DIFFERENCE"
boolean.object = cutter

if hasattr(boolean, "solver"):
    boolean.solver = "EXACT"

# Aplica o booleano usando a malha avaliada, sem depender de contexto de UI.
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()

evaluated_cube = cube.evaluated_get(depsgraph)

try:
    new_mesh = bpy.data.meshes.new_from_object(evaluated_cube, depsgraph=depsgraph)
except TypeError:
    new_mesh = bpy.data.meshes.new_from_object(evaluated_cube)

old_mesh = cube.data
cube.data = new_mesh
cube.modifiers.clear()

if old_mesh and old_mesh.users == 0:
    bpy.data.meshes.remove(old_mesh)

# Remove o cilindro cortante.
cutter_mesh = cutter.data
bpy.data.objects.remove(cutter, do_unlink=True)

if cutter_mesh and cutter_mesh.users == 0:
    bpy.data.meshes.remove(cutter_mesh)

# Deixa apenas o cubo furado selecionado e ativo.
for obj in bpy.context.selected_objects:
    obj.select_set(False)

cube.select_set(True)
bpy.context.view_layer.objects.active = cube
```