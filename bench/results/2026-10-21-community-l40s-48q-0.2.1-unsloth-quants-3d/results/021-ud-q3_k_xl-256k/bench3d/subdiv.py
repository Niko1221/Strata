import bpy

# Cria uma esfera UV
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, 0.0))

# Garante que o objeto criado seja o ativo/selecionado
obj = bpy.context.active_object
if obj is None and bpy.context.selected_objects:
    obj = bpy.context.selected_objects[0]

# Aplica sombreamento suave
for polygon in obj.data.polygons:
    polygon.use_smooth = True

# Adiciona o modificador Subdivision Surface
subsurf = obj.modifiers.new(name="Subdivision Surface", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 2
