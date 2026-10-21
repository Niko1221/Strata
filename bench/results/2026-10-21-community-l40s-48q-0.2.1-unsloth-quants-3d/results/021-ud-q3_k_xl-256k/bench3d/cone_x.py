import bpy

# Cria um cone padrão ao longo do eixo Z, centrado na origem
bpy.ops.mesh.primitive_cone_add(
    radius1=0.5,
    radius2=0.0,
    depth=2.0,
    location=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0)
)

obj = bpy.context.active_object
if obj is None:
    obj = bpy.context.view_layer.objects.active

# Move a malha para que o centro da base fique na origem local
min_z = min(v.co.z for v in obj.data.vertices)
for v in obj.data.vertices:
    v.co.z -= min_z

# Gira os vértices da malha de +Z para +X
for v in obj.data.vertices:
    x, y, z = v.co
    v.co = (z, y, -x)

# Garante que o objeto esteja na origem, sem rotação ou escala
obj.location = (0.0, 0.0, 0.0)
obj.rotation_euler = (0.0, 0.0, 0.0)
obj.scale = (1.0, 1.0, 1.0)
