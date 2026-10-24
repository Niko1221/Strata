import bpy

# Remove um objeto anterior com o mesmo nome, se existir
old_obj = bpy.data.objects.get("Metade")
if old_obj is not None:
    old_mesh = old_obj.data
    bpy.data.objects.remove(old_obj, do_unlink=True)
    if old_mesh is not None and old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)

# Cria a malha do cubo de 1 m centrado em x = 1, portanto totalmente em x > 0
mesh = bpy.data.meshes.new("Metade")

vertices = [
    (0.5, -0.5, -0.5),
    (1.5, -0.5, -0.5),
    (1.5,  0.5, -0.5),
    (0.5,  0.5, -0.5),
    (0.5, -0.5,  0.5),
    (1.5, -0.5,  0.5),
    (1.5,  0.5,  0.5),
    (0.5,  0.5,  0.5),
]

faces = [
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (3, 7, 6, 2),
    (1, 2, 6, 5),
    (0, 4, 7, 3),
]

mesh.from_pydata(vertices, [], faces)
mesh.update()

# Cria o objeto com origem na origem mundial (x = 0),
# para que o Mirror espelhe em relação ao plano x = 0
obj = bpy.data.objects.new("Metade", mesh)
bpy.context.scene.collection.objects.link(obj)

obj.location = (0.0, 0.0, 0.0)
obj.rotation_euler = (0.0, 0.0, 0.0)
obj.scale = (1.0, 1.0, 1.0)

# Adiciona o modificador Mirror no eixo X, sem aplicá-lo
mirror = obj.modifiers.new(name="Mirror", type='MIRROR')
mirror.use_axis_x = True
mirror.use_axis_y = False
mirror.use_axis_z = False
