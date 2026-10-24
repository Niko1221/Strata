import bpy

# Remove um objeto existente com o mesmo nome, caso o script seja executado novamente.
existing = bpy.data.objects.get("Metade")
if existing is not None:
    bpy.data.objects.remove(existing, do_unlink=True)

# Cria a malha original apenas no lado x > 0.
mesh = bpy.data.meshes.new("Metade")

# Cubo de 1 m centrado em x = 1:
# x de 0.5 a 1.5, y de -0.5 a 0.5, z de -0.5 a 0.5.
vertices = [
    (0.5, -0.5, -0.5),
    (1.5, -0.5, -0.5),
    (1.5, 0.5, -0.5),
    (0.5, 0.5, -0.5),
    (0.5, -0.5, 0.5),
    (1.5, -0.5, 0.5),
    (1.5, 0.5, 0.5),
    (0.5, 0.5, 0.5),
]

faces = [
    (0, 3, 2, 1),  # base
    (4, 5, 6, 7),  # topo
    (0, 1, 5, 4),  # frente
    (3, 7, 6, 2),  # trás
    (0, 4, 7, 3),  # esquerda
    (1, 2, 6, 5),  # direita
]

mesh.from_pydata(vertices, [], faces)
mesh.update()

# Cria o objeto com origem em x = 0, para que o Mirror espelhe em relação ao plano x = 0.
obj = bpy.data.objects.new("Metade", mesh)
bpy.context.scene.collection.objects.link(obj)

# Adiciona o modificador Mirror no eixo X, sem aplicá-lo.
mirror = obj.modifiers.new(name="Mirror", type="MIRROR")
mirror.use_axis_x = True
mirror.use_axis_y = False
mirror.use_axis_z = False
