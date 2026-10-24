import bpy

# Cria a malha manualmente: um cubo de 1 m centrado em x = 1,
# portanto todos os vértices originais têm x > 0.
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
    (0, 3, 2, 1),  # face inferior
    (4, 5, 6, 7),  # face superior
    (0, 1, 5, 4),  # face frontal
    (1, 2, 6, 5),  # face direita
    (2, 3, 7, 6),  # face posterior
    (3, 0, 4, 7),  # face esquerda
]

mesh.from_pydata(vertices, [], faces)
mesh.update()

# Cria o objeto com origem no mundo (x = 0), para que o modificador Mirror
# espelhe em relação ao plano x = 0.
obj = bpy.data.objects.new("Metade", mesh)
obj.location = (0.0, 0.0, 0.0)
bpy.context.scene.collection.objects.link(obj)

# Adiciona o modificador Mirror no eixo X, sem aplicá-lo.
mirror = obj.modifiers.new(name="Mirror", type='MIRROR')
mirror.use_axis = (True, False, False)
