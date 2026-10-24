import bpy

# Cria a malha da caixa com dimensões locais 3 m × 2 m × 1 m
mesh = bpy.data.meshes.new("Box")

verts = [
    (-1.5, -1.0, -0.5),
    ( 1.5, -1.0, -0.5),
    ( 1.5,  1.0, -0.5),
    (-1.5,  1.0, -0.5),
    (-1.5, -1.0,  0.5),
    ( 1.5, -1.0,  0.5),
    ( 1.5,  1.0,  0.5),
    (-1.5,  1.0,  0.5),
]

faces = [
    (3, 2, 1, 0),  # base
    (4, 5, 6, 7),  # topo
    (0, 4, 7, 3),  # lado -X
    (1, 2, 6, 5),  # lado +X
    (0, 1, 5, 4),  # lado -Y
    (3, 7, 6, 2),  # lado +Y
]

mesh.from_pydata(verts, [], faces)
mesh.update()

# Cria o objeto e vincula à cena
obj = bpy.data.objects.new("Box", mesh)
bpy.context.scene.collection.objects.link(obj)

# Posiciona a caixa apoiada no chão (z = 0)
obj.location = (0.0, 0.0, 0.5)

# Escala já aplicada
obj.scale = (1.0, 1.0, 1.0)

bpy.context.view_layer.update()
