import bpy

# Cria a malha do cubo com aresta de 2 m
mesh = bpy.data.meshes.new("Cube")

# Vértices: x e y de -1 a 1, z de 0 a 2
vertices = [
    (-1.0, -1.0, 0.0),  # 0
    ( 1.0, -1.0, 0.0),  # 1
    ( 1.0,  1.0, 0.0),  # 2
    (-1.0,  1.0, 0.0),  # 3
    (-1.0, -1.0, 2.0),  # 4
    ( 1.0, -1.0, 2.0),  # 5
    ( 1.0,  1.0, 2.0),  # 6
    (-1.0,  1.0, 2.0),  # 7
]

# Faces com normais apontando para fora
faces = [
    (0, 3, 2, 1),  # face inferior (z = 0)
    (4, 5, 6, 7),  # face superior (z = 2)
    (0, 1, 5, 4),  # frente (y = -1)
    (1, 2, 6, 5),  # direita (x = 1)
    (2, 3, 7, 6),  # trás (y = 1)
    (3, 0, 4, 7),  # esquerda (x = -1)
]

mesh.from_pydata(vertices, [], faces)
mesh.update()

# Cria o objeto e adiciona à cena
cube = bpy.data.objects.new("Cube", mesh)
bpy.context.scene.collection.objects.link(cube)
