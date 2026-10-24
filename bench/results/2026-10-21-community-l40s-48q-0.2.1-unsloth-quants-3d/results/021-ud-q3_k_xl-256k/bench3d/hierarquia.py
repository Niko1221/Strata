import bpy
from mathutils import Matrix

scene = bpy.context.scene
collection = scene.collection

# Empty 'Pivo'
pivo = bpy.data.objects.new('Pivo', None)
collection.objects.link(pivo)

# Posição (0, 0, 1) e rotação exata de +90° em torno de Z.
# Matriz de rotação exata para que o resultado final seja (0, 2, 1).
rot_z_90 = Matrix((
    (0.0, -1.0, 0.0, 0.0),
    (1.0,  0.0, 0.0, 0.0),
    (0.0,  0.0, 1.0, 0.0),
    (0.0,  0.0, 0.0, 1.0),
))

pivo.matrix_basis = Matrix.Translation((0.0, 0.0, 1.0)) @ rot_z_90

# Cubo de 0,5 m criado manualmente como malha centrada na origem.
half = 0.25
vertices = [
    (-half, -half, -half),
    ( half, -half, -half),
    ( half,  half, -half),
    (-half,  half, -half),
    (-half, -half,  half),
    ( half, -half,  half),
    ( half,  half,  half),
    (-half,  half,  half),
]

faces = [
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
]

mesh = bpy.data.meshes.new('Satelite')
mesh.from_pydata(vertices, [], faces)
mesh.update()

satelite = bpy.data.objects.new('Satelite', mesh)
collection.objects.link(satelite)

# Filho do Pivo, deslocado 2 m em +X local do Pivo.
satelite.parent = pivo
satelite.matrix_parent_inverse = Matrix.Identity(4)
satelite.matrix_basis = Matrix.Translation((2.0, 0.0, 0.0))

# Atualiza as matrizes de mundo.
bpy.context.view_layer.update()
