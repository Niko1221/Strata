```python
import bpy
from math import radians
from mathutils import Matrix

# Empty 'Pivo' em (0, 0, 1)
pivo = bpy.data.objects.new('Pivo', None)
pivo.location = (0.0, 0.0, 1.0)
bpy.context.scene.collection.objects.link(pivo)

# Cubo de 0,5 m chamado 'Satelite'
mesh = bpy.data.meshes.new('Satelite')
s = 0.25  # metade do lado: cubo com dimensões 0.5 x 0.5 x 0.5

verts = [
    (-s, -s, -s),
    ( s, -s, -s),
    ( s,  s, -s),
    (-s,  s, -s),
    (-s, -s,  s),
    ( s, -s,  s),
    ( s,  s,  s),
    (-s,  s,  s),
]

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

faces = [
    (0, 3, 2, 1),  # baixo
    (4, 5, 6, 7),  # cima
    (0, 1, 5, 4),  # frente
    (1, 2, 6, 5),  # direita
    (2, 3, 7, 6),  # trás
    (3, 0, 4, 7),  # esquerda
]

mesh.from_pydata(verts, edges, faces)
mesh.update()

satelite = bpy.data.objects.new('Satelite', mesh)
bpy.context.scene.collection.objects.link(satelite)

# Filha do Pivo, deslocada 2 m em +X local
satelite.parent = pivo
satelite.matrix_parent_inverse = Matrix.Identity(4)
satelite.location = (2.0, 0.0, 0.0)

# Gira o Pivo 90° em torno de Z
pivo.rotation_euler = (0.0, 0.0, radians(90))

bpy.context.view_layer.update()
```