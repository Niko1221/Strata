```python
import bpy
import math
from mathutils import Matrix

scene = bpy.context.scene

# Empty 'Pivo' em (0, 0, 1)
pivo = bpy.data.objects.new('Pivo', None)
pivo.location = (0.0, 0.0, 1.0)
scene.collection.objects.link(pivo)

# Cubo de 0,5 m chamado 'Satelite'
half = 0.25
verts = [
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
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]

mesh = bpy.data.meshes.new('Satelite')
mesh.from_pydata(verts, [], faces)
mesh.update()

satelite = bpy.data.objects.new('Satelite', mesh)
scene.collection.objects.link(satelite)

# Filho do Pivo, deslocado 2 m em +X local
satelite.parent = pivo
satelite.matrix_parent_inverse = Matrix.Identity(4)
satelite.location = (2.0, 0.0, 0.0)

# Gira o Pivo 90° em torno de Z
pivo.rotation_euler = (0.0, 0.0, math.radians(90.0))

bpy.context.view_layer.update()
```