```python
import bpy
import math
from mathutils import Matrix

scene = bpy.context.scene
collection = scene.collection

# Empty 'Pivo' em (0, 0, 1)
pivo = bpy.data.objects.new('Pivo', None)
pivo.empty_display_type = 'PLAIN_AXES'
pivo.location = (0.0, 0.0, 1.0)
collection.objects.link(pivo)

# Cubo 'Satelite' com aresta de 0,5 m
mesh = bpy.data.meshes.new('Satelite')
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
    (3, 2, 1, 0),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]

mesh.from_pydata(verts, [], faces)
mesh.update()

satelite = bpy.data.objects.new('Satelite', mesh)
collection.objects.link(satelite)

# Filho do Pivo, deslocado 2 m em +X local
satelite.parent = pivo
satelite.matrix_parent_inverse = Matrix.Identity(4)
satelite.matrix_basis = Matrix.Translation((2.0, 0.0, 0.0))

# Gira o Pivo 90° em torno de Z
pivo.rotation_euler = (0.0, 0.0, math.radians(90.0))

bpy.context.view_layer.update()
```