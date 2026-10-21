import bpy
import math
from mathutils import Matrix

scene = bpy.context.scene

# Empty 'Pivo' em (0, 0, 1)
pivo = bpy.data.objects.new('Pivo', None)
pivo.location = (0.0, 0.0, 1.0)
scene.collection.objects.link(pivo)

# Cubo de 0,5 m chamado 'Satelite'
bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.0, 0.0, 0.0))
satelite = bpy.context.view_layer.objects.active
satelite.name = 'Satelite'
satelite.data.name = 'Satelite'

# Filho do Pivo, deslocado 2 m em +X local
satelite.parent = pivo
satelite.matrix_parent_inverse = Matrix.Identity(4)
satelite.location = (2.0, 0.0, 0.0)

# Gira o Pivo 90° em torno de Z
pivo.rotation_euler = (0.0, 0.0, math.radians(90.0))

bpy.context.view_layer.update()
