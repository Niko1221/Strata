import bpy
import math
from mathutils import Matrix

scene = bpy.context.scene
collection = scene.collection

# Empty 'Pivo' em (0, 0, 1)
pivo = bpy.data.objects.new("Pivo", None)
collection.objects.link(pivo)
pivo.location = (0.0, 0.0, 1.0)
pivo.rotation_euler = (0.0, 0.0, 0.0)

# Malha do cubo com aresta de 0,5 m
mesh = bpy.data.meshes.new("Satelite")
size = 0.5
h = size / 2.0

verts = [
    (-h, -h, -h),
    ( h, -h, -h),
    ( h,  h, -h),
    (-h,  h, -h),
    (-h, -h,  h),
    ( h, -h,  h),
    ( h,  h,  h),
    (-h,  h,  h),
]

faces = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
]

mesh.from_pydata(verts, [], faces)
mesh.update()

# Cubo 'Satelite', filho de 'Pivo', deslocado 2 m em +X local
satelite = bpy.data.objects.new("Satelite", mesh)
collection.objects.link(satelite)

satelite.parent = pivo
satelite.parent_type = 'OBJECT'
satelite.matrix_parent_inverse = Matrix.Identity(4)
satelite.location = (2.0, 0.0, 0.0)
satelite.rotation_euler = (0.0, 0.0, 0.0)
satelite.scale = (1.0, 1.0, 1.0)

# Gira 'Pivo' 90° em torno de Z
pivo.rotation_euler = (0.0, 0.0, math.radians(90.0))

bpy.context.view_layer.update()
