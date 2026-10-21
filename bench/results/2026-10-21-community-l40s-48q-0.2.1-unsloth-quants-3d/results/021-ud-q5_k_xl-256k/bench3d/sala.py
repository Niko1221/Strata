import bpy
import bmesh
from mathutils import Vector


def add_box(name, dimensions, location):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    # Cria um cubo unitário (aresta = 1) e escala para as dimensões desejadas.
    bmesh.ops.create_cube(bm, size=1.0)

    dx, dy, dz = dimensions
    loc = Vector(location)

    for v in bm.verts:
        v.co.x *= dx
        v.co.y *= dy
        v.co.z *= dz
        v.co += loc

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# Garante unidades métricas em metros.
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0

# Piso: 4 m (X) × 5 m (Y) × 0,1 m, face superior em z = 0.
add_box("Floor", (4.0, 5.0, 0.1), (0.0, 0.0, -0.05))

# Paredes: espessura 0,1 m, altura 2,5 m, de z = 0 a z = 2,5.
# Faces externas alinhadas às bordas externas do piso.
# As paredes norte/sul têm comprimento externo total em X = 4 m.
# As paredes leste/oeste têm comprimento interno em Y = 5 m - 2*0,1 m = 4,8 m,
# evitando sobreposição nos cantos.
add_box("Wall_North", (4.0, 0.1, 2.5), (0.0,  2.45, 1.25))
add_box("Wall_South", (4.0, 0.1, 2.5), (0.0, -2.45, 1.25))
add_box("Wall_East",  (0.1, 4.8, 2.5), ( 1.95, 0.0, 1.25))
add_box("Wall_West",  (0.1, 4.8, 2.5), (-1.95, 0.0, 1.25))
