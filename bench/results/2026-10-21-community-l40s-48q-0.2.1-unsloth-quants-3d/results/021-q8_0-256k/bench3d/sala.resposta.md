```python
import bpy
import bmesh

# Garante que apenas os objetos pedidos sejam criados.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

for mesh in list(bpy.data.meshes):
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def add_box(name, dimensions, location):
    """Cria um cubo com dimensões exatas e origem no centro."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    bmesh.ops.create_cube(bm, size=1.0)

    dx, dy, dz = dimensions
    for v in bm.verts:
        v.co.x *= dx
        v.co.y *= dy
        v.co.z *= dz

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    return obj


# Piso: 4 m em X, 5 m em Y, 0,1 m de espessura.
# Face superior em z = 0, portanto o centro fica em z = -0,05.
add_box("Piso", (4.0, 5.0, 0.1), (0.0, 0.0, -0.05))

# Paredes: 0,1 m de espessura e 2,5 m de altura, de z = 0 a z = 2,5.
# As faces externas ficam alinhadas às bordas externas do piso.

# Paredes norte e sul, ao longo do eixo X.
# Comprimento externo em X = 4,0 m.
add_box("Parede_Norte", (4.0, 0.1, 2.5), (0.0, 2.45, 1.25))
add_box("Parede_Sul", (4.0, 0.1, 2.5), (0.0, -2.45, 1.25))

# Paredes leste e oeste, ao longo do eixo Y.
# Comprimento externo em Y = 5,0 m, descontando a espessura das paredes norte/sul:
# 5,0 - 2 * 0,1 = 4,8 m.
add_box("Parede_Leste", (0.1, 4.8, 2.5), (1.95, 0.0, 1.25))
add_box("Parede_Oeste", (0.1, 4.8, 2.5), (-1.95, 0.0, 1.25))
```