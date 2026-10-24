```python
import bpy
import bmesh

# Garanta que a cena final contenha apenas os objetos pedidos.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

for mesh in list(bpy.data.meshes):
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def add_box(name, dimensions, location):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)

    dx, dy, dz = dimensions
    for vert in bm.verts:
        vert.co.x *= dx
        vert.co.y *= dy
        vert.co.z *= dz

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    return obj


# Tampo: 2,0 m (X) × 1,0 m (Y) × 0,05 m de espessura.
# Face superior em z = 0,75 m, logo o centro fica em z = 0,725 m.
add_box("Tampo", (2.0, 1.0, 0.05), (0.0, 0.0, 0.725))

# Pernas: seção 0,06 m × 0,06 m, do chão até a face inferior do tampo.
# Face inferior do tampo: z = 0,75 - 0,05 = 0,70 m.
leg_side = 0.06
leg_height = 0.70

# Centros das pernas para que as faces externas fiquem alinhadas às bordas:
# bordas externas do tampo em x = ±1,0 e y = ±0,5.
leg_x = 1.0 - leg_side / 2.0
leg_y = 0.5 - leg_side / 2.0
leg_z = leg_height / 2.0

for index, (sign_x, sign_y) in enumerate(((1, 1), (-1, 1), (1, -1), (-1, -1)), start=1):
    add_box(
        f"Perna_{index}",
        (leg_side, leg_side, leg_height),
        (sign_x * leg_x, sign_y * leg_y, leg_z),
    )
```