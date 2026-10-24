import bpy
import bmesh

# Remove qualquer objeto existente para que a cena contenha apenas a mesa.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# Remove malhas órfãs, se houver.
for mesh in list(bpy.data.meshes):
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def add_box(name, dimensions, location):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)

    dx, dy, dz = dimensions
    for vert in bm.verts:
        vert.co.x *= dx
        vert.co.y *= dy
        vert.co.z *= dz

    bm.to_mesh(mesh)
    bm.free()

    obj.location = location
    return obj


# Dimensões do tampo.
tabletop_x = 2.0
tabletop_y = 1.0
tabletop_thickness = 0.05
tabletop_top_z = 0.75
tabletop_bottom_z = tabletop_top_z - tabletop_thickness
tabletop_center_z = (tabletop_top_z + tabletop_bottom_z) / 2.0

# Tampo.
add_box(
    "Tampo",
    (tabletop_x, tabletop_y, tabletop_thickness),
    (0.0, 0.0, tabletop_center_z),
)

# Dimensões das pernas.
leg_size = 0.06
leg_height = tabletop_bottom_z
leg_center_z = leg_height / 2.0

# Posições dos centros das pernas, com faces externas alinhadas às bordas do tampo.
x_offset = tabletop_x / 2.0 - leg_size / 2.0
y_offset = tabletop_y / 2.0 - leg_size / 2.0

leg_index = 1
for sx in (-1.0, 1.0):
    for sy in (-1.0, 1.0):
        add_box(
            f"Perna_{leg_index:02d}",
            (leg_size, leg_size, leg_height),
            (sx * x_offset, sy * y_offset, leg_center_z),
        )
        leg_index += 1
