import bpy

# Garante que apenas os objetos da mesa sejam criados.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)


def add_box(name, dimensions, location):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = dimensions
    return obj


# Dimensões da mesa
table_x = 2.0
table_y = 1.0
table_z = 0.05
table_top_z = 0.75

leg_size = 0.06
leg_height = table_top_z - table_z  # 0.70 m

# Posições
table_center_z = table_top_z - table_z / 2.0
leg_center_z = leg_height / 2.0

# Pernas encostadas nas bordas externas do tampo
leg_x = table_x / 2.0 - leg_size / 2.0
leg_y = table_y / 2.0 - leg_size / 2.0

# Tampo
add_box(
    "Tampo",
    (table_x, table_y, table_z),
    (0.0, 0.0, table_center_z),
)

# Quatro pernas nos cantos
for i, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1)), start=1):
    add_box(
        f"Perna_{i}",
        (leg_size, leg_size, leg_height),
        (sx * leg_x, sy * leg_y, leg_center_z),
    )
