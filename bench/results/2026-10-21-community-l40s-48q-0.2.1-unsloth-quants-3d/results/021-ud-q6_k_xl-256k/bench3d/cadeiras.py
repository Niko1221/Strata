import bpy

def add_box(name, dimensions, location):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.dimensions = dimensions
    return obj

scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.length_unit = 'METERS'

# Mesa: caixa 1.6 m (X) × 0.9 m (Y) × 0.75 m (Z)
# Apoiada no chão (z = 0) e centrada em XY na origem.
table_dims = (1.6, 0.9, 0.75)
table_loc = (0.0, 0.0, table_dims[2] / 2.0)
add_box("Mesa", table_dims, table_loc)

# Cadeiras: caixas de 0.45 m de lado, apoiadas no chão.
# Uma de cada lado da mesa no eixo Y, centradas em x = 0.
# Folga de 0.2 m entre a face da cadeira e a face da mesa.
chair_side = 0.45
chair_dims = (chair_side, chair_side, chair_side)
chair_z = chair_side / 2.0

table_y_half = table_dims[1] / 2.0
gap = 0.2
chair_y_center = table_y_half + gap + chair_side / 2.0

add_box("Cadeira_1", chair_dims, (0.0, chair_y_center, chair_z))
add_box("Cadeira_2", chair_dims, (0.0, -chair_y_center, chair_z))
