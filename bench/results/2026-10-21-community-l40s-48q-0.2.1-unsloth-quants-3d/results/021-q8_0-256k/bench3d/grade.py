import bpy

size = 0.5
spacing = 1.0
cols = 3
rows = 4

x_positions = [(i - (cols - 1) / 2) * spacing for i in range(cols)]
y_positions = [(j - (rows - 1) / 2) * spacing for j in range(rows)]
z_center = size / 2

for x in x_positions:
    for y in y_positions:
        bpy.ops.mesh.primitive_cube_add(size=size, location=(x, y, z_center))
