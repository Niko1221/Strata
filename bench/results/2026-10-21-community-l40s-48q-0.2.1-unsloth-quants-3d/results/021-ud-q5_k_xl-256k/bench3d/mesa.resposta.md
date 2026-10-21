```python
import bpy


def add_box(name, center, dimensions):
    dx, dy, dz = dimensions

    x_min, x_max = -dx / 2.0, dx / 2.0
    y_min, y_max = -dy / 2.0, dy / 2.0
    z_min, z_max = -dz / 2.0, dz / 2.0

    verts = [
        (x_min, y_min, z_min),
        (x_max, y_min, z_min),
        (x_max, y_max, z_min),
        (x_min, y_max, z_min),
        (x_min, y_min, z_max),
        (x_max, y_min, z_max),
        (x_max, y_max, z_max),
        (x_min, y_max, z_max),
    ]

    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    bpy.context.scene.collection.objects.link(obj)

    return obj


# Tampo: 2.0 m x 1.0 m x 0.05 m, face superior em z = 0.75 m
add_box("Tampo", (0.0, 0.0, 0.725), (2.0, 1.0, 0.05))

# Pernas: seção 0.06 m x 0.06 m, do chão até a face inferior do tampo
leg_size = 0.06
leg_height = 0.70
table_x_half = 1.0
table_y_half = 0.5

for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)], start=1):
    x = sx * (table_x_half - leg_size / 2.0)
    y = sy * (table_y_half - leg_size / 2.0)
    add_box(f"Perna_{i}", (x, y, leg_height / 2.0), (leg_size, leg_size, leg_height))
```