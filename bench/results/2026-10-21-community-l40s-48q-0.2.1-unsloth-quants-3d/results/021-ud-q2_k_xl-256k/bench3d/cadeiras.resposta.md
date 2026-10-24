```python
import bpy

# Remove qualquer objeto existente para criar apenas o pedido.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 1.0


def add_box(name, dimensions, location):
    hx, hy, hz = (d / 2.0 for d in dimensions)

    verts = [
        (-hx, -hy, -hz),
        ( hx, -hy, -hz),
        ( hx,  hy, -hz),
        (-hx,  hy, -hz),
        (-hx, -hy,  hz),
        ( hx, -hy,  hz),
        ( hx,  hy,  hz),
        (-hx,  hy,  hz),
    ]

    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.dimensions = dimensions

    scene.collection.objects.link(obj)
    return obj


# Mesa: 1.6 m (X) × 0.9 m (Y) × 0.75 m, apoiada no chão e centrada em XY na origem.
table_dims = (1.6, 0.9, 0.75)
add_box("Mesa", table_dims, (0.0, 0.0, table_dims[2] / 2.0))

# Cadeiras: cubos de 0.45 m, apoiadas no chão, uma de cada lado no eixo Y.
chair_side = 0.45
gap = 0.2
table_half_y = table_dims[1] / 2.0
chair_center_y = table_half_y + gap + chair_side / 2.0
chair_dims = (chair_side, chair_side, chair_side)

add_box("Cadeira_1", chair_dims, (0.0, chair_center_y, chair_side / 2.0))
add_box("Cadeira_2", chair_dims, (0.0, -chair_center_y, chair_side / 2.0))
```