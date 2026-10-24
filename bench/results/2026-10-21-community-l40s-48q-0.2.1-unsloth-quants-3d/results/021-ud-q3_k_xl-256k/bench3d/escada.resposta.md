```python
import bpy

STEP_COUNT = 10
STEP_X = 1.0
STEP_Y = 0.30
STEP_Z = 0.18

def add_step(n):
    x_min = -STEP_X / 2.0
    x_max = STEP_X / 2.0

    y_min = (n - 1) * STEP_Y
    y_max = n * STEP_Y

    z_min = (n - 1) * STEP_Z
    z_max = n * STEP_Z

    vertices = [
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
        (0, 3, 2, 1),  # base
        (0, 1, 5, 4),  # frente (-Y)
        (1, 2, 6, 5),  # direita (+X)
        (2, 3, 7, 6),  # fundo (+Y)
        (3, 0, 4, 7),  # esquerda (-X)
        (4, 5, 6, 7),  # topo (+Z)
    ]

    mesh = bpy.data.meshes.new(f"Degrau_{n:02d}")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate()
    mesh.update()

    obj = bpy.data.objects.new(f"Degrau_{n:02d}", mesh)
    bpy.context.scene.collection.objects.link(obj)

for n in range(1, STEP_COUNT + 1):
    add_step(n)
```