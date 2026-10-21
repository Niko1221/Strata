import bpy
import math

# Garante que apenas os cubos pedidos existam na cena.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

cube_size = 1.0
half_size = cube_size / 2.0

# Primeiro cubo: apoiado no chão, sem rotação.
bpy.ops.mesh.primitive_cube_add(
    size=cube_size,
    location=(0.0, 0.0, half_size),
    rotation=(0.0, 0.0, 0.0),
)
bpy.context.active_object.name = "Cube_1"

# Segundo cubo: apoiado sobre o primeiro, girado 15° em torno do eixo Z.
bpy.ops.mesh.primitive_cube_add(
    size=cube_size,
    location=(0.0, 0.0, half_size + cube_size),
    rotation=(0.0, 0.0, math.radians(15.0)),
)
bpy.context.active_object.name = "Cube_2"

# Terceiro cubo: apoiado sobre o segundo, girado 30° em torno do eixo Z.
bpy.ops.mesh.primitive_cube_add(
    size=cube_size,
    location=(0.0, 0.0, half_size + 2.0 * cube_size),
    rotation=(0.0, 0.0, math.radians(30.0)),
)
bpy.context.active_object.name = "Cube_3"
