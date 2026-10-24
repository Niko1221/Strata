import bpy

# Remove qualquer objeto existente para garantir que apenas os cubos pedidos sejam criados.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# Remove dados de malha órfãos deixados por objetos removidos.
for mesh in list(bpy.data.meshes):
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)

# Parâmetros da grade
COLS = 3
ROWS = 4
SPACING = 1.0
CUBE_SIZE = 0.5

# Posições iniciais para centralizar a grade na origem
x_start = -((COLS - 1) * SPACING) / 2.0
y_start = -((ROWS - 1) * SPACING) / 2.0

for col in range(COLS):
    x = x_start + col * SPACING
    for row in range(ROWS):
        y = y_start + row * SPACING

        # Cubo de 0,5 m com centro em z = 0,25 m para apoiar no chão (z = 0)
        bpy.ops.mesh.primitive_cube_add(
            size=CUBE_SIZE,
            location=(x, y, CUBE_SIZE / 2.0)
        )

        cube = bpy.context.active_object
        cube.name = f"Cube_{col + 1}_{row + 1}"

        # Garante as dimensões exatas
        cube.dimensions = (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

        # Aplica a escala para manter a malha consistente
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
