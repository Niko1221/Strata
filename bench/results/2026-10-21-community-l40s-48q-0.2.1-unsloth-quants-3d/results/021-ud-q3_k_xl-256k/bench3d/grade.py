import bpy

# Parâmetros da grade
cols = 3
rows = 4
spacing = 1.0
cube_size = 0.5

# Cria uma malha de cubo centrada na origem
half = cube_size / 2.0
vertices = [
    (-half, -half, -half),
    ( half, -half, -half),
    ( half,  half, -half),
    (-half,  half, -half),
    (-half, -half,  half),
    ( half, -half,  half),
    ( half,  half,  half),
    (-half,  half,  half),
]

faces = [
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
]

mesh = bpy.data.meshes.new("CubeMesh")
mesh.from_pydata(vertices, [], faces)
mesh.update()

scene = bpy.context.scene
collection = scene.collection

# Posição Z do centro para que a base fique no chão (z = 0)
z_center = cube_size / 2.0

# Cria a grade centrada na origem
count = 0
for i in range(cols):
    x = (i - (cols - 1) / 2.0) * spacing
    for j in range(rows):
        y = (j - (rows - 1) / 2.0) * spacing

        obj = bpy.data.objects.new(f"Cube_{count:03d}", mesh)
        obj.location = (x, y, z_center)
        collection.objects.link(obj)

        count += 1
