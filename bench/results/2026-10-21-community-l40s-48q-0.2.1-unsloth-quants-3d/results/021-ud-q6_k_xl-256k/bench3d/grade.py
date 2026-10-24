import bpy


def make_cube_mesh(name, size):
    half = size / 2.0

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
        (0, 3, 2, 1),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),  # front (-Y)
        (3, 7, 6, 2),  # back  (+Y)
        (0, 4, 7, 3),  # left  (-X)
        (1, 2, 6, 5),  # right (+X)
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def main():
    cube_size = 0.5
    spacing = 1.0
    columns = 3
    rows = 4

    x_start = -(columns - 1) * spacing / 2.0
    y_start = -(rows - 1) * spacing / 2.0
    z_center = cube_size / 2.0

    collection = bpy.context.scene.collection

    for col in range(columns):
        for row in range(rows):
            x = x_start + col * spacing
            y = y_start + row * spacing

            mesh = make_cube_mesh(f"CubeMesh_{col}_{row}", cube_size)
            obj = bpy.data.objects.new(f"Cube_{col}_{row}", mesh)
            obj.location = (x, y, z_center)

            collection.objects.link(obj)


main()
