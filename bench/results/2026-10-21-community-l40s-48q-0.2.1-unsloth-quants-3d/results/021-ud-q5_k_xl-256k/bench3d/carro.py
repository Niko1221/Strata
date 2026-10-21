import bpy
import math


def create_cylinder_mesh(name, radius, depth, segments=32):
    mesh = bpy.data.meshes.new(name)
    half_depth = depth / 2.0

    verts = []
    faces = []

    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        verts.append((x, y, -half_depth))
        verts.append((x, y, half_depth))

    for i in range(segments):
        b0 = 2 * i
        b1 = 2 * ((i + 1) % segments)
        t1 = 2 * ((i + 1) % segments) + 1
        t0 = 2 * i + 1
        faces.append((b0, b1, t1, t0))

    bottom_center = len(verts)
    verts.append((0.0, 0.0, -half_depth))

    top_center = len(verts)
    verts.append((0.0, 0.0, half_depth))

    for i in range(segments):
        b0 = 2 * i
        b1 = 2 * ((i + 1) % segments)
        t0 = 2 * i + 1
        t1 = 2 * ((i + 1) % segments) + 1

        faces.append((bottom_center, b1, b0))
        faces.append((top_center, t0, t1))

    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


scene = bpy.context.scene
collection = scene.collection

empty = bpy.data.objects.new("Carro", None)
collection.objects.link(empty)
empty.location = (0.0, 0.0, 0.0)

bpy.context.view_layer.update()

wheel_mesh = create_cylinder_mesh("RodaMesh", 0.3, 0.2)

positions = [
    (1.2, 0.8, 0.3),
    (1.2, -0.8, 0.3),
    (-1.2, 0.8, 0.3),
    (-1.2, -0.8, 0.3),
]

for index, location in enumerate(positions, start=1):
    wheel = bpy.data.objects.new(f"Roda_{index}", wheel_mesh)
    collection.objects.link(wheel)

    wheel.parent = empty
    wheel.matrix_parent_inverse = empty.matrix_world.inverted()
    wheel.location = location
    wheel.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
