import bpy
import bmesh

STEP_COUNT = 10
STEP_WIDTH = 1.0
STEP_DEPTH = 0.30
STEP_HEIGHT = 0.18


def add_step(index):
    name = f"Step_{index:02d}"
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    hx = STEP_WIDTH / 2.0
    hy = STEP_DEPTH / 2.0
    hz = STEP_HEIGHT / 2.0

    verts = [
        bm.verts.new((-hx, -hy, -hz)),
        bm.verts.new(( hx, -hy, -hz)),
        bm.verts.new(( hx,  hy, -hz)),
        bm.verts.new((-hx,  hy, -hz)),
        bm.verts.new((-hx, -hy,  hz)),
        bm.verts.new(( hx, -hy,  hz)),
        bm.verts.new(( hx,  hy,  hz)),
        bm.verts.new((-hx,  hy,  hz)),
    ]

    faces = [
        (0, 3, 2, 1),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),  # -Y
        (1, 2, 6, 5),  # +X
        (2, 3, 7, 6),  # +Y
        (3, 0, 4, 7),  # -X
    ]

    for face in faces:
        bm.faces.new([verts[i] for i in face])

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    obj.location = (
        0.0,
        (index - 0.5) * STEP_DEPTH,
        (index - 0.5) * STEP_HEIGHT,
    )

    return obj


for n in range(1, STEP_COUNT + 1):
    add_step(n)
