import bpy
import math

scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0


def add_cube(name, z, rot_z):
    mesh = bpy.data.meshes.new(name)

    verts = [
        (-0.5, -0.5, -0.5),
        ( 0.5, -0.5, -0.5),
        ( 0.5,  0.5, -0.5),
        (-0.5,  0.5, -0.5),
        (-0.5, -0.5,  0.5),
        ( 0.5, -0.5,  0.5),
        ( 0.5,  0.5,  0.5),
        (-0.5,  0.5,  0.5),
    ]

    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)

    obj.location = (0.0, 0.0, z)
    obj.rotation_euler = (0.0, 0.0, rot_z)


add_cube("Cube_1", 0.5, 0.0)
add_cube("Cube_2", 1.5, math.radians(15.0))
add_cube("Cube_3", 2.5, math.radians(30.0))
