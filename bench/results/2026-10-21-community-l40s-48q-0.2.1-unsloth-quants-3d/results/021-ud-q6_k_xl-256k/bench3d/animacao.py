import bpy

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 50

bpy.ops.mesh.primitive_cube_add(size=1, location=(0.0, 0.0, 0.0))
movel = bpy.context.active_object
movel.name = "Movel"

movel.location = (0.0, 0.0, 0.0)
movel.keyframe_insert(data_path="location", frame=1)

movel.location = (5.0, 0.0, 0.0)
movel.keyframe_insert(data_path="location", frame=50)

scene.frame_set(1)
