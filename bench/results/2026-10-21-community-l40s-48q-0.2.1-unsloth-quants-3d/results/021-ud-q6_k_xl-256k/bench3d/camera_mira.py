import bpy
from mathutils import Vector

scene = bpy.context.scene

camera_data = bpy.data.cameras.new("Camera")
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)

camera.location = (10.0, -10.0, 8.0)

direction = Vector((0.0, 0.0, 0.0)) - camera.location
camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

scene.camera = camera
