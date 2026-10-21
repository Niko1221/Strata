import bpy
from mathutils import Vector

scene = bpy.context.scene

camera_data = bpy.data.cameras.new("Camera")
camera_obj = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera_obj)

camera_obj.location = (10.0, -10.0, 8.0)

target = Vector((0.0, 0.0, 0.0))
direction = target - camera_obj.location
camera_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

scene.camera = camera_obj
