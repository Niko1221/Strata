```python
import bpy
from mathutils import Vector

camera_data = bpy.data.cameras.new("Camera")
camera = bpy.data.objects.new("Camera", camera_data)

camera.location = (10.0, -10.0, 8.0)

direction = Vector((0.0, 0.0, 0.0)) - camera.location
camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

bpy.context.scene.collection.objects.link(camera)
bpy.context.scene.camera = camera
```