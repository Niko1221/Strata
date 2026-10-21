```python
import bpy
from mathutils import Vector

# Criar os dados da câmera
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)

# Vincular à cena
bpy.context.collection.objects.link(cam)

# Posição da câmera
cam.location = (10.0, -10.0, 8.0)

# Apontar exatamente para a origem
target = Vector((0.0, 0.0, 0.0))
direction = target - cam.location
cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# Definir como câmera ativa da cena
bpy.context.scene.camera = cam
```