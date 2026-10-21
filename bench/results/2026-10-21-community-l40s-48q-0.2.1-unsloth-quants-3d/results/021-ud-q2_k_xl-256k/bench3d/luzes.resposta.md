```python
import bpy

# Luz do tipo Sol com força 3, apontando diretamente para baixo (-Z)
sun_data = bpy.data.lights.new(name="Sun", type='SUN')
sun_data.energy = 3.0
sun_obj = bpy.data.objects.new(name="Sun", object_data=sun_data)
sun_obj.location = (0.0, 0.0, 0.0)
sun_obj.rotation_euler = (0.0, 0.0, 0.0)
bpy.context.scene.collection.objects.link(sun_obj)

# Luz de área quadrada 2 m × 2 m em (0, 0, 4), apontando para baixo, potência 500 W
area_data = bpy.data.lights.new(name="Area", type='AREA')
area_data.shape = 'SQUARE'
area_data.size = 2.0
area_data.size_y = 2.0
area_data.energy = 500.0
area_obj = bpy.data.objects.new(name="Area", object_data=area_data)
area_obj.location = (0.0, 0.0, 4.0)
area_obj.rotation_euler = (0.0, 0.0, 0.0)
bpy.context.scene.collection.objects.link(area_obj)
```