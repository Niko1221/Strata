```python
import bpy

curve = bpy.data.curves.new(name="TextoCurve", type="FONT")
curve.body = "Olá 3D"
curve.extrude = 0.1
curve.align_horizontal = "CENTER"

obj = bpy.data.objects.new(name="Texto", object_data=curve)
bpy.context.scene.collection.objects.link(obj)
```