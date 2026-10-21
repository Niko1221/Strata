```python
import bpy

# Criar dados de texto
text_data = bpy.data.curves.new(name="Texto", type='FONT')
text_data.body = "Olá 3D"
text_data.extrude = 0.1
text_data.align_x = 'CENTER'

# Criar objeto de texto
text_obj = bpy.data.objects.new(name="Texto", object_data=text_data)

# Vincular ao collection ativo
bpy.context.collection.objects.link(text_obj)
```