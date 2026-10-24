import bpy

text_data = bpy.data.curves.new(name="Olá 3D", type="FONT")
text_data.body = "Olá 3D"
text_data.extrude = 0.1
text_data.align_x = "CENTER"

text_obj = bpy.data.objects.new(name="Olá 3D", object_data=text_data)
bpy.context.collection.objects.link(text_obj)
