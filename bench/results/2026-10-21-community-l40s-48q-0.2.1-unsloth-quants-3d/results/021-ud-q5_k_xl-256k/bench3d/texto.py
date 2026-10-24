import bpy

text_curve = bpy.data.curves.new(name="Texto", type='FONT')
text_curve.body = "Olá 3D"
text_curve.extrude = 0.1
text_curve.align_x = 'CENTER'

text_obj = bpy.data.objects.new(name="Texto", object_data=text_curve)
bpy.context.scene.collection.objects.link(text_obj)
