import bpy

scene = bpy.context.scene

sun_data = bpy.data.lights.new("Sun", type='SUN')
sun_data.energy = 3.0
sun_obj = bpy.data.objects.new("Sun", sun_data)
sun_obj.location = (0.0, 0.0, 0.0)
sun_obj.rotation_euler = (0.0, 0.0, 0.0)
scene.collection.objects.link(sun_obj)

area_data = bpy.data.lights.new("Area", type='AREA')
area_data.shape = 'SQUARE'
area_data.size = 2.0
area_data.size_y = 2.0
area_data.energy = 500.0
area_obj = bpy.data.objects.new("Area", area_data)
area_obj.location = (0.0, 0.0, 4.0)
area_obj.rotation_euler = (0.0, 0.0, 0.0)
scene.collection.objects.link(area_obj)
