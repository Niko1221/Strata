import bpy

scene = bpy.context.scene

scene.render.engine = 'CYCLES'
scene.cycles.samples = 64

scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100

scene.render.image_settings.file_format = 'PNG'
