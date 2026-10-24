import bpy

bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 1.0))

cube = bpy.context.active_object
cube.name = "Cube"
cube.dimensions = (2.0, 2.0, 2.0)

bpy.context.view_layer.update()
