import bpy

# Cubo de 2 m apoiado no chão: centro em z = 1
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 1.0))
cube = bpy.context.active_object
cube.name = "Cube"

# Esfera de raio 1 m apoiada no topo do cubo: centro em z = 3
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, 3.0))
sphere = bpy.context.active_object
sphere.name = "Sphere"
