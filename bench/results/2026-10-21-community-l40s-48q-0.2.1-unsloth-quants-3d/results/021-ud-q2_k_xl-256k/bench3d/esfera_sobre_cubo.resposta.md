```python
import bpy

# Garanta que apenas os objetos pedidos existam na cena
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# Cubo de 2 m apoiado no chão (base em z = 0), centrado em x = 0, y = 0
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
cube = bpy.context.active_object
cube.name = "Cube"

# Esfera de raio 1 m apoiada exatamente no topo do cubo (base em z = 2)
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 3))
sphere = bpy.context.active_object
sphere.name = "Sphere"
```