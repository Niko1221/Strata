```python
import bpy

bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 1.0))
cube = bpy.data.objects[-1]
cube.name = "Cube"
cube.dimensions = (2.0, 2.0, 2.0)
cube.location = (0.0, 0.0, 1.0)

bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, 3.0))
sphere = bpy.data.objects[-1]
sphere.name = "Sphere"
sphere.dimensions = (2.0, 2.0, 2.0)
sphere.location = (0.0, 0.0, 3.0)
```