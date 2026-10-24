```python
import bpy
import math

for i, rot_deg in enumerate((0.0, 15.0, 30.0)):
    z_center = 0.5 + i

    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(0.0, 0.0, z_center)
    )

    cube = bpy.context.active_object
    cube.name = f"Cubo_{i + 1}"
    cube.rotation_euler = (0.0, 0.0, math.radians(rot_deg))
```