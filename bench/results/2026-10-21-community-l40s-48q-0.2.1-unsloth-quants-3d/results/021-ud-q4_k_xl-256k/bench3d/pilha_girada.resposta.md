```python
import bpy
import math

for i, angle_deg in enumerate((0.0, 15.0, 30.0)):
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(0.0, 0.0, 0.5 + i),
        rotation=(0.0, 0.0, math.radians(angle_deg))
    )
```