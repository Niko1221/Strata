```python
import bpy
import math

bpy.ops.mesh.primitive_cylinder_add(
    vertices=32,
    radius=0.25,
    depth=4.0,
    location=(0.0, 0.0, 0.0),
    rotation=(0.0, math.pi / 2.0, 0.0)
)
```