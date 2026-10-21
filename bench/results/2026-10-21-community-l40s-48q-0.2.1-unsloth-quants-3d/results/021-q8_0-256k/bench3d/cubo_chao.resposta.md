```python
import bpy

bpy.ops.mesh.primitive_cube_add(size=2, location=(0.0, 0.0, 1.0))
cube = bpy.context.active_object
cube.name = "Cube"
```