```python
import bpy

bpy.ops.mesh.primitive_cube_add(size=1.0)
cube = bpy.context.active_object

array_mod = cube.modifiers.new(name="Array", type='ARRAY')
array_mod.count = 5
array_mod.use_relative_offset = True
array_mod.relative_offset_displace = (1.5, 0.0, 0.0)
array_mod.use_constant_offset = False
```