```python
import bpy

cols = 3
rows = 4
spacing = 1.0
cube_size = 0.5
z_center = cube_size / 2.0

for i in range(cols):
    x = (i - (cols - 1) / 2.0) * spacing
    for j in range(rows):
        y = (j - (rows - 1) / 2.0) * spacing
        bpy.ops.mesh.primitive_cube_add(size=cube_size, location=(x, y, z_center))
```