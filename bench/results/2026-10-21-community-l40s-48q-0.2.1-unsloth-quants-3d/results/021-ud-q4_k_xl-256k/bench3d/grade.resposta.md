```python
import bpy

cube_size = 0.5
spacing = 1.0
cols = 3
rows = 4

x_positions = [(i - (cols - 1) / 2) * spacing for i in range(cols)]
y_positions = [(j - (rows - 1) / 2) * spacing for j in range(rows)]
z_center = cube_size / 2

mesh = bpy.data.meshes.new("CubeMesh")
h = cube_size / 2

vertices = [
    (-h, -h, -h),
    ( h, -h, -h),
    ( h,  h, -h),
    (-h,  h, -h),
    (-h, -h,  h),
    ( h, -h,  h),
    ( h,  h,  h),
    (-h,  h,  h),
]

faces = [
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]

mesh.from_pydata(vertices, [], faces)
mesh.update()

collection = bpy.context.scene.collection

for i, x in enumerate(x_positions):
    for j, y in enumerate(y_positions):
        obj = bpy.data.objects.new(f"Cube_{i}_{j}", mesh)
        obj.location = (x, y, z_center)
        collection.objects.link(obj)
```