```python
import bpy

# Cria a malha do cubo com dimensões de 1 m
mesh = bpy.data.meshes.new("Cube")

verts = [
    (-0.5, -0.5, -0.5),
    ( 0.5, -0.5, -0.5),
    ( 0.5,  0.5, -0.5),
    (-0.5,  0.5, -0.5),
    (-0.5, -0.5,  0.5),
    ( 0.5, -0.5,  0.5),
    ( 0.5,  0.5,  0.5),
    (-0.5,  0.5,  0.5),
]

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

faces = [
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
]

mesh.from_pydata(verts, edges, faces)
mesh.update()

# Cria o único objeto
obj = bpy.data.objects.new("Cube", mesh)
bpy.context.scene.collection.objects.link(obj)

# Adiciona o modificador Array sem aplicá-lo
array_mod = obj.modifiers.new("Array", 'ARRAY')
array_mod.count = 5
array_mod.use_relative_offset = True
array_mod.relative_offset_displace = (1.5, 0.0, 0.0)
```