```python
import bpy

# Cria a malha original somente no lado x > 0
mesh = bpy.data.meshes.new("Metade")

# Cubo de 1 m centrado em x = 1, y = 0, z = 0
verts = [
    (0.5, -0.5, -0.5),
    (1.5, -0.5, -0.5),
    (1.5,  0.5, -0.5),
    (0.5,  0.5, -0.5),
    (0.5, -0.5,  0.5),
    (1.5, -0.5,  0.5),
    (1.5,  0.5,  0.5),
    (0.5,  0.5,  0.5),
]

faces = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 4, 7, 3),
    (1, 5, 6, 2),
    (3, 7, 6, 2),
    (0, 1, 5, 4),
]

mesh.from_pydata(verts, [], faces)
mesh.update()

# Cria o objeto com origem em x = 0 para que o Mirror espelhe em relação ao plano x = 0
obj = bpy.data.objects.new("Metade", mesh)
obj.location = (0.0, 0.0, 0.0)
bpy.context.scene.collection.objects.link(obj)

# Adiciona o modificador Mirror no eixo X, sem aplicá-lo
mirror = obj.modifiers.new(name="Mirror", type='MIRROR')
mirror.use_axis[0] = True
mirror.use_axis[1] = False
mirror.use_axis[2] = False
```