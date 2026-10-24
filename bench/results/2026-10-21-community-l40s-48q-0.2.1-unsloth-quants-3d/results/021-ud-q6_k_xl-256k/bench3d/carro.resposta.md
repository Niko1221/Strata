```python
import bpy
import bmesh
import math

scene = bpy.context.scene
collection = scene.collection

# Empty 'Carro' na origem
carro = bpy.data.objects.new('Carro', None)
collection.objects.link(carro)
carro.location = (0.0, 0.0, 0.0)

bpy.context.view_layer.update()

# Posições das rodas: (x, y)
posicoes = [
    (1.2, 0.8),
    (1.2, -0.8),
    (-1.2, 0.8),
    (-1.2, -0.8),
]

for i, (x, y) in enumerate(posicoes, start=1):
    # Malha do cilindro (eixo local Z)
    mesh = bpy.data.meshes.new(f'Roda_{i}')
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=32,
        radius1=0.3,
        radius2=0.3,
        depth=0.2,
    )
    bm.to_mesh(mesh)
    bm.free()

    # Objeto da roda
    roda = bpy.data.objects.new(f'Roda_{i}', mesh)
    collection.objects.link(roda)

    # Centro da roda
    roda.location = (x, y, 0.3)

    # Eixo do cilindro paralelo ao eixo Y
    roda.rotation_euler = (math.pi / 2.0, 0.0, 0.0)

    # Filha do Empty 'Carro'
    roda.parent = carro
    roda.matrix_parent_inverse = carro.matrix_world.inverted()

bpy.context.view_layer.update()
```