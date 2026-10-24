import bpy
import bmesh
import math

scene = bpy.context.scene

# Empty pai
carro = bpy.data.objects.new("Carro", None)
scene.collection.objects.link(carro)

def criar_roda(nome, localizacao):
    # Malha do cilindro (eixo padrão em Z)
    malha = bpy.data.meshes.new(nome)
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=32,
        radius1=0.3,
        radius2=0.3,
        depth=0.2
    )
    bm.to_mesh(malha)
    bm.free()

    # Objeto da roda
    roda = bpy.data.objects.new(nome, malha)
    scene.collection.objects.link(roda)

    # Posição e rotação: eixo do cilindro paralelo ao eixo Y
    roda.location = localizacao
    roda.rotation_euler = (math.pi / 2.0, 0.0, 0.0)

    # Filha do Empty
    roda.parent = carro
    roda.matrix_parent_inverse = carro.matrix_world.inverted()

    return roda

# Quatro rodas nas posições (±1.2, ±0.8, 0.3)
for x in (-1.2, 1.2):
    for y in (-0.8, 0.8):
        nome = f"Roda_{x}_{y}"
        criar_roda(nome, (x, y, 0.3))

bpy.context.view_layer.update()
