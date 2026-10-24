import bpy
import math
import bmesh

NUM_CILINDROS = 8
RAIO_CILINDRO = 0.2
ALTURA_CILINDRO = 1.0
RAIO_CIRCUNFERENCIA = 3.0
SEGMENTOS = 32

for i in range(NUM_CILINDROS):
    angulo = math.tau * i / NUM_CILINDROS
    x = RAIO_CIRCUNFERENCIA * math.cos(angulo)
    y = RAIO_CIRCUNFERENCIA * math.sin(angulo)

    mesh = bpy.data.meshes.new(f"Cilindro_{i + 1}")
    bm = bmesh.new()

    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=SEGMENTOS,
        radius1=RAIO_CILINDRO,
        radius2=RAIO_CILINDRO,
        depth=ALTURA_CILINDRO,
    )

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(f"Cilindro_{i + 1}", mesh)
    bpy.context.scene.collection.objects.link(obj)

    # Origem no centro geométrico do cilindro; base apoiada em z = 0.
    obj.location = (x, y, ALTURA_CILINDRO / 2.0)
