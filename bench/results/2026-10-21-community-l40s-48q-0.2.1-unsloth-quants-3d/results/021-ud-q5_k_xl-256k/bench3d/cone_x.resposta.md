```python
import bpy
import math

# Cria um cone com raio da base 0.5 m e altura 2 m.
# O cone é criado centrado na origem e depois ajustado para que
# o centro da base fique exatamente na origem.
bpy.ops.mesh.primitive_cone_add(
    radius1=0.5,
    radius2=0.0,
    depth=2.0,
    location=(0.0, 0.0, 0.0),
    rotation=(0.0, math.pi / 2.0, 0.0),
)

cone = bpy.context.object
if cone is None:
    cone = bpy.data.objects[-1]

# Move a malha localmente para que a base fique em z = 0
# e a ponta em z = 2. Com a rotação, isso aponta para +X.
for vertex in cone.data.vertices:
    vertex.co.z += 1.0

cone.data.update()
```