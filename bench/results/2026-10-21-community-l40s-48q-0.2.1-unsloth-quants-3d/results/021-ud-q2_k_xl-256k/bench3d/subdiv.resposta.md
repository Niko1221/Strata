```python
import bpy
import bmesh

# Criar a malha da esfera UV
mesh = bpy.data.meshes.new("UVSphere")
bm = bmesh.new()
bmesh.ops.create_uvsphere(
    bm,
    u_segments=32,
    v_segments=16,
    radius=1.0
)
bm.to_mesh(mesh)
bm.free()

# Criar o objeto e adicionar à cena
obj = bpy.data.objects.new("UVSphere", mesh)
bpy.context.collection.objects.link(obj)

# Sombreamento suave
for poly in mesh.polygons:
    poly.use_smooth = True

# Modificador Subdivision Surface
subsurf = obj.modifiers.new(name="Subdivision Surface", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 2
```