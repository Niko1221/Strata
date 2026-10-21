```python
import bpy
import bmesh

bm = bmesh.new()
bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)

mesh = bpy.data.meshes.new("Bola")
bm.to_mesh(mesh)
bm.free()

obj = bpy.data.objects.new("Bola", mesh)
bpy.context.scene.collection.objects.link(obj)

mat = bpy.data.materials.get("Vermelho")
if mat is None:
    mat = bpy.data.materials.new("Vermelho")

mat.use_nodes = True
nt = mat.node_tree

bsdf = nt.nodes.get("Principled BSDF")
if bsdf is None:
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = nt.nodes.get("Material Output")
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
bsdf.inputs["Metallic"].default_value = 1.0
bsdf.inputs["Roughness"].default_value = 0.2

obj.data.materials.clear()
obj.data.materials.append(mat)
```