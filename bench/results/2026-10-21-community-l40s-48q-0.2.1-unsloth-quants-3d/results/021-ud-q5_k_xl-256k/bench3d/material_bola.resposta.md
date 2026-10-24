```python
import bpy

# Create the sphere
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, 0.0))
bola = bpy.context.active_object
bola.name = "Bola"
bola.data.name = "Bola"

# Create the material
mat = bpy.data.materials.new("Vermelho")
mat.use_nodes = True

nt = mat.node_tree
bsdf = nt.nodes.get("Principled BSDF")

if bsdf is None:
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    output = nt.nodes.get("Material Output")
    if output is None:
        output = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
bsdf.inputs["Metallic"].default_value = 1.0
bsdf.inputs["Roughness"].default_value = 0.2

bola.data.materials.clear()
bola.data.materials.append(mat)
```