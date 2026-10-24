import bpy
import bmesh

# Criar a malha da esfera
mesh = bpy.data.meshes.new("Bola")
bm = bmesh.new()
bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
bm.to_mesh(mesh)
bm.free()

# Criar o objeto da esfera
bola = bpy.data.objects.new("Bola", mesh)
bpy.context.scene.collection.objects.link(bola)

# Criar o material vermelho
material = bpy.data.materials.new("Vermelho")
material.use_nodes = True
material.diffuse_color = (1.0, 0.0, 0.0, 1.0)

if hasattr(material, "metallic"):
    material.metallic = 1.0
if hasattr(material, "roughness"):
    material.roughness = 0.2

# Configurar o nó Principled BSDF
nodes = material.node_tree.nodes
links = material.node_tree.links

bsdf = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
if bsdf is None:
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
if output is None:
    output = nodes.new("ShaderNodeOutputMaterial")

bsdf_output = bsdf.outputs.get("BSDF") or bsdf.outputs[0]
surface_input = output.inputs.get("Surface") or output.inputs[0]

if not surface_input.links:
    links.new(bsdf_output, surface_input)

def set_input(node, name, value):
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value

set_input(bsdf, "Base Color", (1.0, 0.0, 0.0, 1.0))
set_input(bsdf, "Metallic", 1.0)
set_input(bsdf, "Roughness", 0.2)

# Atribuir o material à esfera
bola.data.materials.append(material)
