import bpy
import bmesh

# Criar a malha da esfera
mesh = bpy.data.meshes.new("Bola")
bm = bmesh.new()
bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
bm.to_mesh(mesh)
bm.free()

# Criar o objeto da esfera
obj = bpy.data.objects.new("Bola", mesh)
bpy.context.scene.collection.objects.link(obj)

# Criar ou reutilizar o material
mat = bpy.data.materials.get("Vermelho")
if mat is None:
    mat = bpy.data.materials.new("Vermelho")

mat.use_nodes = True
nt = mat.node_tree

# Obter ou criar o nó Principled BSDF
bsdf = next((node for node in nt.nodes if node.type == "BSDF_PRINCIPLED"), None)
output = next((node for node in nt.nodes if node.type == "OUTPUT_MATERIAL"), None)

if bsdf is None:
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
if output is None:
    output = nt.nodes.new("ShaderNodeOutputMaterial")

# Garantir que o BSDF esteja conectado à saída do material
if not any(link.from_node == bsdf and link.to_socket == output.inputs["Surface"] for link in nt.links):
    nt.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

# Configurar o material
bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
bsdf.inputs["Metallic"].default_value = 1.0
bsdf.inputs["Roughness"].default_value = 0.2

# Atribuir o material ao objeto
obj.data.materials.clear()
obj.data.materials.append(mat)
