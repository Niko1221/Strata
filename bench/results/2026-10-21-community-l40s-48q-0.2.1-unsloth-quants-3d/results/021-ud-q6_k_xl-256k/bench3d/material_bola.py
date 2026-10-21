import bpy

# Criar a esfera
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, 0.0))
bola = bpy.context.active_object
bola.name = "Bola"

# Criar o material
material = bpy.data.materials.new(name="Vermelho")
material.use_nodes = True

# Configurar o nó Principled BSDF
nodes = material.node_tree.nodes
links = material.node_tree.links
nodes.clear()

output_node = nodes.new("ShaderNodeOutputMaterial")
output_node.location = (300, 0)

bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
bsdf_node.location = (0, 0)

links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

bsdf_node.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
bsdf_node.inputs["Metallic"].default_value = 1.0
bsdf_node.inputs["Roughness"].default_value = 0.2

# Atribuir o material à esfera
bola.data.materials.clear()
bola.data.materials.append(material)
