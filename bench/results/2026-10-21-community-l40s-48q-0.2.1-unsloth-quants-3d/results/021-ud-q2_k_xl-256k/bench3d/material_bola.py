import bpy
import bmesh

# Remove objeto anterior com o mesmo nome, se existir
old_obj = bpy.data.objects.get('Bola')
if old_obj is not None:
    old_mesh = old_obj.data
    bpy.data.objects.remove(old_obj, do_unlink=True)
    if old_mesh is not None and isinstance(old_mesh, bpy.types.Mesh) and old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)

# Criar malha da esfera
mesh = bpy.data.meshes.new('Bola')
bm = bmesh.new()
bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
bm.to_mesh(mesh)
bm.free()

# Criar objeto da esfera
obj = bpy.data.objects.new('Bola', mesh)
bpy.context.collection.objects.link(obj)

# Criar ou reutilizar o material
mat = bpy.data.materials.get('Vermelho')
if mat is None:
    mat = bpy.data.materials.new('Vermelho')

mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

# Encontrar ou criar o nó Principled BSDF
bsdf = None
for node in nodes:
    if node.type == 'BSDF_PRINCIPLED':
        bsdf = node
        break

if bsdf is None:
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')

    output = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL':
            output = node
            break

    if output is None:
        output = nodes.new('ShaderNodeOutputMaterial')

    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

# Definir propriedades do material
def set_socket(node, identifier, value):
    socket = node.inputs.get(identifier)
    if socket is None:
        for inp in node.inputs:
            if inp.identifier == identifier:
                socket = inp
                break

    if socket is not None:
        if isinstance(value, tuple):
            socket.default_value = value
        else:
            socket.default_value = value

set_socket(bsdf, 'Base Color', (1.0, 0.0, 0.0, 1.0))
set_socket(bsdf, 'Metallic', 1.0)
set_socket(bsdf, 'Roughness', 0.2)

# Atribuir material ao objeto
obj.data.materials.append(mat)
