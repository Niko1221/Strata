"""tarefas3d.py - as tarefas da bateria 3D (Blender 5.2, bpy, sem tela). Importado dentro do Blender por executor3d.py.

Cada tarefa: id, categoria, pedido (o texto que o modelo recebe), checar() (roda depois do script do modelo, devolve
(passou, detalhe)) e REF[id] (uma solucao de referencia, usada so para validar os verificadores: bench3d.py validar).
Unidades em metros; o chao e o plano XY (z=0); +Z para cima. Tolerancia padrao 2 cm / 1 grau.
"""
import math

try:
    import bpy
    from mathutils import Vector
except ImportError:          # fora do Blender (bench3d.py so le os pedidos)
    bpy = None

TOL = 0.02


def _dg():
    bpy.context.view_layer.update()
    return bpy.context.evaluated_depsgraph_get()


def malhas():
    return [o for o in bpy.context.scene.objects if o.type == 'MESH']


def bb(o):
    """caixa envolvente no mundo da geometria avaliada (com modificadores e restricoes): (min, max)"""
    dg = _dg(); oe = o.evaluated_get(dg); me = oe.to_mesh()
    try:
        pts = [oe.matrix_world @ v.co for v in me.vertices]
    finally:
        oe.to_mesh_clear()
    if not pts:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    return (Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))),
            Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts))))


def tam(o):
    a, b = bb(o); return b - a


def centro(o):
    a, b = bb(o); return (a + b) / 2


def perto(a, b, t=TOL):
    return abs(a - b) <= t


def vperto(a, b, t=TOL):
    return all(abs(x - y) <= t for x, y in zip(a, b))


def obj(nome):
    o = bpy.data.objects.get(nome)
    if o is None:
        for x in bpy.data.objects:
            if x.name.lower().startswith(nome.lower()):
                return x
    return o


def mundo(o):
    return o.evaluated_get(_dg()).matrix_world


def direcao(o):            # para onde aponta o -Z local (cameras e luzes)
    return (mundo(o).to_quaternion() @ Vector((0, 0, -1))).normalized()


def ang(u, v):
    return math.degrees(u.angle(v))


def fmt(v):
    return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"


def r(ok, msg):
    return bool(ok), msg


def n_malhas(n):
    m = malhas()
    return len(m) == n, f"{len(m)} malhas (esperado {n})"


# ---------------------------------------------------------------- verificadores
def c_cubo_chao():
    ok, d = n_malhas(1)
    if not ok: return r(False, d)
    o = malhas()[0]; a, b = bb(o)
    return r(vperto(tam(o), (2, 2, 2)) and perto(a.z, 0) and vperto(centro(o)[:2], (0, 0)), f"tam {fmt(tam(o))} min.z {a.z:.3f} centro {fmt(centro(o))}")


def c_esferas_fila():
    ok, d = n_malhas(3)
    if not ok: return r(False, d)
    m = sorted(malhas(), key=lambda o: centro(o).x)
    oks = [vperto(tam(o), (1, 1, 1), 0.05) and perto(bb(o)[0].z, 0, 0.03) and perto(centro(o).y, 0) and perto(centro(o).x, x)
           for o, x in zip(m, (-2, 0, 2))]
    return r(all(oks), "centros " + " ".join(fmt(centro(o)) for o in m))


def c_pilha_girada():
    ok, d = n_malhas(3)
    if not ok: return r(False, d)
    m = sorted(malhas(), key=lambda o: centro(o).z); oks = []
    for i, (o, rot) in enumerate(zip(m, (0, 15, 30))):
        a, b = bb(o); yaw = math.degrees(mundo(o).to_euler().z) % 90
        oks.append(perto(a.z, i) and perto(b.z, i + 1) and vperto(centro(o)[:2], (0, 0)) and min(abs(yaw - rot), 90 - abs(yaw - rot)) <= 1)
    return r(all(oks), "z/rot " + " ".join(f"{bb(o)[0].z:.2f}/{math.degrees(mundo(o).to_euler().z) % 90:.1f}" for o in m))


def c_anel_cilindros():
    ok, d = n_malhas(8)
    if not ok: return r(False, d)
    angs, oks = [], []
    for o in malhas():
        c = centro(o); oks.append(vperto(tam(o), (0.4, 0.4, 1), 0.03) and perto(bb(o)[0].z, 0) and perto(math.hypot(c.x, c.y), 3, 0.03))
        angs.append(math.degrees(math.atan2(c.y, c.x)) % 360)
    angs.sort(); difs = [(angs[(i + 1) % 8] - angs[i]) % 360 for i in range(8)]
    return r(all(oks) and all(abs(x - 45) <= 1.5 for x in difs), f"forma/raio ok {sum(oks)}/8; passos {[round(x, 1) for x in difs]}")


def c_mesa():
    ok, d = n_malhas(5)
    if not ok: return r(False, d)
    m = malhas(); topo = max(m, key=lambda o: tam(o).x * tam(o).y); pernas = [o for o in m if o is not topo]
    a, b = bb(topo)
    ok_t = vperto(tam(topo), (2, 1, 0.05)) and perto(b.z, 0.75) and vperto(centro(topo)[:2], (0, 0))
    cantos, oks = set(), []
    for p in pernas:
        pa, pb = bb(p); t = tam(p)
        sx = -1 if perto(pa.x, -1) else (1 if perto(pb.x, 1) else 0); sy = -1 if perto(pa.y, -0.5) else (1 if perto(pb.y, 0.5) else 0)
        cantos.add((sx, sy)); oks.append(bool(vperto(t, (0.06, 0.06, 0.70)) and perto(pa.z, 0) and perto(pb.z, 0.70) and sx and sy))
    return r(ok_t and all(oks) and len(cantos) == 4, f"tampo {fmt(tam(topo))} topo {b.z:.3f}; pernas ok {sum(oks)}/4; cantos {len(cantos)}")


def c_esfera_sobre_cubo():
    ok, d = n_malhas(2)
    if not ok: return r(False, d)
    m = sorted(malhas(), key=lambda o: centro(o).z); cubo, esf = m
    return r(vperto(tam(cubo), (2, 2, 2)) and perto(bb(cubo)[0].z, 0) and vperto(tam(esf), (2, 2, 2), 0.05) and perto(bb(esf)[0].z, 2, 0.03)
             and vperto(centro(esf)[:2], (0, 0)) and vperto(centro(cubo)[:2], (0, 0)), f"cubo {fmt(bb(cubo)[0])}..{fmt(bb(cubo)[1])} esfera min.z {bb(esf)[0].z:.3f}")


def c_grade():
    ok, d = n_malhas(12)
    if not ok: return r(False, d)
    alvo = {(x, y) for x in (-1, 0, 1) for y in (-1.5, -0.5, 0.5, 1.5)}; achados = set()
    for o in malhas():
        c = centro(o)
        if vperto(tam(o), (0.5, 0.5, 0.5)) and perto(bb(o)[0].z, 0):
            for p in alvo:
                if vperto(c[:2], p): achados.add(p)
    return r(achados == alvo, f"{len(achados)}/12 posicoes certas")


def c_cilindro_deitado():
    ok, d = n_malhas(1)
    if not ok: return r(False, d)
    o = malhas()[0]
    return r(vperto(tam(o), (4, 0.5, 0.5)) and vperto(centro(o), (0, 0, 0)), f"tam {fmt(tam(o))} centro {fmt(centro(o))}")


def c_escada():
    ok, d = n_malhas(10)
    if not ok: return r(False, d)
    m = sorted(malhas(), key=lambda o: centro(o).z); oks = []
    for n, o in enumerate(m, 1):
        a, b = bb(o)
        oks.append(vperto(tam(o), (1, 0.3, 0.18)) and perto(b.z, n * 0.18) and perto(a.y, (n - 1) * 0.3) and perto(b.y, n * 0.3) and perto(centro(o).x, 0))
    return r(all(oks), f"degraus certos {sum(oks)}/10")


def c_sala():
    ok, d = n_malhas(5)
    if not ok: return r(False, d)
    m = malhas(); piso = max(m, key=lambda o: tam(o).x * tam(o).y)
    ok_p = vperto(tam(piso), (4, 5, 0.1)) and perto(bb(piso)[1].z, 0) and vperto(centro(piso)[:2], (0, 0))
    px, py = [], []
    for w in (o for o in m if o is not piso):
        a, b = bb(w); t = tam(w)
        if not (perto(a.z, 0) and perto(b.z, 2.5)): continue
        if perto(t.x, 0.1) and 4.79 <= t.y <= 5.01 and (perto(a.x, -2) or perto(b.x, 2)): px.append(round(a.x))
        if perto(t.y, 0.1) and 3.79 <= t.x <= 4.01 and (perto(a.y, -2.5) or perto(b.y, 2.5)): py.append(round(b.y))
    return r(ok_p and len(set(px)) == 2 and len(set(py)) == 2, f"piso {fmt(tam(piso))}; paredes X {len(px)} Y {len(py)}")


def c_camera_mira():
    c = bpy.context.scene.camera
    if c is None: return r(False, "sem camera ativa")
    loc = mundo(c).translation; alvo = (Vector((0, 0, 0)) - loc).normalized(); a = ang(direcao(c), alvo)
    return r(vperto(loc, (10, -10, 8)) and a <= 1, f"camera em {fmt(loc)}, desvio {a:.2f} graus")


def c_luzes():
    sol = [o for o in bpy.context.scene.objects if o.type == 'LIGHT' and o.data.type == 'SUN']
    area = [o for o in bpy.context.scene.objects if o.type == 'LIGHT' and o.data.type == 'AREA']
    if len(sol) != 1 or len(area) != 1: return r(False, f"{len(sol)} sol, {len(area)} area")
    s, a = sol[0], area[0]; baixo = Vector((0, 0, -1)); ad = a.data
    lado = (ad.size, ad.size) if ad.shape in ('SQUARE', 'DISK') else (ad.size, ad.size_y)
    return r(perto(s.data.energy, 3, 0.01) and ang(direcao(s), baixo) <= 1 and vperto(mundo(a).translation, (0, 0, 4)) and perto(ad.energy, 500, 0.5)
             and vperto(lado, (2, 2)) and ang(direcao(a), baixo) <= 1,
             f"sol {s.data.energy} desvio {ang(direcao(s), baixo):.1f}; area {fmt(mundo(a).translation)} {ad.energy}W {lado}")


def _principled(m):
    if m is None or not m.node_tree: return None
    for n in m.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED': return n


def c_material_bola():
    o = bpy.data.objects.get("Bola")
    if o is None or o.type != 'MESH': return r(False, "sem objeto 'Bola'")
    mats = [s.material for s in o.material_slots if s.material]; m = next((x for x in mats if x.name.startswith("Vermelho")), None)
    p = _principled(m)
    if p is None: return r(False, f"materiais {[x.name for x in mats]} sem Principled 'Vermelho'")
    cor = p.inputs["Base Color"].default_value
    return r(vperto(cor[:3], (1, 0, 0), 0.01) and perto(p.inputs["Metallic"].default_value, 1, 0.01) and perto(p.inputs["Roughness"].default_value, 0.2, 0.01),
             f"cor {fmt(cor[:3])} metal {p.inputs['Metallic'].default_value:.2f} rug {p.inputs['Roughness'].default_value:.2f}")


def c_hierarquia():
    p, s = bpy.data.objects.get("Pivo"), bpy.data.objects.get("Satelite")
    if p is None or s is None: return r(False, "faltam 'Pivo' ou 'Satelite'")
    return r(p.type == 'EMPTY' and s.parent == p and vperto(mundo(s).translation, (0, 2, 1)) and vperto(mundo(p).translation, (0, 0, 1))
             and vperto(tam(s), (0.5, 0.5, 0.5)), f"pai {s.parent.name if s.parent else None}; satelite em {fmt(mundo(s).translation)}")


def c_array():
    ok, d = n_malhas(1)
    if not ok: return r(False, d)
    o = malhas()[0]; mods = [m for m in o.modifiers if m.type == 'ARRAY']
    if len(mods) != 1: return r(False, f"{len(mods)} modificadores Array")
    a = mods[0]
    return r(a.count == 5 and a.use_relative_offset and perto(a.relative_offset_displace[0], 1.5, 0.001) and perto(tam(o).x, 7, 0.03),
             f"count {a.count} rel {tuple(round(x, 3) for x in a.relative_offset_displace)} comprimento {tam(o).x:.3f}")


def c_espelho():
    o = bpy.data.objects.get("Metade")
    if o is None or o.type != 'MESH': return r(False, "sem objeto 'Metade'")
    mods = [m for m in o.modifiers if m.type == 'MIRROR']
    if not mods or not mods[0].use_axis[0]: return r(False, "sem Mirror em X")
    xs = [(o.matrix_world @ v.co).x for v in o.data.vertices]; a, b = bb(o)
    return r(min(xs) >= -0.001 and perto(a.x, -b.x) and b.x > 0.1, f"x original {min(xs):.3f}..{max(xs):.3f}; avaliado {a.x:.3f}..{b.x:.3f}")


def c_boolean_furo():
    ok, d = n_malhas(1)
    if not ok: return r(False, d)
    o = malhas()[0]; dg = _dg(); sc = bpy.context.scene
    meio = sc.ray_cast(dg, Vector((0, 0, 5)), Vector((0, 0, -1)))
    lado = sc.ray_cast(dg, Vector((0.8, 0, 5)), Vector((0, 0, -1)))
    furo = not meio[0] or meio[1].z < -1.01
    return r(vperto(tam(o), (2, 2, 2)) and furo and lado[0] and perto(lado[1].z, 1), f"tam {fmt(tam(o))}; raio no centro acerta: {meio[0]}; ao lado: z={lado[1].z:.3f}")


def c_animacao():
    o = bpy.data.objects.get("Movel"); sc = bpy.context.scene
    if o is None: return r(False, "sem 'Movel'")
    pos = {}
    for f in (1, 25, 50):
        sc.frame_set(f); pos[f] = mundo(o).translation.copy()
    sc.frame_set(1)
    return r(sc.frame_start == 1 and sc.frame_end == 50 and vperto(pos[1], (0, 0, 0)) and vperto(pos[50], (5, 0, 0)) and 0.1 < pos[25].x < 4.9,
             f"quadros {sc.frame_start}-{sc.frame_end}; 1 {fmt(pos[1])} 25 {fmt(pos[25])} 50 {fmt(pos[50])}")


def c_colecao():
    c = bpy.data.collections.get("Moveis")
    if c is None: return r(False, "sem colecao 'Moveis'")
    ligada = c.name in [x.name for x in bpy.context.scene.collection.children_recursive]
    nomes = sorted(o.name for o in c.objects)
    return r(ligada and nomes == ["Armario", "Cadeira", "Mesa"], f"ligada {ligada}; objetos {nomes}")


def c_escala_aplicada():
    ok, d = n_malhas(1)
    if not ok: return r(False, d)
    o = malhas()[0]
    return r(vperto(o.scale, (1, 1, 1), 0.001) and vperto(tam(o), (3, 2, 1)) and perto(bb(o)[0].z, 0), f"escala {fmt(o.scale)} tam {fmt(tam(o))} min.z {bb(o)[0].z:.3f}")


def c_carro():
    c = bpy.data.objects.get("Carro")
    if c is None or c.type != 'EMPTY': return r(False, "sem Empty 'Carro'")
    rodas = [o for o in c.children if o.type == 'MESH']
    if len(rodas) != 4: return r(False, f"{len(rodas)} rodas filhas")
    alvo = {(sx * 1.2, sy * 0.8) for sx in (-1, 1) for sy in (-1, 1)}; achados = set()
    for o in rodas:
        cc = centro(o)
        if vperto(tam(o), (0.6, 0.2, 0.6)) and perto(cc.z, 0.3):
            for p in alvo:
                if vperto(cc[:2], p): achados.add(p)
    return r(achados == alvo, f"rodas certas {len(achados)}/4; tamanhos {[fmt(tam(o)) for o in rodas]}")


def c_cone_x():
    ok, d = n_malhas(1)
    if not ok: return r(False, d)
    o = malhas()[0]; a, b = bb(o)
    vs = [o.matrix_world @ v.co for v in o.evaluated_get(_dg()).data.vertices]; ponta = max(vs, key=lambda p: p.x)
    return r(vperto(tam(o), (2, 1, 1)) and perto(a.x, 0) and perto(b.x, 2) and vperto(centro(o)[1:], (0, 0)) and vperto(ponta[1:], (0, 0)),
             f"x {a.x:.3f}..{b.x:.3f} tam {fmt(tam(o))} ponta {fmt(ponta)}")


def c_render():
    s = bpy.context.scene; rd = s.render
    return r(rd.engine == 'CYCLES' and s.cycles.samples == 64 and rd.resolution_x == 1920 and rd.resolution_y == 1080 and rd.resolution_percentage == 100
             and rd.image_settings.file_format == 'PNG', f"{rd.engine} {s.cycles.samples} {rd.resolution_x}x{rd.resolution_y} {rd.resolution_percentage}% {rd.image_settings.file_format}")


def c_cadeiras():
    ok, d = n_malhas(3)
    if not ok: return r(False, d)
    m = malhas(); mesa = max(m, key=lambda o: tam(o).x * tam(o).y); cad = sorted((o for o in m if o is not mesa), key=lambda o: centro(o).y)
    ok_m = vperto(tam(mesa), (1.6, 0.9, 0.75)) and perto(bb(mesa)[0].z, 0) and vperto(centro(mesa)[:2], (0, 0))
    ok_c = all(vperto(tam(o), (0.45, 0.45, 0.45)) and perto(bb(o)[0].z, 0) and perto(centro(o).x, 0) for o in cad)
    return r(ok_m and ok_c and perto(bb(cad[0])[1].y, -0.65) and perto(bb(cad[1])[0].y, 0.65),
             f"mesa {fmt(tam(mesa))}; cadeiras y {bb(cad[0])[1].y:.3f} e {bb(cad[1])[0].y:.3f}")


def c_texto():
    t = [o for o in bpy.context.scene.objects if o.type == 'FONT']
    if len(t) != 1: return r(False, f"{len(t)} objetos de texto")
    d = t[0].data
    return r(d.body == "Olá 3D" and perto(d.extrude, 0.1, 0.001) and d.align_x == 'CENTER', f"'{d.body}' extrude {d.extrude} align {d.align_x}")


def c_subdiv():
    ok, d = n_malhas(1)
    if not ok: return r(False, d)
    o = malhas()[0]; mods = [m for m in o.modifiers if m.type == 'SUBSURF']
    suave = all(p.use_smooth for p in o.data.polygons)
    return r(bool(mods) and mods[0].levels == 2 and mods[0].render_levels == 2 and suave and len(o.data.polygons) > 100,
             f"subsurf {[(m.levels, m.render_levels) for m in mods]} suave {suave} faces {len(o.data.polygons)}")


TAREFAS = [
    ("cubo_chao", "medidas", "Crie um cubo de 2 m de aresta apoiado no chão (a face de baixo em z = 0), centrado em x = 0, y = 0.", c_cubo_chao),
    ("esferas_fila", "medidas", "Crie três esferas de raio 0,5 m alinhadas no eixo X, com os centros em x = −2, 0 e 2 e y = 0, todas apoiadas no chão.", c_esferas_fila),
    ("pilha_girada", "relacoes", "Empilhe três cubos de 1 m: o primeiro apoiado no chão e cada um apoiado sobre o anterior, todos centrados em x = 0, y = 0. O primeiro fica sem rotação; gire o segundo 15° e o terceiro 30° em torno do eixo Z.", c_pilha_girada),
    ("anel_cilindros", "relacoes", "Crie 8 cilindros de raio 0,2 m e altura 1 m, em pé e apoiados no chão, com os centros distribuídos igualmente sobre uma circunferência de raio 3 m centrada na origem.", c_anel_cilindros),
    ("mesa", "relacoes", "Modele uma mesa com 5 objetos: um tampo de 2,0 m (X) × 1,0 m (Y) × 0,05 m de espessura, com a face de cima a 0,75 m do chão, e quatro pernas de seção quadrada 0,06 m × 0,06 m que vão do chão até a face de baixo do tampo. Cada perna fica num canto, com as duas faces externas alinhadas às bordas do tampo. A mesa fica centrada na origem.", c_mesa),
    ("esfera_sobre_cubo", "relacoes", "Crie um cubo de 2 m apoiado no chão e uma esfera de raio 1 m apoiada exatamente no topo do cubo, os dois centrados em x = 0, y = 0.", c_esfera_sobre_cubo),
    ("grade", "relacoes", "Crie uma grade de 3 colunas (ao longo de X) por 4 linhas (ao longo de Y) de cubos de 0,5 m, com 1 m entre os centros vizinhos, a grade centrada na origem e todos os cubos apoiados no chão.", c_grade),
    ("cilindro_deitado", "medidas", "Crie um cilindro de raio 0,25 m e 4 m de comprimento, deitado ao longo do eixo X e centrado na origem.", c_cilindro_deitado),
    ("escada", "relacoes", "Modele uma escada de 10 degraus que sobe na direção +Y. Cada degrau é uma caixa de 1,0 m (X) × 0,30 m (Y) × 0,18 m (Z), centrada em x = 0. O degrau n (de 1 a 10) tem a face de cima a n × 0,18 m de altura e ocupa y de (n − 1) × 0,30 m até n × 0,30 m.", c_escada),
    ("sala", "relacoes", "Modele uma sala centrada na origem: um piso de 4 m (X) × 5 m (Y) × 0,1 m de espessura com a face de cima em z = 0, e quatro paredes de 0,1 m de espessura e 2,5 m de altura, de z = 0 a z = 2,5, com as faces externas alinhadas às bordas do piso.", c_sala),
    ("camera_mira", "camera_luz", "Adicione uma câmera na posição (10, −10, 8) apontando exatamente para a origem e defina-a como a câmera ativa da cena.", c_camera_mira),
    ("luzes", "camera_luz", "Adicione uma luz do tipo Sol com força 3 apontando diretamente para baixo (−Z), e uma luz de área quadrada de 2 m × 2 m na posição (0, 0, 4), também apontando para baixo, com potência de 500 W.", c_luzes),
    ("material_bola", "material_cena", "Crie uma esfera chamada 'Bola' com um material chamado 'Vermelho': cor base vermelha pura (1, 0, 0), metálico 1,0 e rugosidade 0,2.", c_material_bola),
    ("hierarquia", "hierarquia_mod", "Crie um Empty chamado 'Pivo' na posição (0, 0, 1) e um cubo de 0,5 m chamado 'Satelite', filho do Pivo, deslocado 2 m em +X em relação a ele. Depois gire o Pivo 90° em torno de Z, de modo que o Satelite termine em (0, 2, 1) no mundo.", c_hierarquia),
    ("array", "hierarquia_mod", "Crie um único objeto: um cubo de 1 m com um modificador Array (não aplicado) que gera 5 cópias ao longo de X, com deslocamento relativo de 1,5 entre elas.", c_array),
    ("espelho", "hierarquia_mod", "Crie um objeto de malha chamado 'Metade' cuja malha original fica só do lado x > 0 (por exemplo, um cubo de 1 m centrado em x = 1) e adicione um modificador Mirror no eixo X (não aplicado), de modo que o resultado fique simétrico em relação ao plano x = 0.", c_espelho),
    ("boolean_furo", "hierarquia_mod", "Faça um cubo de 2 m centrado na origem com um furo cilíndrico de raio 0,5 m que o atravessa de cima a baixo pelo centro (eixo Z). Aplique a operação booleana: no fim deve sobrar na cena só o objeto do cubo furado, sem o cilindro.", c_boolean_furo),
    ("animacao", "animacao", "Anime um cubo chamado 'Movel': no quadro 1 ele está em (0, 0, 0) e no quadro 50 em (5, 0, 0). Defina o intervalo de quadros da cena de 1 a 50.", c_animacao),
    ("colecao", "material_cena", "Crie uma coleção chamada 'Moveis', ligada à cena, contendo exatamente três objetos de malha chamados 'Mesa', 'Cadeira' e 'Armario' (podem ser cubos de tamanhos diferentes).", c_colecao),
    ("escala_aplicada", "medidas", "Crie uma caixa de 3 m (X) × 2 m (Y) × 1 m (Z) apoiada no chão, com a escala aplicada: a escala do objeto deve ficar (1, 1, 1).", c_escala_aplicada),
    ("carro", "hierarquia_mod", "Crie um Empty chamado 'Carro' na origem e quatro rodas, cilindros de raio 0,3 m e largura 0,2 m, filhas dele. O eixo de cada roda fica paralelo ao eixo Y, e os centros das rodas ficam em (±1,2; ±0,8; 0,3).", c_carro),
    ("cone_x", "medidas", "Crie um cone com raio da base de 0,5 m e altura de 2 m, com o centro da base na origem e a ponta apontando para +X.", c_cone_x),
    ("render", "material_cena", "Configure a renderização da cena: motor Cycles, 64 amostras, resolução 1920 × 1080 a 100% e saída em PNG.", c_render),
    ("cadeiras", "relacoes", "Crie uma mesa, uma caixa de 1,6 m (X) × 0,9 m (Y) × 0,75 m apoiada no chão e centrada na origem, e duas cadeiras, caixas de 0,45 m de lado apoiadas no chão, uma de cada lado da mesa no eixo Y, centradas em x = 0, com 0,2 m de folga entre a face da cadeira e a face da mesa.", c_cadeiras),
    ("texto", "material_cena", "Crie um objeto de texto com o conteúdo 'Olá 3D', extrusão de 0,1 m e alinhamento horizontal centralizado.", c_texto),
    ("subdiv", "material_cena", "Crie uma esfera UV com sombreamento suave e um modificador Subdivision Surface com nível 2 na viewport e nível 2 no render.", c_subdiv),
]

REF = {
    "cubo_chao": "bpy.ops.mesh.primitive_cube_add(size=2, location=(0,0,1))",
    "esferas_fila": "for x in (-2,0,2): bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(x,0,0.5))",
    "pilha_girada": "import math\nfor i,r in enumerate((0,15,30)): bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,i+0.5), rotation=(0,0,math.radians(r)))",
    "anel_cilindros": "import math\nfor i in range(8):\n a=math.radians(45*i); bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=1, location=(3*math.cos(a),3*math.sin(a),0.5))",
    "mesa": "bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,0.725)); bpy.context.object.dimensions=(2,1,0.05)\nfor sx in (-1,1):\n for sy in (-1,1):\n  bpy.ops.mesh.primitive_cube_add(size=1, location=(sx*0.97, sy*0.47, 0.35)); bpy.context.object.dimensions=(0.06,0.06,0.70)",
    "esfera_sobre_cubo": "bpy.ops.mesh.primitive_cube_add(size=2, location=(0,0,1)); bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0,0,3))",
    "grade": "for x in (-1,0,1):\n for y in (-1.5,-0.5,0.5,1.5): bpy.ops.mesh.primitive_cube_add(size=0.5, location=(x,y,0.25))",
    "cilindro_deitado": "import math\nbpy.ops.mesh.primitive_cylinder_add(radius=0.25, depth=4, rotation=(0,math.radians(90),0))",
    "escada": "for n in range(1,11): bpy.ops.mesh.primitive_cube_add(size=1, location=(0,(n-0.5)*0.3,n*0.18-0.09)); bpy.context.object.dimensions=(1,0.3,0.18)",
    "sala": "def caixa(l,d): bpy.ops.mesh.primitive_cube_add(size=1, location=l); bpy.context.object.dimensions=d\ncaixa((0,0,-0.05),(4,5,0.1)); caixa((-1.95,0,1.25),(0.1,5,2.5)); caixa((1.95,0,1.25),(0.1,5,2.5)); caixa((0,-2.45,1.25),(3.8,0.1,2.5)); caixa((0,2.45,1.25),(3.8,0.1,2.5))",
    "camera_mira": "from mathutils import Vector\nbpy.ops.object.camera_add(location=(10,-10,8)); c=bpy.context.object\nc.rotation_euler=(Vector((0,0,0))-c.location).to_track_quat('-Z','Y').to_euler(); bpy.context.scene.camera=c",
    "luzes": "bpy.ops.object.light_add(type='SUN', location=(0,0,10)); bpy.context.object.data.energy=3\nbpy.ops.object.light_add(type='AREA', location=(0,0,4)); l=bpy.context.object.data; l.shape='SQUARE'; l.size=2; l.energy=500",
    "material_bola": "bpy.ops.mesh.primitive_uv_sphere_add(); o=bpy.context.object; o.name='Bola'\nm=bpy.data.materials.new('Vermelho'); m.use_nodes=True; p=m.node_tree.nodes['Principled BSDF']\np.inputs['Base Color'].default_value=(1,0,0,1); p.inputs['Metallic'].default_value=1; p.inputs['Roughness'].default_value=0.2; o.data.materials.append(m)",
    "hierarquia": "import math\nbpy.ops.object.empty_add(location=(0,0,1)); p=bpy.context.object; p.name='Pivo'\nbpy.ops.mesh.primitive_cube_add(size=0.5); s=bpy.context.object; s.name='Satelite'; s.parent=p; s.location=(2,0,0)\np.rotation_euler[2]=math.radians(90)",
    "array": "bpy.ops.mesh.primitive_cube_add(size=1); m=bpy.context.object.modifiers.new('Array','ARRAY'); m.count=5; m.relative_offset_displace=(1.5,0,0)",
    "espelho": "bpy.ops.mesh.primitive_cube_add(size=1, location=(1,0,0)); o=bpy.context.object; o.name='Metade'; bpy.ops.object.transform_apply(location=True)\nm=o.modifiers.new('Mirror','MIRROR'); m.use_axis[0]=True",
    "boolean_furo": "bpy.ops.mesh.primitive_cube_add(size=2); c=bpy.context.object\nbpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=3); k=bpy.context.object\nm=c.modifiers.new('B','BOOLEAN'); m.operation='DIFFERENCE'; m.object=k\nbpy.context.view_layer.objects.active=c; bpy.ops.object.modifier_apply(modifier='B'); bpy.data.objects.remove(k)",
    "animacao": "bpy.ops.mesh.primitive_cube_add(); o=bpy.context.object; o.name='Movel'; s=bpy.context.scene; s.frame_start=1; s.frame_end=50\no.location=(0,0,0); o.keyframe_insert('location', frame=1); o.location=(5,0,0); o.keyframe_insert('location', frame=50)",
    "colecao": "c=bpy.data.collections.new('Moveis'); bpy.context.scene.collection.children.link(c)\nfor n,z in (('Mesa',0.5),('Cadeira',0.3),('Armario',1)):\n bpy.ops.mesh.primitive_cube_add(size=1); o=bpy.context.object; o.name=n\n for k in list(o.users_collection): k.objects.unlink(o)\n c.objects.link(o)",
    "escala_aplicada": "bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,0.5)); bpy.context.object.scale=(3,2,1); bpy.ops.object.transform_apply(scale=True)",
    "carro": "import math\nbpy.ops.object.empty_add(); c=bpy.context.object; c.name='Carro'\nfor sx in (-1,1):\n for sy in (-1,1):\n  bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.2, location=(sx*1.2,sy*0.8,0.3), rotation=(math.radians(90),0,0)); bpy.context.object.parent=c",
    "cone_x": "import math\nbpy.ops.mesh.primitive_cone_add(radius1=0.5, depth=2, location=(1,0,0), rotation=(0,math.radians(90),0))",
    "render": "s=bpy.context.scene; s.render.engine='CYCLES'; s.cycles.samples=64; s.render.resolution_x=1920; s.render.resolution_y=1080; s.render.resolution_percentage=100; s.render.image_settings.file_format='PNG'",
    "cadeiras": "def caixa(l,d): bpy.ops.mesh.primitive_cube_add(size=1, location=l); bpy.context.object.dimensions=d\ncaixa((0,0,0.375),(1.6,0.9,0.75)); caixa((0,-0.875,0.225),(0.45,0.45,0.45)); caixa((0,0.875,0.225),(0.45,0.45,0.45))",
    "texto": "bpy.ops.object.text_add(); d=bpy.context.object.data; d.body='Olá 3D'; d.extrude=0.1; d.align_x='CENTER'",
    "subdiv": "bpy.ops.mesh.primitive_uv_sphere_add(); bpy.ops.object.shade_smooth(); m=bpy.context.object.modifiers.new('S','SUBSURF'); m.levels=2; m.render_levels=2",
}
