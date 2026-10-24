"""executor3d.py - roda DENTRO do Blender (blender -b --factory-startup --python executor3d.py -- <tarefa> <script> <saida.json>).
Comeca de uma cena vazia, executa o script (do modelo ou de referencia), roda o verificador da tarefa (tarefas3d.py) e grava
{"erro": ..., "passou": ..., "detalhe": ...} em saida.json. Um erro no script ou no verificador conta como "nao passou".
"""
import json, os, sys, traceback

import bpy

tarefa, script, saida = sys.argv[sys.argv.index("--") + 1:][:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
res = {"tarefa": tarefa, "erro": None, "passou": False, "detalhe": ""}
try:
    codigo = open(script, encoding="utf-8").read()
    exec(compile(codigo, "script_do_modelo.py", "exec"), {"__name__": "__main__", "bpy": bpy})
except BaseException:
    res["erro"] = "".join(traceback.format_exc().splitlines(True)[-4:])[-800:]
if res["erro"] is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tarefas3d
    checar = {t[0]: t[3] for t in tarefas3d.TAREFAS}[tarefa]
    try:
        ok, det = checar()
        res["passou"], res["detalhe"] = bool(ok), det
    except BaseException:
        res["detalhe"] = "erro no verificador: " + "".join(traceback.format_exc().splitlines(True)[-3:])[-600:]
with open(saida, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False)
