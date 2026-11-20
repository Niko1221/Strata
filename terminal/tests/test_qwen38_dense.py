"""Bounded product wiring; fixtures do not qualify a real checkpoint."""
import hashlib
import json
import os
import struct
from pathlib import Path
from unittest import mock

import pytest

from strata_control import models, qwen38_dense as dense, engine_component as ec
from tests.terminal.test_engine_component import make_package, rewrite

def fixture(path, *, width=5120, codec=23, architecture="qwen35", split=False, nine=False):
    def string(s):
        b=s.encode("utf-8"); return struct.pack("<Q",len(b))+b
    dims={"block_count":32 if nine else 64,"embedding_length":4096 if nine else width,
          "feed_forward_length":12288 if nine else 17408,"attention.head_count":16 if nine else 24,
          "attention.head_count_kv":4,"attention.key_length":256,"attention.value_length":256,
          "ssm.state_size":128,"ssm.group_count":16,"ssm.time_step_rank":32 if nine else 48,
          "ssm.inner_size":4096 if nine else 6144}
    pairs=[("general.architecture",8,string(architecture)),("tokenizer.ggml.model",8,string("gpt2")),
           ("tokenizer.ggml.tokens",9,struct.pack("<IQ",8,3)+string("a")+string("<|im_end|>")+string("<|endoftext|>")),
           ("tokenizer.ggml.token_type",9,struct.pack("<IQiii",5,3,1,3,3)),
           ("tokenizer.ggml.eos_token_id",4,struct.pack("<I",1)),
           ("tokenizer.chat_template",8,string("original:{% for m in messages %}{{m.content}}{% endfor %}")),
           ("tokenizer.ggml.merges",9,struct.pack("<IQ",8,1)+string("a a"))]
    pairs += [("qwen35."+k,4,struct.pack("<I",v)) for k,v in dims.items()]
    if split:pairs.append(("split.count",4,struct.pack("<I",1)))
    header=b"GGUF"+struct.pack("<IQQ",3,1,len(pairs))+b"".join(string(k)+struct.pack("<I",t)+v for k,t,v in pairs)
    count,size={23:(256,136),142:(128,34),21:(256,110),22:(256,82),12:(256,144),14:(256,210),2:(32,18)}.get(codec,(32,18))
    header+=string("token_embd.weight")+struct.pack("<IQIQ",1,count,codec,0)
    path.write_bytes(header+b"\0"*(-len(header)%32)+b"\0"*size)
    return path

@pytest.mark.parametrize("name,codec",[("Renamed-Flash-Swift.gguf",23),("fine-tune.gguf",142),("mix.gguf",21),("other.gguf",22)])
def test_preview_register_production_arguments(tmp_path,name,codec):
    model=fixture(tmp_path/name,codec=codec)
    before=model.read_bytes()
    plan=models.plan_import(str(model))
    assert not plan["issues"]
    assert plan["candidate_selection"]["family"]==dense.ARCH
    with mock.patch("urllib.request.urlopen",side_effect=AssertionError("unexpected network")):
        saved=dense.register(str(model),directory=tmp_path,context=98304)
    cfg=json.loads(Path(saved["config"]).read_text(encoding="utf-8"))
    assert cfg["native_runner"]==dense.ARCH and cfg["exe"]=="component:strata-engine"
    from serve.server import engine_args
    args=engine_args(cfg)
    assert args==["--model",str(model.resolve()),"--ctx-size","98304","--kv","auto","--output-language","ja"]
    assert model.read_bytes()==before
    cfg.update(output_language="model-default",dense_kv="f16",unrelated={"kept":True})
    Path(saved["config"]).write_text(json.dumps(cfg),encoding="utf-8")
    dense.register(str(model),directory=tmp_path)
    assert json.loads(Path(saved["config"]).read_text(encoding="utf-8"))==cfg

def test_preview_accepts_9b_geometry(tmp_path):
    plan=models.plan_import(str(fixture(tmp_path/"Qwen3.5-9B-Q4_K_M.gguf",nine=True,codec=12)))
    assert not plan["issues"]
    selection=plan["candidate_selection"]
    assert selection["family"]==dense.ARCH and selection.get("geometry")=="q38-9b"

@pytest.mark.parametrize("options,reason",[({"width":4096},"geometry"),({"split":True},"unsplit"),({"codec":2},"codecs"),({"architecture":"qwen35moe"},None)])
def test_preview_rejects_incompatible_dense(tmp_path,options,reason):
    plan=models.plan_import(str(fixture(tmp_path/"Swift-Flash.gguf",**options)))
    assert (plan["candidate_selection"] or {}).get("family")!=dense.ARCH
    if reason:assert reason in " ".join(plan["issues"])

@pytest.mark.parametrize("patch",[{"output_language":"zh"},{"dense_kv":"turbo5"},{"max_context":True},{"max_context":262145},{"parallel":2},{"mtp":True},{"args":["--model","C:/x.gguf","--gpu-layers","64"]}])
def test_invalid_dense_options(patch):
    with pytest.raises(ValueError):dense.arguments({"args":["--model",str(Path.cwd()/"x.gguf")],**patch})

@pytest.mark.parametrize("source", ["user", "raw", "edited", "generated"])
def test_dense_explicit_argument_precedence_and_scalar_reports(source):
    model=["--model",str(Path.cwd()/"original.gguf")]
    choices=["--ctx-size","131072","--kv","f16","--output-language","model-default"]
    defaults=["--ctx-size","4096","--kv","auto","--output-language","ja"]
    cfg={"args":model,"max_context":4096,"dense_kv":"auto","output_language":"ja"}
    if source=="user":cfg.update(args=model+defaults,args_written=model+defaults,user_args=choices)
    elif source=="raw":cfg["args"]=model+choices
    elif source=="edited":cfg.update(args=model+choices,args_written=model+defaults,user_args=defaults)
    else:cfg.update(args=model+defaults,args_written=model+defaults,max_context=131072,dense_kv="f16",output_language="model-default")
    assert dense.arguments(cfg)==model+choices
    assert (cfg["max_context"],cfg["dense_kv"],cfg["output_language"])==(131072,"f16","model-default")
    assert dense.arguments(cfg)==model+choices

@pytest.mark.parametrize("choices", [["--ctx-size","0"],["--ctx-size","1e5"],["--kv","invalid"],["--output-language","zh"],["--gpu-layers","64"]])
def test_explicit_arguments_cannot_bypass_validation(choices):
    import copy
    cfg={"args":["--model",str(Path.cwd()/"original.gguf")],"user_args":choices,"max_context":4096,"dense_kv":"auto","output_language":"ja"}
    before=copy.deepcopy(cfg)
    with pytest.raises(ValueError):dense.arguments(cfg)
    assert cfg==before

def test_terminal_reconciles_user_arguments_for_saved_and_future_load(tmp_path):
    from tests.terminal.launch_support import Home, catalog
    from strata_cli.launch.settings import LaunchSettings
    from strata_cli.launch.runner import Runner
    from serve.server import engine_args
    model=fixture(tmp_path/"original.gguf")
    choices=["--ctx-size","131072","--kv","f16","--output-language","model-default"]
    with Home():
        runner=Runner(LaunchSettings(model_path=str(model)),catalog=catalog())
        runner.plan()
        base=runner.ensure_model()
        cfg=json.loads(base.read_text(encoding="utf-8"));cfg["user_args"]=choices
        base.write_text(json.dumps(cfg),encoding="utf-8")
        run=json.loads(runner.build_config(base).read_text(encoding="utf-8"))
        assert (run["max_context"],run["dense_kv"],run["output_language"])==(131072,"f16","model-default")
        assert engine_args(run)[-6:]==choices
        runner.ensure_model()  # normal re-registration reconciles future reporting too
        saved=json.loads(base.read_text(encoding="utf-8"))
        assert (saved["max_context"],saved["dense_kv"],saved["output_language"])==(131072,"f16","model-default")
        assert saved["user_args"][-6:]==choices
        assert engine_args(saved)[-6:]==choices

def test_signed_parent_full_inventory_and_covered_helper(tmp_path):
    helper="strata-q38.exe" if os.name=="nt" else "strata-q38"
    folder=make_package(tmp_path,extra={helper:b"runner", "sidecar.dat":b"covered"})
    rewrite(folder,signature={"alg":"ed25519","key_id":"fixture","value":"AA=="})
    cfg={"native_runner":dense.ARCH,"backend":"cuda"}
    with mock.patch.object(ec,"_verifier",lambda *_:None):
        assert Path(ec.expand("component:strata-engine",cfg,root=tmp_path,full=False)).name==helper
        (folder/"sidecar.dat").write_bytes(b"changed")
        with pytest.raises(ec.EngineMissing):ec.expand("component:strata-engine",cfg,root=tmp_path,full=False)
        (folder/"sidecar.dat").write_bytes(b"covered")
        (folder/helper).write_bytes(b"edited")
        with pytest.raises(ec.EngineMissing):ec.expand("component:strata-engine",cfg,root=tmp_path)
        (folder/helper).write_bytes(b"runner")
        manifest=json.loads((folder/ec.MANIFEST).read_text())
        rewrite(folder,files=[f for f in manifest["files"] if f["path"]!=helper])
        with pytest.raises(ec.EngineMissing):ec.expand("component:strata-engine",cfg,root=tmp_path)

def test_unsigned_parent_cannot_supply_helper(tmp_path):
    helper="strata-q38.exe" if os.name=="nt" else "strata-q38"
    make_package(tmp_path,extra={helper:b"runner"})
    with pytest.raises(ec.EngineMissing):ec.expand("component:strata-engine",{"native_runner":dense.ARCH,"backend":"cuda"},root=tmp_path)

def test_literal_and_rendering_production_boundary():
    from serve.output_language import literal_exceptions,INSTRUCTION,protocol_exceptions,dense_sampling_keys
    from serve.frontend import ChatTemplate
    from serve.server import Service
    from types import SimpleNamespace
    svc=object.__new__(Service)
    svc.engine=SimpleNamespace(output_language="ja");svc.effort_end=False
    svc.template=ChatTemplate(None,source="{% for m in messages %}{{m.role}}={{m.content}};{% endfor %}{{tools}}")
    messages=[{"role":"system","content":"keep system"},{"role":"user","content":"國/臺灣 𠮷 JSON"},{"role":"assistant","content":"這"}]
    tools=[{"name":"literal_學","parameters":{}}]
    rendered=svc.render_prompt(messages,tools,{})
    assert INSTRUCTION in rendered and "keep system" in rendered and "國/臺灣 𠮷 JSON" in rendered and "literal_學" in rendered
    assert "推論欄（thinking/reasoning）の本文自体" in rendered and "最初から最後まで日本語" in rendered
    assert "指定された出力形式" in rendered and "コードは原文を忠実に保ってください" in rendered
    assert messages[0]["content"]=="keep system"
    mask=int(literal_exceptions(messages,tools),16)
    assert mask&(1<<7) and mask&(1<<22) and mask&(1<<23) and not mask&1
    assert protocol_exceptions(format(mask,"x")).startswith(" ja_allow=")
    svc.engine.output_language="model-default"
    assert INSTRUCTION not in svc.render_prompt(messages,tools,{})
    assert dense_sampling_keys({"temperature":0,"top_k":0,"seed":0})==" temperature=0 top_k=0 seed=0"
    with pytest.raises(ValueError):dense_sampling_keys({"logit_bias":{"2":3}})
    for bad in ("0\nGEN", "1000000", -1):
        with pytest.raises(ValueError):protocol_exceptions(bad)

@pytest.mark.parametrize("language", ["ja", "model-default"])
def test_actual_prepare_run_native_protocol_reasoning_and_final(language):
    import io
    import queue
    import threading
    from types import SimpleNamespace
    from serve import server
    from serve.frontend import ChatTemplate
    from serve.output_language import INSTRUCTION, literal_exceptions
    tok=server.ByteTokenizer()
    engine=object.__new__(server.StrataEngine)
    engine.alive=lambda:True
    engine.batch,engine.silence_s,engine.max_context=0,0,4096
    engine.proc=SimpleNamespace(stdin=io.StringIO(),pid=4242)
    engine.lines=queue.Queue()
    answer="日本語で考えます。</think>\n答えは國と臺灣です。"
    for token in tok.encode(answer):engine.lines.put(f"T {token}")
    engine.lines.put(f"DONE {len(tok.encode(answer))} 1 1 1 stop")
    engine.last={};engine.info={"engine":"qwen38_dense"};engine.can_stop=True;engine.output_language=language
    svc=server.Service(engine,tok,ChatTemplate(None,source="{% for m in messages %}{{m.content}}\n{% endfor %}<think>"))
    messages=[{"role":"system","content":"Existing system"},{"role":"user","content":"國と臺灣"},{"role":"assistant","content":"這"}]
    cancel=threading.Event()
    ids,thinking,cap=svc.prepare(messages,None,{},256,cancel=cancel)
    assert (INSTRUCTION in tok.decode(ids))==(language=="ja")
    events=list(svc.run(ids,thinking,None,cap,{"temperature":0,"top_k":0,"seed":0},cancel))
    streamed=[e for kind,e in events if kind=="event"]
    assert "日本語で考えます。" in "".join(e.text or "" for e in streamed if e.kind=="reasoning")
    assert "答えは國と臺灣です。" in "".join(e.text or "" for e in streamed if e.kind=="content")
    wire=engine.proc.stdin.getvalue()
    assert "temperature=0 top_k=0 seed=0" in wire
    assert ("ja_allow="+literal_exceptions(messages) in wire)==(language=="ja")
    assert wire.startswith("GEN ") and "cache_scope" not in wire and "hold=" not in wire

def test_terminal_local_plan_registration_and_profile_without_acquisition(tmp_path):
    from tests.terminal.launch_support import Home, catalog
    from strata_cli.launch.settings import LaunchSettings
    from strata_cli.launch.runner import Runner
    from serve.server import engine_args
    model=fixture(tmp_path/"renamed-native.gguf")
    with Home() as home, mock.patch("urllib.request.urlopen",side_effect=AssertionError("unexpected network")):
        s=LaunchSettings(model_path=str(model),context=131072,kv="f16_turbo4",output_language="model-default",vision="off",parallel=1)
        runner=Runner(s,catalog=catalog())
        plan=runner.plan()
        assert plan["model"]["family"]==dense.ARCH and not plan["would_write"]
        assert not list(home.base.rglob("strata-q38-dense-*.json"))
        base=runner.ensure_model()
        config=runner.build_config(base)
        runner.ensure_profile(config)
        cfg=json.loads(config.read_text(encoding="utf-8"))
        assert cfg["max_context"]==131072 and cfg["output_language"]=="model-default"
        assert engine_args(cfg)==["--model",str(model.resolve()),"--ctx-size","131072","--kv","f16_turbo4","--output-language","model-default"]

def test_original_embedded_template_tokenizer_and_eos(tmp_path):
    from serve.server import dense_tokenizer, Service, MockEngine
    from serve.frontend import ChatTemplate
    path=fixture(tmp_path/'original.gguf')
    with mock.patch("urllib.request.urlopen",side_effect=AssertionError("unexpected network")):
        tok,template=dense_tokenizer(path)
    assert tok.encode("a<|im_end|>",parse_special=True)==[0,1]
    svc=Service(MockEngine(tok,"a"),tok,ChatTemplate(None,source=template))
    assert svc.render_prompt([{"role":"user","content":"a"}],None,{})=="original:a"
    assert 1 in svc.stop_ids
