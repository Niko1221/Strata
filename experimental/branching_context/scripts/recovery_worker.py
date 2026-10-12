"""Separate-process recovery measurement; stdout contains measured JSON only."""
import json
import hashlib
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context_time_travel.real import BranchingModel
from recovery_observation import record

profile = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
backend = BranchingModel.load_backend(profile)
try:
    sut = BranchingModel(sys.argv[1],backend,profile)
    sid = sys.argv[2]
    manifest_path = sut.checkpoint_directory/(sid+'.json')
    if manifest_path.exists():
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)['manifest']
        payload = (sut.checkpoint_directory/(manifest['payload_sha256']+'.state')).read_bytes()
        record('checkpoint-manifest',snapshot=sid,manifest_sha256=hashlib.sha256(raw).hexdigest(),
            manifest_bytes=len(raw),payload_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
            native_state_bytes=manifest['state_bytes'],logits_count=manifest['logits_count'])
    result = dict(pid=os.getpid(),history=sut.history(sid),logits=sut.next_logits(sid),controller=sut.controller(sid))
    if len(sys.argv)>3:
        if sys.argv[3]=='--feed':
            tip = sut.feed(sid,json.loads(sys.argv[4]))
        else:
            generated = sut.generate(sid,int(sys.argv[3]))
            tip = generated.snapshot_id
            result['generated'] = generated.accepted_tokens
        result.update(final_history=sut.history(tip),final_logits=sut.next_logits(tip),final_controller=sut.controller(tip))
    result['counters'] = dict(backend.stats)
    record('cold-process-restore',consumed=len(result['history']),continued=len(result.get('final_history',result['history']))-len(result['history']),
        counters=result['counters'],controller=result['controller'])
    print(json.dumps(result))
finally:
    backend.close()
