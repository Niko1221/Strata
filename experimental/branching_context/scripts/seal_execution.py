"""Record a rebuilt bridge without changing the independently frozen tolerances."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
location = ROOT/'build/real-profile.json'
raw = location.read_bytes()
profile = json.loads(raw)
build = json.loads((ROOT/'build/build.json').read_text())
oracle = profile['oracle_binary']
if build['sha256'][oracle] != profile['native_build']['sha256'][oracle]:
    raise SystemExit('Oracle binary changed: new independent calibration required')
old = ROOT/'evidence/runs'/('baseline-profile-'+hashlib.sha256(raw).hexdigest()+'.json')
old.write_bytes(raw)
profile['baseline_profile_sha256'] = hashlib.sha256(raw).hexdigest()
profile['native_build'] = build
profile['execution_profile_id'] = hashlib.sha256(json.dumps(dict(build=build,
    execution=profile['execution'],hardware=profile['hardware']),sort_keys=True).encode()).hexdigest()
location.write_text(json.dumps(profile,indent=2)+'\n')
print('Sealed current bridge; oracle binary and baseline tolerances unchanged')
