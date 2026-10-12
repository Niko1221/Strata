"""Seal only executed recovery evidence; process termination is not power loss."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--report',type=Path,required=True)
p.add_argument('--measurements',type=Path,required=True)
p.add_argument('--observations',type=Path,required=True)
args = p.parse_args()
report = json.loads(args.report.read_text())
required = {r['test'] for r in json.loads((ROOT/'docs/acceptance.json').read_text())['cases'] if r['gate']=='G2'}
executed = {name.rsplit('.',1)[-1] for name in report['executed_ids']}
if required-executed or report['status']!='PASS' or any(report[k] for k in ('errors','failures','skipped','expected_failures')):
    raise SystemExit('Missing or failed recovery acceptance case')
rows = [json.loads(line) for line in args.measurements.read_text().splitlines()]
observations = [json.loads(line) for line in args.observations.read_text().splitlines()]
points = sorted({r['point'] for r in observations if r['kind']=='process-termination'})
required_points = {'before_snapshot_publish','after_refs_stage_fsync','after_refs_rename_before_dir_fsync','after_token_consume'}
if set(points)!=required_points:
    raise SystemExit('Missing actual process termination measurements')
oom = [r for r in observations if r['kind']=='allocation-failure' and r['allocation_failed']]
restores = [r for r in observations if r['kind']=='cold-process-restore']
manifests = [r for r in observations if r['kind']=='checkpoint-manifest']
if not oom or len(restores)<2 or not all(r['continued']>0 for r in restores):
    raise SystemExit('Missing actual allocation-failure or continuation-after-restart measurements')
if len(manifests)<2 or any(r['payload_bytes']!=r['native_state_bytes']+4*r['logits_count'] for r in manifests):
    raise SystemExit('Missing or inconsistent real saved-byte measurements')
inspection = json.loads((ROOT/'evidence/runs/backend-inspection.json').read_text())
if inspection['filesystem']['filesystems'][0]['fstype']!='ext4':
    raise SystemExit('Recovery certification here requires the measured ext4 target')
profile = json.loads((ROOT/'build/real-profile.json').read_text())
result = dict(gate='G2',status='PASS',test_count=report['test_count'],executed_ids=report['executed_ids'],
    filesystem=inspection['filesystem'],failure_points_executed=points,
    checkpoint_manifest_sha256=manifests[0]['manifest_sha256'],checkpoint_files_measured=manifests,
    model_restore_check=dict(max_absolute_error=max(r['numerics']['max_absolute_error'] for r in rows),
        max_normalized_error=max(r['numerics']['max_normalized_error'] for r in rows),
        comparisons=len(rows),atol=profile['atol'],rtol=profile['rtol']),
    process_restart_results=restores,allocation_failure_results=oom,power_loss_tested=False,
    source_sha256=report['source_sha256'],
    model_weights_sha256=profile['model_weights_sha256'],backend_revision=profile['backend_revision'],
    execution_profile_id=profile['execution_profile_id'],native_build=profile['native_build'],
    max_owned_checkpoint_bytes=max(r['owned_checkpoint_bytes'] for r in rows),
    evidence_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.report,args.measurements,args.observations)},
    limitations=['Process termination and restart on local ext4; no power-loss simulator',
        'Actual host state-export allocation exhaustion; not GPU OOM certification',
        'No network filesystem or Windows durability claim','One pinned text-only full-attention model'])
if result['model_restore_check']['max_normalized_error']>1:
    raise SystemExit('Numerical failure')
(ROOT/'evidence/profiles/G2.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['model_restore_check']))
