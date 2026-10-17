import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
class GyroSAuxiliaryTests(unittest.TestCase):
 def setUp(self):
  self.data=json.loads((ROOT/'data/ALL-METRICS.json').read_text());self.model=self.data['models']['gyro-s'];self.audit=json.loads((ROOT/'data/gyro-s-rc2-auxiliary-audit.json').read_text())
 def test_import_hashes_and_identity(self):
  ref=self.data['gyro_s_campaign']['auxiliary_audit'];self.assertEqual(hashlib.sha256((ROOT/ref['path']).read_bytes()).hexdigest(),ref['sha256'])
  for k in ['model_revision','model_sha256','engine_commit','engine_sha256']:self.assertEqual(self.audit['identity'][k],self.model['setup_identity'][k])
  for name,v in self.audit['artifacts'].items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),v['packaged_sha256'])
 def test_valid_report_denominators(self):
  texts={k:(ROOT/'data'/self.model['knowledge'][k]['_file']).read_text() for k in ['gsm8k','ifeval']}
  self.assertIn('(96/100)',texts['gsm8k']);self.assertIn('(38/50)',texts['ifeval']);self.assertIn('(63/76)',texts['ifeval'])
  needle=(ROOT/'data'/self.model['needle']['_file']).read_text();self.assertIn('(25/25)',needle);self.assertEqual(self.model['needle']['effective_context'],129024)
 def test_fivebugs_independent_counts(self):
  matrix=json.loads((ROOT/'data/gyro-s-rc2-fivebugs-matrix.json').read_text());tag='gyro-s-rc2-'+self.audit['run_id'];m=matrix[tag]
  values=[x for v in m['per_task'].values() for x in v];self.assertEqual(len(values),25)
  self.assertEqual(sum(x['vis'] for x in values),21);self.assertEqual(sum(x['hid'] for x in values),16)
  records=[json.loads(l) for l in (ROOT/'data/gyro-s-rc2-fivebugs.jsonl').read_text().splitlines()]
  self.assertEqual({(x['task'],x['run']) for x in records},{(f't{i}',j) for i in range(1,6) for j in range(1,6)})
  self.assertTrue(all(x['meta']['reasoning_effort']=='xhigh' for x in records));self.assertEqual(sum(x['meta']['usage']['completion_tokens'] for x in records),self.model['fivebugs']['tokens'])
  self.assertEqual(self.model['fivebugs']['wall_s'],441.19)
 def test_algebra_resolved_offline_rc1_preserved(self):
  self.assertEqual(self.model['knowledge']['mmlu']['accuracy'],79.0);self.assertEqual(self.model['knowledge']['mmlu']['engine_cohort'],'rc2')
  self.assertEqual(self.model['legacy_rc1_auxiliary']['knowledge']['mmlu']['accuracy'],76)
  self.assertIn('HTTP Error 429',(ROOT/'data/gyro-s-rc2-mmlu-failure.log').read_text())
  self.assertEqual(self.data['gyro_s_campaign']['run_done']['outcome'],'aborted_or_invalid')
 def test_four_valid_slices_not_discarded_with_failed_auxiliary(self):
  self.assertEqual(set(self.audit['validated_slices']),{'plugin-gsm8k','plugin-ifeval','plugin-needle','fivebugs'})
  self.assertEqual(self.model['auxiliary_cohort'],'rc2');self.assertEqual(self.model['knowledge']['gsm8k']['accuracy'],96)
  self.assertEqual(self.model['knowledge']['ifeval']['prompt_accuracy'],76)
if __name__=='__main__':unittest.main()
