import copy,json,unittest
from pathlib import Path
from decision_matrix import build_matrix,POLICY,RECOMMENDATIONS
ROOT=Path(__file__).resolve().parent.parent
class DecisionMatrixTests(unittest.TestCase):
 def setUp(self):
  self.data=json.loads((ROOT/'data/ALL-METRICS.json').read_text());self.rows=build_matrix(self.data)
 def test_all_twenty_unique_without_scores(self):
  self.assertEqual(len(self.rows),20);self.assertEqual(len({(r['key'],r['arm']) for r in self.rows}),20)
  self.assertTrue(all(not any(k in r for k in ['score','rank','overall','composite']) for r in self.rows));self.assertIsNone(POLICY['weights'])
 def test_gyro_s_sampled_complete(self):
  for r in self.rows:
   self.assertIsNotNone(r['sampled']);self.assertIsNone(r['legacy_sampled'])
   if r['key']=='gyro-s':
    self.assertEqual(r['sampled']['algebra'],79.0);self.assertEqual(r['sampled']['gsm8k'],96.0)
    self.assertEqual(r['sampled']['ifeval_prompt'],76.0);self.assertEqual(r['sampled']['debug_hidden'],'16/25')
    self.assertNotEqual(r['verdict'],'Recommended default')
 def test_every_family_preserved(self):
  for r in self.rows:
   sample=r['sampled'] or r['legacy_sampled'];m=self.data['models'][r['key']]
   self.assertEqual(sample['gsm8k'],m['knowledge']['gsm8k']['accuracy']);self.assertEqual(sample['algebra'],m['knowledge']['mmlu']['accuracy'])
   self.assertEqual(sample['ifeval_prompt'],m['knowledge']['ifeval']['prompt_accuracy']);self.assertEqual(sample['ifeval_instruction'],m['knowledge']['ifeval']['instruction_accuracy'])
   self.assertEqual(sample['debug_hidden'],m['fivebugs']['hid']);self.assertEqual(sample['needle_context_tokens'],m['needle']['effective_context'])
 def test_default_is_unique_and_stock(self):
  chosen=[r for r in self.rows if r['verdict']=='Recommended default'];self.assertEqual(len(chosen),1)
  self.assertEqual((chosen[0]['key'],chosen[0]['arm']),('swift-iq3_s','low'));self.assertEqual(chosen[0]['cohort'],'stock')
  self.assertEqual(RECOMMENDATIONS[0]['key'],chosen[0]['key']);self.assertTrue(all('rank' not in r for r in RECOMMENDATIONS))
 def test_no_new_coverage_or_speed_invented(self):
  mutated=copy.deepcopy(self.data)
  for m in mutated['models'].values():m['perf']={}
  self.assertEqual(build_matrix(mutated),self.rows)
  del mutated['models']['swift-iq3_s']['knowledge']['gsm8k']
  with self.assertRaises(KeyError):build_matrix(mutated)
 def test_inputs_unchanged_and_shared_sample_not_new_effort_run(self):
  before=copy.deepcopy(self.data);build_matrix(self.data);self.assertEqual(before,self.data)
  for key in self.data['models']:
   rr=[r for r in self.rows if r['key']==key]
   self.assertEqual(rr[0]['sampled'],rr[1]['sampled']);self.assertIn('sample',rr[0]['sampled_scope'] if key!='gyro-s' else 'sample')
if __name__=='__main__':unittest.main()
