"""Regression checks for request boundaries, phase isolation and held-out profiles."""
import gzip
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_domain_profiles as tool
import make_profile as mp


def window(expert,tokens=1,k=1):
    return b''.join(struct.pack('<ii',l,k)+struct.pack('<'+'i'*k,*([expert]*k))+b'\0'*(4*k)
                    for l in range(mp.N_LAYER) for _ in range(tokens))


class DomainProfiles(unittest.TestCase):
    def test_prompt_routing_is_excluded_using_exact_decode_entries(self):
        c,audit = tool.decode_counts(window(0,12)+window(1,3),mp.N_LAYER*3,n_expert=4)
        self.assertEqual(int(c[:,0].sum()),0)
        self.assertEqual(int(c[:,1].sum()),mp.N_LAYER*3)
        self.assertEqual(audit['prompt_entries'],mp.N_LAYER*12)

    def test_generated_count_cannot_split_a_speculative_window(self):
        with self.assertRaisesRegex(ValueError,'complete-window suffix'):
            tool.decode_counts(window(1,4),mp.N_LAYER*3,n_expert=4)

    def test_truncated_and_empty_records_fail(self):
        for data in (b'',window(1)[:-1],window(1)[:-4]):
            with self.subTest(bytes=len(data)),self.assertRaises(ValueError):
                tool.decode_counts(data,mp.N_LAYER,n_expert=4)

    def test_reordered_layers_fail(self):
        data = bytearray(window(1))
        data[:4] = struct.pack('<i',1)
        with self.assertRaises(ValueError):
            tool.decode_counts(data,mp.N_LAYER,n_expert=4)

    def test_invalid_experts_and_k_fail(self):
        for data in (window(4),struct.pack('<ii',0,0)+b'\0'*8):
            with self.assertRaises(ValueError):
                tool.decode_counts(data,mp.N_LAYER,n_expert=4)

    def samples(self):
        samples=[]
        for group,expert in [('a',1),('b',2)]:
            for i in range(4):
                counts=np.zeros((mp.N_LAYER,4),np.int64)
                counts[:,expert]=i+1
                samples.append(dict(id=f'{group}{i}',domain=group,counts=counts,
                                    audit=dict(decode_entries=int(counts.sum()),prompt_entries=0,windows=1)))
        return samples

    def test_domain_profiles_exclude_entire_test_fold(self):
        r=tool.evaluate(self.samples(),mp.N_LAYER,folds=2)
        for row in r['rows']:
            heldout={x['id'] for x in r['rows'] if x['fold']==row['fold']}
            self.assertTrue(heldout.isdisjoint(row['training_ids']))
            self.assertTrue(heldout.isdisjoint(row['domain_training_ids']))
            self.assertEqual(row['domain_coverage'],1)
        self.assertGreater(r['domain_minus_global_pp'],0)

    def test_fold_assignment_does_not_depend_on_manifest_order(self):
        a=tool.evaluate(self.samples(),48,folds=2)
        b=tool.evaluate(list(reversed(self.samples())),48,folds=2)
        self.assertEqual([(x['id'],x['fold']) for x in a['rows']],[(x['id'],x['fold']) for x in b['rows']])
        self.assertEqual(a['mean_domain'],b['mean_domain'])

    def test_single_request_domain_and_invalid_budgets_fail(self):
        for samples,slots,folds in [(self.samples()[:1],48,2),(self.samples(),0,2),(self.samples(),193,2),(self.samples(),48,1)]:
            with self.subTest(slots=slots,folds=folds),self.assertRaises(ValueError):
                tool.evaluate(samples,slots,folds)

    def test_layer_quotas_are_preserved(self):
        score=np.random.default_rng(1).random((mp.N_LAYER,4))
        quotas=np.arange(mp.N_LAYER)%5
        result=tool.quota_profile(score,quotas)
        np.testing.assert_array_equal(np.bincount(result//4,minlength=mp.N_LAYER),quotas)

    def test_profile_bank_roundtrip_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'profiles'
            tool.write_bank(self.samples(),out,48)
            index=json.loads((out/'index.json').read_text())
            self.assertEqual(set(index['domains']),{'a','b'})
            for value in [index['global_profile'],*index['domains'].values()]:
                pairs=mp.read_profile(out/value['path'],4)
                self.assertEqual(len(pairs),mp.N_LAYER*4)
                self.assertEqual(len(set(pairs)),len(pairs))
            with self.assertRaises(FileExistsError):
                tool.write_bank(self.samples(),out,48)

    def test_domain_named_global_cannot_replace_fallback(self):
        samples=self.samples()
        for s in samples:
            if s['domain']=='a':
                s['domain']='global'
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'bank'
            tool.write_bank(samples,out,48)
            index=json.loads((out/'index.json').read_text())
            self.assertNotEqual(index['global_profile']['path'],index['domains']['global']['path'])

    def test_profile_prefixes_rank_selected_members_by_frequency(self):
        samples=self.samples()
        rng=np.random.default_rng(981)
        for s in samples:
            s['counts']=rng.integers(1,100,(mp.N_LAYER,4))
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'bank'
            tool.write_bank(samples,out,48)
            index=json.loads((out/'index.json').read_text())
            global_pairs=mp.read_profile(out/index['global_profile']['path'],4)[:48]
            quotas=np.bincount([l for l,e in global_pairs],minlength=mp.N_LAYER)
            for domain,info in index['domains'].items():
                pairs=mp.read_profile(out/info['path'],4)[:48]
                score=sum(s['counts']/s['counts'].sum() for s in samples if s['domain']==domain)
                values=np.array([score[l,e] for l,e in pairs])
                self.assertTrue(np.all(np.diff(values)<=0))
                np.testing.assert_array_equal(np.bincount([l for l,e in pairs],minlength=mp.N_LAYER),quotas)

    def test_manifest_accepts_gzip_and_rejects_duplicate_or_overlapping_requests(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)
            data=window(1)
            (directory/'data.gz').write_bytes(gzip.compress(data))
            row=dict(id='a',domain='code',trace='data.gz',start=0,end=len(data),decode_entries=48)
            path=directory/'manifest.json'
            path.write_text(json.dumps(dict(samples=[row])))
            self.assertEqual(int(tool.load_samples(path,4)[0]['counts'].sum()),48)
            for rows in ([row,row],[row,dict(row,id='b')],[dict(row,start=-4)],[dict(row,end=len(data)+4)]):
                path.write_text(json.dumps(dict(samples=rows)))
                with self.assertRaises(ValueError):
                    tool.load_samples(path,4)


if __name__=='__main__':
    unittest.main()
