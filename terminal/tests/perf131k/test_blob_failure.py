"""Extracted consumer CPU/fake-I/O tests. No CUDA DLL or product executable is loaded."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest
import uuid

ROOT=Path(__file__).resolve().parents[2]


def function(text,signature):
    start=text.index(signature);opening=text.index('{',start);end=opening+1;depth=1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]


class BlobFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=Path(os.environ['P01_DIAGNOSTICS_ARTIFACTS'])/uuid.uuid4().hex;cls.base.mkdir()
        source=Path(os.environ.get('P01_BLOB_SOURCE_UNDER_TEST') or ROOT/'src/core/expert_source.cpp').read_text(encoding='utf-8')
        generate=Path(os.environ.get('P01_GENERATE_UNDER_TEST') or ROOT/'src/program/generate.cpp').read_text(encoding='utf-8')
        direct=function(source,'bool FileExpertSource::read_direct(')
        direct=direct[:direct.index('#elif defined(__linux__)')]+'#endif\n}'
        staged=function(source,'const uint8_t* FileExpertSource::staged_blob(')
        start=source.index('        void* p = nullptr;',source.index('bool FileExpertSource::claim_stage('))
        end=source.index('        std::unique_ptr<uint8_t[], StageBufFree> owned',start)
        alloc=source[start:end]
        old='const uint8_t* b = srcp->blob_stable(profile[(size_t) i].first'
        anchor='strata::perf131k::blob_failure::Scope profile_blob_scope'
        start=generate.index(anchor if anchor in generate else old)
        end=generate.index('        fill_bytes +=',start)
        template=(ROOT/'tests/perf131k/blob_failure_cpu.cpp').read_text(encoding='utf-8')
        cpp=cls.base/'consumer.cpp'
        cpp.write_text(template.replace('// @DIRECT_FUNCTION@',direct).replace('// @STAGED_FUNCTION@',staged)
                       .replace('// @ALLOC_BLOCK@',alloc).replace('// @PROFILE_BLOCK@',generate[start:end]),encoding='utf-8')
        receipt=json.loads(Path('H:/Strata/evidence/p01-b0-20261025/build/compile-expert_source.json').read_text())
        cl=Path(receipt['command'][0]);env=dict(os.environ,**{k:receipt['environment'][k] for k in ('INCLUDE','LIB')})
        env['PATH']=str(cl.parent)+os.pathsep+env['PATH']
        include=Path(os.environ.get('P01_BLOB_HEADER_ROOT') or ROOT/'include')
        cls.exe=cls.base/'consumer.exe'
        command=[str(cl),'/nologo','/std:c++20','/EHsc','/Od','/I'+str(include),str(cpp),
                 '/Fe:'+str(cls.exe),'/Fo:'+str(cls.base/'consumer.obj'),'/link','/INCREMENTAL:NO']
        (cls.base/'command.json').write_text(json.dumps({'command':command,'compiler_sha256':hashlib.sha256(cl.read_bytes()).hexdigest(),
            'INCLUDE':env['INCLUDE'],'LIB':env['LIB'],'scope':'CPU extracted functions and fake APIs, not product build'}),encoding='utf-8')
        with (cls.base/'compile.txt').open('xb') as out:
            code=subprocess.run(command,cwd=cls.base,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=20).returncode
        if code:raise AssertionError(str(cls.base/'compile.txt'))

    def run_case(self,mode,on=True,env_value=None):
        name=mode+('-on' if on else '-off')+'-'+uuid.uuid4().hex
        env=dict(os.environ);env.pop('STRATA_PERF131K_BLOB_DIAG',None)
        if on:env['STRATA_PERF131K_BLOB_DIAG']='1'
        if env_value is not None:env['STRATA_PERF131K_BLOB_DIAG']=env_value
        with (self.base/(name+'.txt')).open('xb') as out:
            code=subprocess.run([str(self.exe),mode],env=env,cwd=self.base,stdout=out,stderr=subprocess.STDOUT,timeout=5).returncode
        self.assertEqual(code,0)
        raw=(self.base/(name+'.txt')).read_text(encoding='utf-8',errors='replace')
        records=[json.loads(line[len('BLOB_DIAG '):]) for line in raw.splitlines() if line.startswith('BLOB_DIAG ')]
        state=next(line for line in raw.splitlines() if line.startswith('STATE '))
        return records,state,raw

    def test_exact_opt_in_and_success_path(self):
        for value in ('', '0', 'true', '01', '1x'):
            with self.subTest(value=value):
                records,_,_=self.run_case('issue',False,env_value=value)
                self.assertEqual(records,[])
        for on in (True,False):
            records,state,_=self.run_case('normal',on)
            self.assertEqual(records,[])
            self.assertIn('ok=1 issued=2 waited=2 error_reads=2',state)

    def test_direct_error_is_recorded_before_later_api_overwrites_it(self):
        records,state,_=self.run_case('issue')
        self.assertEqual(len(records),1)
        event=next(r for r in records[0]['records'] if r['stage']=='ReadFile')
        self.assertEqual(event['code'],5)
        self.assertEqual(event['requested_bytes'],4096)
        self.assertIn('issued=1 waited=1',state)
        self.assertIn('order=alloc,event,event,read,read,wait,',state)

    def test_off_has_no_diagnostic_output_and_preserves_io_order(self):
        on,on_state,_=self.run_case('issue');off,off_state,_=self.run_case('issue',False)
        self.assertEqual(off,[])
        self.assertEqual(on_state,off_state) # ReadFile already reads GetLastError in the original code.
        self.assertEqual(len(on),1)

    def test_direct_failure_stages_and_drain(self):
        for mode,stage,code in [('alloc','VirtualAlloc',1455),('event','CreateEventW',8),
                                ('wait','GetOverlappedResult',23),('short','short_read',None)]:
            with self.subTest(mode=mode):
                records,state,_=self.run_case(mode)
                event=next(r for r in records[0]['records'] if r['stage']==stage)
                self.assertEqual(event['code'],code)
                if mode in ('wait','short'):self.assertIn('issued=2 waited=2',state)
                self.assertGreater(event['requested_bytes'],0)

    def test_profile_consumer_and_stage_null_branches(self):
        for mode,stage in [('claim','stage_claim_failed'),('key','stage_key_changed'),('fill','stage_fill_failed')]:
            with self.subTest(mode=mode):
                records,state,_=self.run_case(mode)
                self.assertEqual(len(records),1)
                self.assertEqual(records[0]['terminal']['source'],'file')
                self.assertEqual(records[0]['terminal']['layer'],0)
                self.assertTrue(any(r['stage']==stage and r['code'] is None for r in records[0]['records']))
                self.assertIn('transfers=0',state)

    def test_pin_failure_can_recover_and_heap_failure_is_unknown(self):
        records,state,_=self.run_case('pin')
        self.assertEqual(records,[]) # Pin failure recovered through the original pageable fallback.
        self.assertIn('ok=1',state)
        records,state,_=self.run_case('heap')
        event=next(r for r in records[0]['records'] if r['stage']=='new_nothrow')
        self.assertIsNone(event['code']);self.assertEqual(event['requested_bytes'],16)
        self.assertIn('ok=0',state)

    def test_buffer_output_and_drop_count_are_bounded(self):
        records,_,raw=self.run_case('flood');d=records[0]
        self.assertEqual(d['capacity'],32);self.assertEqual(d['event_capacity'],31)
        self.assertEqual(len(d['records']),31);self.assertEqual(d['dropped_at_snapshot'],69)
        self.assertEqual(d['status'],'SNAPSHOT');self.assertEqual(d['later_events'],'unknown')
        self.assertLess(len(raw),32768)

    def test_parallel_records_have_unique_slots_and_unknown_context(self):
        records,_,_=self.run_case('threads');d=records[0]
        self.assertEqual(len(d['records']),31);self.assertEqual(d['dropped_at_snapshot'],49)
        self.assertEqual(len({r['sequence'] for r in d['records']}),31)
        self.assertTrue(all(r['request'] is None and r['layer'] is None and not r['same_terminal_request']
                            for r in d['records']))


if __name__=='__main__':unittest.main()
