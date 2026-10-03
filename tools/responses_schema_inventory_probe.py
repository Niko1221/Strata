"""Generate every catalog sample through a real Responses JSON-schema server.

The prompt requests a known valid sample: this tests schema admission, generation,
serialization and final validation, not model coding quality or adversarial schema
enforcement. Other native matcher/negative validation tests cover illegal output.
"""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from jsonschema import Draft202012Validator, FormatChecker
from responses_prompt_examples import attempts, schema_request, schema_examples, request_with_examples, prompt_receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--catalog',type=Path,default=Path(__file__).resolve().parents[1]/'docs/json-schema-examples.json')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--start-case',type=int,default=1,help='1-based catalog case to start at (diagnostic runs)')
    parser.add_argument('--examples',type=int,choices=(0,1),default=0,help='each schema has one explicitly known-value demonstration')
    parser.add_argument('--hint-on-failure',action='store_true')
    parser.add_argument('--reasoning',choices=('none','medium'),default='none')
    args=parser.parse_args()
    counts=attempts(args.examples,args.hint_on_failure,available=1)
    args.out.mkdir(exist_ok=False,parents=True)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report={'scripted_engine':False,'prompt_requests_known_valid_sample':True,'reasoning':args.reasoning,
            'hint_on_failure':args.hint_on_failure,'maximum_examples':1,'cases':[],'result':'running'}
    try:
        for case in json.loads(args.catalog.read_text(encoding='utf-8'))[args.start_case-1:]:
            name,schema,sample=case['name'],case['schema'],case['sample']
            Draft202012Validator.check_schema(schema)
            validator=Draft202012Validator(schema,format_checker=FormatChecker())
            validator.validate(sample)
            row={'name':name,'attempts':[]}
            report['cases'].append(row)
            for count in counts:
                body=request_with_examples(schema_request(case,args.model,args.reasoning),schema_examples(case),count)
                prefix=name+'.examples-'+str(count)
                (args.out/(prefix+'.request.json')).write_text(json.dumps(body,indent=2,ensure_ascii=False),encoding='utf-8')
                request=urllib.request.Request(args.base_url.rstrip('/')+'/responses',data=json.dumps(body).encode(),
                    headers={'Authorization':'Bearer '+os.environ['STRATA_API_KEY'],'Content-Type':'application/json'})
                trial=prompt_receipt(body,count);row['attempts'].append(trial)
                started=time.monotonic()
                try:
                    response=opener.open(request,timeout=180)
                except urllib.error.HTTPError as error:
                    response=error
                with response:
                    status=response.status
                    result=json.load(response)
                (args.out/(prefix+'.response.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
                trial.update(http_status=status,status=result.get('status'),seconds=time.monotonic()-started)
                try:
                    assert status==200 and result['status']=='completed',result.get('error')
                    value=json.loads(''.join(p['text'] for item in result['output'] if item['type']=='message' for p in item['content']))
                    validator.validate(value)
                    assert value==sample,(value,sample)
                    trial.update(schema_valid=True,sample_matches=True,passed=True)
                except Exception as error:
                    trial.update(failure=str(error),passed=False)
                print(name,'examples='+str(count),'PASS' if trial['passed'] else 'FAIL',round(trial['seconds'],3),trial.get('failure',''),flush=True)
                if trial['passed']:
                    row['passed_with_examples']=count
                    break
            row['baseline_passed']=row['attempts'][0]['passed'] if row['attempts'][0]['example_count']==0 else None
        report['baseline_passes']=sum(r['baseline_passed'] is True for r in report['cases'])
        report['hinted_passes']=sum(r.get('passed_with_examples',0)>0 for r in report['cases'])
        report['result']='pass' if all('passed_with_examples' in row for row in report['cases']) else 'fail'
    except Exception as error:
        report['result']='fail';report['failure']=str(error)
        raise
    finally:
        (args.out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if report['result']!='pass':
        raise SystemExit(1)


if __name__=='__main__':main()
