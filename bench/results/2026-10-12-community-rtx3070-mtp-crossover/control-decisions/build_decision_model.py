"""Derive the chart's policy model from measured phase timings and arrivals."""
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
results=json.loads((DATA/'results.json').read_text())
cases={(c['input_target'],c['mtp']):c for c in results['cases']}
rows=[]
for length in [49152,65536,98304,131072]:
    off,on=cases[(length,False)],cases[(length,True)]
    assert off['output_tokens']==on['output_tokens']==1000
    assert off['prompt_sha256']==on['prompt_sha256']
    oa=json.loads((DATA/f'{length}-off-arrivals.json').read_text())
    na=json.loads((DATA/f'{length}-on-arrivals.json').read_text())
    p0,p1=off['timings']['prompt_ms']/1000,on['timings']['prompt_ms']/1000
    t0,t1=off['timings']['decode_ms']/1000000,on['timings']['decode_ms']/1000000
    penalty=p1-p0
    per_token_saving=t0-t1
    assert penalty>0 and per_token_saving>0
    threshold=penalty/per_token_saving
    decisions=[]
    for n in [10,100,1000,10000,100000]:
        fits=length+n<=results['max_context']
        gain=oa[n-1]-na[n-1] if n<=1000 else n*per_token_saving-penalty
        decisions.append(dict(output_tokens=n,
                              basis='recorded arrival prefix' if n<=1000 else 'constant-rate extrapolation',
                              mtp_on_savings_seconds=gain,
                              fits_tested_context=fits,
                              mode='outside tested capacity' if not fits else 'on candidate' if gain>0 else 'off',
                              projected_off_seconds=p0+n*t0 if n>1000 else None,
                              projected_on_seconds=p1+n*t1 if n>1000 else None))
    rows.append(dict(input_tokens=length,output_tokens=1000,
                     off_seconds=off['wall_seconds'],on_seconds=on['wall_seconds'],
                     seconds_saved_using_mtp_off=on['wall_seconds']-off['wall_seconds'],
                     prefill_penalty_seconds=penalty,
                     decode_seconds_saved_using_mtp_on=per_token_saving*1000,
                     projected_break_even_tokens=threshold,
                     minimum_projected_profitable_integer_output=math.floor(threshold)+1,
                     break_even_fraction_of_input=threshold/length,
                     tested_output_capacity=results['max_context']-length,
                     projected_break_even_fits=threshold<=results['max_context']-length,
                     decisions=decisions,
                     repeated_uncached_1000_calls=[dict(calls=k,
                        seconds_saved_using_mtp_off=k*(on['wall_seconds']-off['wall_seconds'])) for k in [10,100]]))
payload=dict(assumptions='Warm engine; fresh uncached input. Long-output decode rates held constant at measured 1000-token averages. Capacity changes, cached turns and output quality are not qualified.',
             max_context=results['max_context'],rows=rows)
(DATA/'decision-model.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
print('Derived prefix timings, four projected crossings, long-output estimates and capacity flags')
