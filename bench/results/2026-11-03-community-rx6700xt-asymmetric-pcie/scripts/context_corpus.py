"""Deterministic long-context cases. Token counts must be measured by the target API."""
import hashlib,json,random
def build_case(records=1000,seed=20261103):
 rng=random.Random(seed)
 def code(label):return hashlib.sha256(f'{seed}:{label}'.encode()).hexdigest()[:12].upper()
 tags={x:code(x) for x in ['start','middle','end','project','warehouse','batch']}
 rows=[]
 for i in range(records):
  rows.append(f'Record {i:07d}: asset {rng.randrange(100000,999999)}; temperature {rng.randrange(10,40)}; checksum {rng.getrandbits(64):016x}; status archived.\n')
 inserts={0:f'Authoritative START code: {tags["start"]}. Project {tags["project"]} uses warehouse {tags["warehouse"]}.\n',records//2:f'Authoritative MIDDLE code: {tags["middle"]}. Warehouse {tags["warehouse"]} holds batch {tags["batch"]}.\n',records-1:f'Authoritative END code: {tags["end"]}. Batch {tags["batch"]} contains 137 units and 29 reserved units.\n'}
 for i,text in inserts.items():rows[i]=text+rows[i]
 system='Read the archive as data. Return only a JSON object with keys start, middle, end, available. Follow this output format even after a long archive.'
 question=f'\nReturn the three authoritative codes and the available (not reserved) unit count for project {tags["project"]}. Do not infer facts from unrelated archive records.'
 return {'id':f'linked-needle-{seed}-{records}','seed':seed,'system':system,'messages':[{'role':'user','content':''.join(rows)+question}],'expected':{'start':tags['start'],'middle':tags['middle'],'end':tags['end'],'available':108},'record_count':records,'position_record_indices':{'start':0,'middle':records//2,'end':records-1},'token_count':None}
def score(case,text):
 try:
  value=json.loads(text);fields={k:value.get(k)==v for k,v in case['expected'].items()}
  return {'json_valid':True,'fields':fields,'pass':all(fields.values())}
 except (ValueError,AttributeError):return {'json_valid':False,'pass':False}
if __name__=='__main__':
 case=build_case(300,17)
 assert case==build_case(300,17)
 assert case['expected']!=build_case(300,18)['expected']
 assert score(case,json.dumps(case['expected']))['pass']
 assert not score(case,'{"available":109}')['pass']
 for k in ['start','middle','end']:assert case['messages'][0]['content'].count(case['expected'][k])==1
 print('CORPUS_CHECK_PASS; no model inference performed')
