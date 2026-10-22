from pathlib import Path
import sqlite3,json,hashlib
root=Path.cwd();out=(root/'../Strata-data/dflash2-projection-streams-20261022').resolve()
c=sqlite3.connect(out/'projection-trace.sqlite')
q="select distinct k.graphId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where s.value like '%conv_kernel%' and s.value like '%strata%' and k.graphId>0"
summary={'trace_sha256':hashlib.sha256((out/'projection-trace.nsys-rep').read_bytes()).hexdigest(),'graphs':[]}
for (gid,) in c.execute(q):
 rows=c.execute('select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.graphId=? order by k.start',(gid,)).fetchall()
 streams=sorted(set(r[2] for r in rows));main=streams[0]
 branch=[r for r in rows if r[2]!=main]
 projections=[r for r in rows if 'mmvq' in r[3]]
 examples=[];overlap_count=0
 for a in branch:
  others=[b for b in projections if b[2]!=a[2] and b[0]<a[1] and b[1]>a[0]]
  if others:
   overlap_count+=1
   if len(examples)<3:
    b=others[0];examples.append({'streams':[a[2],b[2]],'overlap_ns':min(a[1],b[1])-max(a[0],b[0]),'kernels':[a[3],b[3]]})
 summary['graphs'].append({'graph_id':gid,'streams':streams,'branch_kernel_count':len(branch),'branch_kernels_overlapping_another_projection':overlap_count,'examples':examples})
summary['successful_graph_instantiations']=c.execute("select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like 'cudaGraphInstantiate%' and r.returnValue=0").fetchone()[0]
assert any(g['branch_kernels_overlapping_another_projection']>0 for g in summary['graphs'])
assert any(len(g['streams'])==1 for g in summary['graphs']) # reference request's serial graphs
(Path(__file__).resolve().parent/'trace-summary.json').write_text(json.dumps(summary,indent=2))
print('PASS: real graph branches and projection kernel overlap; reference graphs remain serial')
print([(g['graph_id'],g['branch_kernel_count'],g['branch_kernels_overlapping_another_projection']) for g in summary['graphs']])
