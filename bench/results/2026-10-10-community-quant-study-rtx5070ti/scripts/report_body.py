"""Shared report narrative/tables: all models inline, coherent recommendations.
Executed by build_revision.py with its already-extracted metrics and chart helpers.
"""
import os
import tempfile
H=[]; TABLE_COUNT=0; TABLE_META=[]
def w(s):H.append(s)
def section(title,color='#0b3d5c'):w(f'<h2 class="sec" style="background:{color}">{title}</h2>')
def how(s):w(f'<div class="how">{s}</div>')
def verdict(s):w(f'<div class="verdict">{s}</div>')
def call(s):w(f'<div class="callout">{s}</div>')
def figure(fn,caption,big=False,tall=False):
 classes='figure'+(' big' if big else '')+(' tall' if tall else '')
 w(f'<figure class="{classes}"><img src="charts/{fn}"><figcaption>{caption}</figcaption></figure>')
def table(headers,rows,title='Results'):
 global TABLE_COUNT
 TABLE_COUNT+=1; tid=f'T{TABLE_COUNT:02d}'
 TABLE_META.append({'id':tid,'title':title,'rows':len(rows),'headers':headers})
 w(f'<table data-table-id="{tid}"><caption><b>{tid} · {escape(title)}</b></caption><thead><tr>'+''.join(f'<th>{s}</th>' for s in headers)+'</tr></thead><tbody>')
 for row in rows:
  hosted=any(any(name in str(cell) for name in HOSTED_NAMES) for cell in row)
  w('<tr'+(' class="hosted-reference"' if hosted else '')+'>'+''.join(f'<td>{s}</td>' for s in row)+'</tr>')
 w(f'</tbody><tfoot><tr><td colspan="{len(headers)}">{tid} · {len(rows)} rows · end of table</td></tr></tfoot></table>')
def links(items):w('<div class="links">'+''.join(f'<a href="{u}">{escape(l)}</a>' for l,u in items)+'</div>')
def bullet(items):w('<ul>'+''.join(f'<li>{s}</li>' for s in items)+'</ul>')
def card(title,body,warn=''):return f'<div class="card"><h4>{title}</h4>{body}'+(f'<div class="warn">{warn}</div>' if warn else '')+'</div>'
def eng(k):return {'stock':'0.1.40.3','rc1':'rc1','rc2':'rc2'}[M[k]['cohort']]
def metric_eng(k):return ('rc2' if k=='gyro-s' else eng(k))
def arm(k,e):return name(k)+' '+escape(e)
def ranktag(k,e):
 rank=rankmap.get((k,e))
 return f'#{rank}' if rank is not None else 'unranked'
def note(k):return 'experimental fork · rc2' if k in ['gyro-s','gyro-m'] else 'reference'
HOSTED_NAME_BY_KEY={
 'hosted-reference-luna':'<b>GPT-6-luna MAX</b> <span class="lab" style="color:#7B2CBF">OpenAI · hosted API reference</span>',
 'hosted-reference-sol':'<b>GPT-6.1-SOL ultra</b> <span class="lab" style="color:#9B59B6">OpenAI · hosted Codex reference</span>',
}
HOSTED_ENGINE_BY_KEY={h['key']:h['engine'] for h in HOSTED_REFERENCES}
HOSTED_LABEL_BY_KEY={h['key']:h['label'] for h in HOSTED_REFERENCES}
def hosted_needle_cell(hosted):
 data=hosted['ref']['sampled_knowledge_and_recall']['needle']
 if hosted['key']=='hosted-reference-sol':return f'{f(data["accuracy_percent"],0)} @ {data["actual_max_input_tokens"]:,} actual'
 return f'{f(data["accuracy_percent"],0)} @ {data["effective_context_tokens"]:,} effective'
def model_order(keys,value):
 return sorted(keys,key=lambda k:(M[k]['cohort']!='stock',-value(k)))
def arm_order(rr,value,complete=False):
 return sorted(rr,key=lambda x:(M[x[0]]['cohort']!='stock',not x[2]['audit']['complete_scoring'] if complete else False,-value(x[2])))
lo=G['heplus']['low'];hi=G['heplus']['xhigh'];tl=G['trials']['low'];th=G['trials']['xhigh']
alltools=[(k,e,t) for k in ORDER for e,t in (M[k].get('trials')or{}).items()]
GMG=M['gyro-m']['trials'];assert all(isinstance(GMG[e].get('raw_safety_gate_passed'),bool) and isinstance(GMG[e].get('raw_safety_gate_warnings'),list) for e in ['low','xhigh'])
GM_GATE_SUMMARY='; '.join(f'{e}: {"passed" if GMG[e]["raw_safety_gate_passed"] else "failed"} ({len(GMG[e]["raw_safety_gate_warnings"])} raw warnings)' for e in ['low','xhigh'])
stock_he=[(k,e,r) for k,e,r in FULL if M[k]['cohort']=='stock']
CODE_PICK=max(stock_he,key=lambda x:(x[2]['score'],x[2]['first_attempt_score']))
DEBUG_PICK=max(STOCK,key=lambda k:int(M[k]['fivebugs']['hid'].split('/')[0]))
RECS=[dict(r) for r in RECOMMENDATIONS]
claims['recommendations']=RECS
(BASE/'claims.json').write_text(json.dumps(claims,indent=2)+'\n')
css=(BASE/'scripts/style.css').read_text()
w('<!doctype html><html><head><meta charset="utf-8"><title>Qwen3.8-Flash-Next quant comparison · hosted quality references</title><style>'+css+'</style></head><body>')
w(f'<div class="hero"><h1>Qwen3.8-Flash-Next:<br>which quant should YOU run?</h1><div class="sub">{D["inventory"]["models"]} local files · 3 quant/finetune labs · {D["inventory"]["effort_arms"]} effort arms · {D["inventory"]["tool_arms"]} five-run tool arms · one measured RTX 5070 Ti (16 GB).</div><div class="chips"><span class="chip">Ryzen 9800X3D · 64 GB RAM · Windows + WSL2</span><span class="chip">Oct 7–13, 2026 local study</span><span class="chip">Engine column: Strata 0.1.40.3 / AgentionAI rc1 / rc2</span><span class="chip">Blue = ISTA-DASLab · Orange = UkisAI · Green = AgentionAI</span><span class="chip">GPT-6-luna MAX (OpenAI) · hosted reference</span><span class="chip">GPT-6.1-SOL ultra (OpenAI) · hosted reference</span><span class="chip">PR #1850 · revised {D["revision"]["date"]}</span></div></div>')
section('IN A HURRY? Choose a daily driver, not a benchmark trophy')
bullet([f'<b>Recommended default:</b> {hlabel("swift-iq3_s")} low on the stock study setup. This is a reasoned choice across the full evidence matrix, not a master-score winner or a command to change the deployed model.',f'<b>Broad capability:</b> Swift IQ3_S low scored 157/164 coding, first 92.7%, in {wall(M["swift-iq3_s"]["heplus"]["low"]["total_time"])}; tool mean 88.2 and all-five floor 76.1. Its per-file samples scored math 98%, algebra 85%, instructions 84.0/89.5% prompt/instruction, hidden debugging 20/25, and retrieval 100% at 260,096 effective tokens.','<b>Choose the trade-off:</b> Swift Q2_0 low is the fastest total workload (55m54s code+debug+tool) and has the best stock tool floor, with lower sampled knowledge/debug scores; Swift IQ3_XXS low completes coding fastest (62m31s total); Q-IQ3_S xhigh has the highest observed stock final coding score but takes far longer (~160m). The recommended default is mid-pack on speed (75m03s), chosen for breadth.','<b>Fork results stay visible, not silently promoted:</b> Gyro-S rc2 has valid coding/tools, math 96%, algebra 79%, IFEval 76.0/82.9%, debugging 21/25 visible and 16/25 hidden, and 100% retrieval at 129,024 tokens. Gyro-M has narrow memory headroom and campaign/safety caveats. Neither is the default recommendation.','<b>All testing contributes:</b> coding/repair, five-run tools and hard chains, instruction following, sampled math/algebra, visible/hidden debugging, context retrieval, actual task times and deployment evidence. TPS is separate diagnostics, not a daily-driver score.','<b>Confidence is limited:</b> small sampled suites and one rig do not establish universal model superiority, multi-day stability, comprehensive safety or a model-only causal difference. No fabricated missing values and no arbitrary master score.'])
section('THE FOUR QUESTIONS — practical trade-offs','#a1662f')
w('<div class="cards">'+card('Which default?',hlabel('swift-iq3_s')+' low: broad stock-cohort capability with a short complete coding job.','Benchmark-backed recommendation, not long-term stability certification.')+card('Which specialist?',hlabel('swift-q2_0')+' low for the tool floor and fastest total wall (56m); '+hlabel('swift-iq3_xxs')+' low for coding latency.','A specialist advantage is not a universal default win.')+card('Why not crown the fork?', 'Gyro-S offers a strong tool floor but lower observed algebra/instruction/debug scores and shorter tested recall capacity than the stock default. Gyro-M has narrow 16 GB margin, failed raw gates and campaign caveats.','Capability and deployment suitability are separate decisions.')+card('How was the choice made?','The final matrix displays all test families, total wall time and limitations; the default rationale is explicit.','No weights, numeric overall rank or invented master score. Speed is shown as total wall, not folded into a score.')+'</div>')
links([('PR #1850',D['revision']['pr'])])
section('THE CAST — ten files, one base model, three lineages','#0072B2')
how('All local files share Qwen3.8-Flash-Next ancestry (Qwen / Alibaba). Q- labels: ISTA-DASLab quants; S- labels: UkisAI Swift derivatives; Gyro: AgentionAI APR. File sizes are decimal GB, not VRAM requirements. The two OpenAI-hosted references — GPT-6-luna MAX via Responses API and GPT-6.1-SOL ultra via Codex CLI — appear only beside comparable quality suites. Neither is a local file, local run, or recommendation.')
table(['File / lab','Quant','File GB','Engine','Status','Download'],[[name(k),M[k]['quant'],f(M[k]['size_gb']),eng(k),note(k),f'<a href="{SRC[M[k]["lab"]]}">HF repo</a>'] for k in ORDER],title='Model provenance and engine cohorts')
call('The eight reference files use stock Strata 0.1.40.3; Gyro-S and Gyro-M use separate AgentionAI rc2 setups. Shared evaluator and prompts do not remove the engine/configuration variable. Complete cross-engine setup ranks are descriptive and do not prove model-only causality.')
section('THE SCORECARD — all models in one view','#14927a')
how('Tools: low-effort mean of five runs and empirical all-five floor. Coding: final percentage plus the actual solved/scored count. All local coding arms now have complete 164-task scoring. Knowledge/debugging are limited once-per-file tests; Gyro-S valid rc2 math, algebra, IFEval, Five Bugs and Needle measurements are included. The separate scoreboard keeps all 20 effort rows visible and shows all 20 effort setups with all test families and an explicit daily-driver recommendation, without an overall numeric rank.')
kk=model_order(ORDER,lambda k:(M[k].get('trials')or{}).get('low',{}).get('pass_hat_5',-1))
floors=[M[k]['trials']['low']['pass_hat_5'] for k in STOCK]
rows=[]
for k in kk:
 m=M[k];t=(m.get('trials')or{}).get('low');a=m['heplus']['low'];b=m['heplus']['xhigh'];aux=''
 rows.append([name(k),eng(k),f'{f(t["mean"])} +/- {f(t["std"])}' if t else 'n/a',f(t['pass_hat_5'])+(' '+medal(t['pass_hat_5'],floors) if k in STOCK else '') if t else 'n/a',f'{f(a["score"])} ({a["passed"]}/{a["scored_total"]})'+('*' if not a['audit']['complete_scoring'] else ''),f'{f(b["score"])} ({b["passed"]}/{b["scored_total"]})'+('*' if not b['audit']['complete_scoring'] else ''),f'{f(m["knowledge"]["gsm8k"]["accuracy"],0)}{aux}',f'{f(m["knowledge"]["mmlu"]["accuracy"],0)}{aux}',m['fivebugs']['hid']+aux])
table(['File / lab','Engine','Tool mean','Pass^5','HE+ low','HE+ xhigh','Math','Algebra','Debug'],rows,title='Integrated headline scorecard')
w('<div class="small">Medals show the top three distinct stock-reference score levels, ties share. Fork rows do not receive stock medals. Gyro-S rows use current rc2 measurements. Its algebra cell shows the valid rc2 79% (first 100 abstract-algebra questions, 5-shot); old rc1 algebra 76% is archived and not substituted. n/a is an intentional unavailable measurement, not zero.</div>')
call('External quality anchors, not picks: GPT-6-luna MAX and GPT-6.1-SOL ultra (OpenAI-hosted) appear as score-ordered reference rows/bars in comparable suite views, with no medal decoration. Neither ran on the RTX 5070 Ti; both are excluded from local counts, ranks, recommendations and the OVERALL composite.')
figure('F2-wall.png','Low-effort observed time-to-quality. All local points have complete scoring; green points use separate fork engines. No model-only causal speedup is claimed.',True)
section('RELIABILITY & SAFETY — averages are not guarantees','#b9770e')
how(f'Tool-eval-bench: {tl["n_scenarios"]} realistic scenarios x five identical-seed runs at each effort. Score 0 / 1 / 2; 2 = full marks. Pass@5: full marks at least once, the empirical capability ceiling. Pass^5: full marks in ALL five, the reliability floor. Partial credit every run never passes the floor. These summaries are not deployment guarantees. The table is ordered by Pass^5 floor (descending); mean breaks ties.')
figure('F3-reliability.png','Local files and both hosted references are position-sorted by Pass^5. The line joins mean and floor; engine/source labels preserve local versus hosted provenance. The table gap is Pass@5 minus Pass^5.')
floorvals=[t['pass_hat_5'] for _,_,t in stocktools];meanvals=[t['mean'] for _,_,t in stocktools]
tool_entries=alltools+[(h['key'],'hosted',h['ref']['tool_eval_bench']) for h in HOSTED_REFERENCES]
def hosted_pass_floor(tool):return tool.get('pass_hat_5_percent',tool.get('pass_all5_percent'))
tool_entries.sort(key=lambda x:(-(hosted_pass_floor(x[2]) if x[0] in HOSTED_BY_KEY else x[2]['pass_hat_5']),-(x[2]['mean'] if x[0] in HOSTED_BY_KEY else x[2]['mean'])))
tool_rows=[]
for k,e,t in tool_entries:
 if k in HOSTED_BY_KEY:
  hosted=HOSTED_BY_KEY[k];hr=hosted['ref']['tool_eval_bench'];floor=hosted_pass_floor(hr)
  tool_rows.append([HOSTED_NAME_BY_KEY[k],f'{f(hr["mean"])} +/- {f(hr["stddev"])}',f(hr['pass_at_5_percent']),f(floor),f(hr['reliability_gap_percent']),str(hr['final_trial_safety_warnings']),HOSTED_ENGINE_BY_KEY[k]])
 else:tool_rows.append([name(k)+' '+e,f'{f(t["mean"])} +/- {f(t["std"])}'+(' '+medal(t['mean'],meanvals) if k in STOCK else ''),f(t['pass_at_5']),f(t['pass_hat_5'])+(' '+medal(t['pass_hat_5'],floorvals) if k in STOCK else ''),f(t['gap']),str(t['n_warnings']),eng(k)])
table(['Arm / lab','Mean +/- SD','Pass@5','Pass^5 floor','Ceiling-floor gap','Warnings (last trial)','Engine'],tool_rows,title='Integrated tool reliability — sorted by Pass^5 floor')
w('<div class="small">Both hosted rows are distinct external runs: Luna MAX through the Responses API and SOL ultra through Codex CLI. Confidence intervals are not shown in this common table (SOL publishes one; Luna does not). The SOL tool protocol used concurrency 4 versus the study’s 1 and a JSON-schema bridge; trial-2 TC-56 timeout recovery is disclosed in the source detail. Both rows stay outside the local recommendation cohort. Warning counts describe each final primary trial.</div>')
countcap=sum('safety-capped' in t['rating'] for _,_,t in stocktools)
w(f'<div class="small">{countcap}/{len(stocktools)} stock arms are capped on the stored last-trial rating. Gyro-M raw safety_gate.passed: {GM_GATE_SUMMARY}. This authoritative gate is distinct from stored rating labels. Not capped is NOT the same as no warnings or production safety. Warning lists/rating describe the final primary trial, not a five-run union.</div>')
figure('F9-decomposition.png','The all-five floor is the green count / 92. Rows are ordered by that floor; both hosted references are interleaved at their measured scores and remain outside the local scorecard and composite.',True)
verdict(f'Gyro-M illustrates the trade-off: always-full scenarios {tl["decomposition"]["solid"]} low / {th["decomposition"]["solid"]} xhigh; flaky {tl["decomposition"]["flaky"]} / {th["decomposition"]["flaky"]}. Its mean confidence intervals overlap, so the observed mean gain is modest. No causal mechanism is inferred from the aggregate alone.')
links([('Tool-eval-bench','https://github.com/SeraphimSerapis/tool-eval-bench')])
section('AGENTIC TOOL USE — hard chains reward extra thought','#4a5fc1')
how('Hard Mode (P) stresses long chains and recovery. Every category figure uses the five-run mean, not the last-trial score. Effort comparisons stay within a measured file/build. The table orders local arms and separate hosted references by Hard Mode mean; hosted protocols are not identical.')
figure('F4-hardmode.png','Local arms and both hosted references are shown as score-ordered bars. Hosted runs are external observations, not local models.',True)
hard_entries=alltools+[(h['key'],'hosted',h['ref']['tool_eval_bench']) for h in HOSTED_REFERENCES]
hard_entries.sort(key=lambda x:-(x[2]['hard_mode_mean_percent'] if x[0] in HOSTED_BY_KEY else x[2]['cats']['P']))
pvals=[t['cats']['P'] for _,_,t in stocktools]
hard_rows=[]
for k,e,t in hard_entries:
 if k in HOSTED_BY_KEY:
  hosted=HOSTED_BY_KEY[k];hr=hosted['ref']['tool_eval_bench'];detail=hosted['detail']['metrics']['tool_eval_bench']
  turn_budget=detail.get('max_turns',detail.get('hard_mode_max_turns'))
  hard_rows.append([HOSTED_NAME_BY_KEY[k],f(hr['hard_mode_mean_percent']),f(hr['hard_mode_stddev_percent']),str(turn_budget),HOSTED_ENGINE_BY_KEY[k]])
 else:hard_rows.append([name(k)+' '+e,f(t['cats']['P'])+(' '+medal(t['cats']['P'],pvals) if k in STOCK else ''),f(t['cats_std']['P']),str(t['config']['max_turns']),eng(k)])
table(['Arm / lab','Hard Mode mean','SD across five','Turn budget','Engine'],hard_rows,title='Multi-step tool-chain results — sorted by Hard Mode mean')
bullet([f'Gyro-M Hard Mode rises from {f(tl["cats"]["P"])}% to {f(th["cats"]["P"])}%, while its all-five floor falls. Optimizing only one of those axes is not a reliable workload decision.','Warnings include premature notifications, missed dependency ordering and claims that an invoice is paid while funds remain held. They are scripted failure traps, not permissions.','rc1 forced tool calls were ignored with Gyro-S. Both Gyro-S and Gyro-M now have scored low/xhigh tool arms on the pinned rc2 engine; these remain engine-qualified setup results, not model-only causal claims.'])
section('CODE — every denominator visible','#0b5c8a')
how('164 function-writing tasks via BenchKit / EvalPlus, greedy generation, one verifier-guided repair. First = one-shot score; final includes the replacement. Scored timeout/length/blank outputs remain failures. Excluded harness errors can reduce the denominator; those rows retain observed percentages but are qualified and excluded from full-coverage/composite claims. The table orders complete-coverage runs by final score.')
figure('F1-coding.png','All local rows have complete scoring and are score-sorted. Both hosted quality references are positioned by final score.',tall=True)
rr=[(k,e,M[k]['heplus'][e]) for k in ORDER for e in ['low','xhigh']]
rr=arm_order(rr,lambda r:r['score'],True)
good=[r['score'] for k,e,r in rr if k in STOCK and r['audit']['complete_scoring']];walls=[r['total_time'] for k,e,r in rr if k in STOCK and r['audit']['complete_scoring']]
he_entries=[(k,e,r,None) for k,e,r in rr]+[(h['key'],'hosted',h['ref']['human_eval_plus'],h) for h in HOSTED_REFERENCES]
def hosted_task_total(metrics):return metrics.get('task_total',len(metrics.get('tasks',[])))
def _he_order_key(x):
 k,e,r,hosted=x
 complete=(r.get('infra_excluded',r.get('harness_errors',0))==0 and hosted_task_total(r)==164) if hosted else r['audit']['complete_scoring']
 return (not complete,-(r['final_percent'] if hosted else r['score']))
he_entries.sort(key=_he_order_key)
he_rows=[]
for k,e,r,hosted in he_entries:
 if hosted:
  total=hosted_task_total(r);excluded=r.get('infra_excluded',r.get('harness_errors',0))
  he_rows.append([HOSTED_NAME_BY_KEY[k],f(r['first_percent']),f(r['final_percent']),f'{r["final_passes"]}/{total}',f'{r["repair_successes"]}/{r["repair_attempted"]}',str(excluded),HOSTED_ENGINE_BY_KEY[k]])
 else:he_rows.append([name(k)+' '+e,f(r['first_attempt_score']),f(r['score'])+('*' if not r['audit']['complete_scoring'] else (' '+medal(r['score'],good) if k in STOCK else '')),f'{r["passed"]}/{r["scored_total"]}',f'{r["repair_successes"]}/{r["repair_attempted"]}',str(r.get('harness_errors',0)),eng(k)])
table(['Arm / lab','First %','Final %','Solved / scored','Repairs','Infra / harness excluded','Run / protocol'],he_rows,title='Integrated coding results — sorted by final score')
call('Hosted coding rows report scores only: wall time and token-rate columns are omitted because they are not comparable across the local engine, Luna Responses API and SOL Codex CLI. Luna’s provider-reported 42 tok/s includes hidden reasoning and is not visible-text throughput; neither hosted result supplies a matched decode measurement.')
call('GPT-6.1-SOL source: [first upload, commit a48292f](https://github.com/Jhacarreiro/Strata/commit/a48292f5766f66efcdb1e23396569b0907c44c04). Only that upload’s independent-run scores are used; the later feedback-assisted Five Bugs validation is excluded.')
call('Retrospective coverage correction: Q-IQ3_XXS low 155/163 is superseded by offline adjudication of its model SyntaxError failure: 155/164 (94.5%), first 92.1%, measured wall 865.8 s unchanged. Swift IQ3_S low 157/163 is superseded by the complete matched 2026-10-13 run: 157/164 (95.7%), first 92.7%, wall 831.7 s; no retry splicing. Original throughput aggregates retain 99.2% and 94.4% coverage respectively, not full-task decode timing. Current Gyro-S rc2 coding is 157/164 at low and xhigh with complete scoring. Historical Gyro-S rc1 xhigh 153/162 after two HTTP 502s is superseded and not mixed into rc2. Gyro-M xhigh blank finals on HumanEval/39 and HumanEval/99 ended on length and remain scored failures; 157/164 is not a clean-response certificate.')
bullet([f'Gyro-M xhigh costs {f(hewallratio)}x low wall and {f(hetokenratio)}x tokens for one additional final pass. Hard Mode benefits do not imply a coding-efficiency benefit.',f'Gyro-M low vs Gyro-S rc2 low differs by {f(lo["first_attempt_score"]-M["gyro-s"]["heplus"]["low"]["first_attempt_score"])} points on first attempts and {f(lo["score"]-M["gyro-s"]["heplus"]["low"]["score"])} points on final score. These are distinct model revisions/hashes; this setup pair does not establish broad quant-only causality.'])
links([('EvalPlus','https://github.com/evalplus/evalplus'),('BenchKit','https://github.com/DogukanUrker/BenchKit')])
section('KNOWLEDGE & DEBUGGING — the test we actually ran','#6c3483')
knowledge_entries=[('local',k) for k in ORDER]+[('hosted',h['key']) for h in HOSTED_REFERENCES]
def knowledge_hidden(row):
 kind,k=row
 return HOSTED_BY_KEY[k]['ref']['five_bugs']['hidden_passes'] if kind=='hosted' else int(M[k]['fivebugs']['hid'].split('/')[0])
def knowledge_label(row):
 kind,k=row
 return HOSTED_BY_KEY[k]['name'] if kind=='hosted' else f'{M[k]["short"]} ({M[k]["lab"]})'
knowledge_entries.sort(key=lambda row:(-knowledge_hidden(row),row[0]=='hosted'))
top_hidden=max(knowledge_hidden(row) for row in knowledge_entries)
top_ties=', '.join(knowledge_label(row) for row in knowledge_entries if knowledge_hidden(row)==top_hidden)
how('GSM8K: 100 shuffled questions, 8-shot; SOL chose seed 42 but the source study seed was not retained, so exact question identity is unverified. MMLU: first 100, 5-shot, entirely abstract algebra — NOT a 57-subject exam. IFEval: first 50 prompts / 76 constraints. Five Bugs: five files x five independent sessions. For SOL, this table uses only the first-upload aggregate (25/25 visible, 20/25 hidden); the first score export has no per-task outcome matrix, and the later feedback-assisted validation is excluded. Needle results have different measured token bases; see the note below. These are limited samples, not a whole-model intelligence ranking. Rows are sorted by hidden Five Bugs score.')
hidden=[int(M[k]['fivebugs']['hid'].split('/')[0]) for k in STOCK]
knowledge_rows=[]
for kind,k in knowledge_entries:
 if kind=='hosted':
  hosted=HOSTED_BY_KEY[k];ref=hosted['ref'];hks=ref['sampled_knowledge_and_recall'];hfb=ref['five_bugs']
  knowledge_rows.append([HOSTED_NAME_BY_KEY[k],f(hks['gsm8k']['accuracy_percent'],0),f(hks['mmlu']['accuracy_percent'],0),f'{f(hks["ifeval"]["prompt_accuracy_percent"])} / {f(hks["ifeval"]["instruction_accuracy_percent"])}',f'{hfb["visible_passes"]}/{hfb["visible_total"]}',f'{hfb["hidden_passes"]}/{hfb["hidden_total"]}',hosted_needle_cell(hosted),HOSTED_ENGINE_BY_KEY[k]])
 else:knowledge_rows.append([name(k),f(M[k]['knowledge']['gsm8k']['accuracy'],0),f(M[k]['knowledge']['mmlu']['accuracy'],0),f'{f(M[k]["knowledge"]["ifeval"]["prompt_accuracy"])} / {f(M[k]["knowledge"]["ifeval"]["instruction_accuracy"])}',M[k]['fivebugs']['vis'],M[k]['fivebugs']['hid']+(' '+medal(int(M[k]['fivebugs']['hid'].split('/')[0]),hidden) if k in STOCK else ''),f'{f(M[k]["needle"]["retrieval_accuracy"],0)} @ {int(M[k]["needle"]["effective_context"]):,} effective',metric_eng(k)])
table(['File / lab','GSM8K /100','Algebra /100','IFEval prompt / instruction %','Bugs visible','Bugs hidden','Needle % @ measured context','Engine'],knowledge_rows,title='Integrated sampled suites — sorted by hidden debugging passes')
call(f'Top hidden Five Bugs score: {top_hidden}/25, tied among {top_ties}. {knowledge_label(knowledge_entries[0])} appears first because local rows precede hosted rows on ties; it is not a unique win.')
figure('F5-debug.png','Ten local files plus both hosted quality references; hosted bars are separate observations, not local quant results.',True)
verdict(f'Within the local stock cohort, {hlabel(DEBUG_PICK)} remains the debugging specialist at {M[DEBUG_PICK]["fivebugs"]["hid"]} hidden sessions and leads this algebra sample at {f(M[DEBUG_PICK]["knowledge"]["mmlu"]["accuracy"],0)}%. Both hosted references report 20/25 hidden checks; SOL is specifically the first-upload independent-run score, not the later corrected validation.')
call('Needle context qualification: SOL hit all 25 cells at a 260,096-token target, but the five largest inputs measured 221,595–221,877 actual tokens and used same-role history injection. Luna’s 25/25 is recorded at 260,096 effective tokens. Keep these as distinct context/protocol observations, not equal-length tests.')
links([('Five Bugs upstream','https://github.com/vyang472/five-bugs'),('TEB knowledge / needle plugins','https://github.com/SeraphimSerapis/tool-eval-bench')])
section('REASONING ECONOMICS — what you wait for','#009e73')
how('Total output tokens include reasoning, repairs and failures. Wall includes request/harness overhead. Failure time share and passing-task cost come from task-level response times, not wall subtraction. Different engine/protocol rows remain descriptive.')
figure('F6-tokens.png','Completion-token totals for all attempted tasks, all files, both efforts. All local coding arms have 164 scored tasks; throughput coverage remains separately qualified.')
rr=[(k,e,M[k]['heplus'][e]) for k in ORDER for e in ['low','xhigh']]
rr.sort(key=lambda x:(M[x[0]]['cohort']!='stock',x[2]['total_time']))
table(['Arm / lab','Output tokens','Passing tokens','Passing seconds','Failed response time %','Wall','Engine'],[[name(k)+' '+e,f(r['total_output_tokens'],0),f(r['audit']['passing_tokens'],0),f(r['audit']['passing_response_s'],0),f(r['audit']['failing_time_share_pct'])+'%',wall(r['total_time']),eng(k)] for k,e,r in rr],title='Integrated time-to-result economics')
verdict('Pick effort for the task: more deliberation can improve long tool chains while making code loops slower. The daily-driver choice considers all tested families, real task times and deployment caveats; no aggregate score replaces that judgment.')
section('SPEED — real decode; corrected first-token timing','#c0392b')
how('llama-benchy 0.4.0: prompt 2,048, generation 256, concurrency 1, three repeats/depth/effort. Raw pp_throughput in the old PDF was inflated by early SSE role chunks and is retired. Gyro-S decode values come from audited engine-log samples paired with direct-client TTFT; effective first-token rate includes queue/network/reasoning and is not kernel prefill. Gyro-M client observations are excluded from comparative speed views: the client targeted :8090, the recorded launcher used :8080, available router evidence is :8081→:8080, and no route proof or engine-side request log exists. Its exact zero-depth prompt-token count is also missing. Deep measured contexts differ: stock 131,072; rc2 122,880. Hosted rates remain withheld.')
figure('F7-decode-all.png','Attribution-eligible low-effort profiles only; Gyro-M is excluded because its client route to the pinned setup is unverified. The lightest bar is the deepest measured point; row tables state the actual depths.')
figure('F8-decode-effort.png','Attribution-eligible zero-background decode for both efforts; Gyro-M endpoint observations are excluded. A fork-engine result is not a same-engine model A/B.')
for effort in ['low','xhigh']:
 kk=model_order(ORDER,lambda k:-1 if any(r.get('comparison_eligible') is False for r in ((M[k].get('perf')or{}).get(effort)or{}).values()) else ((M[k].get('perf')or{}).get(effort)or{}).get('0',{}).get('tg_mean',-1));rows=[]
 for k in kk:
  pp=(M[k].get('perf')or{}).get(effort)
  if not pp:rows.append([name(k),eng(k),'n/a','n/a','n/a','n/a']);continue
  ps=sorted(pp.values(),key=lambda r:r['context_tokens'])
  rows.append([name(k),eng(k),' / '.join(f(r['context_tokens'],0) for r in ps),' / '.join(f(r['tg_mean']) for r in ps),' / '.join(f(r['prefill_e2e_tps'],0) for r in ps),f(ps[-1]['e2e_ttft_ms']/1000,2)])
 table(['File / lab','Engine','Context tokens: 0 / mid / deep','Decode tok/s: 0 / mid / deep','First-token rate: 0 / mid / deep','Deep TTFT s'],rows,title=f'Integrated {effort}-effort performance profiles')
call('The Gyro-M endpoint audit retains all six raw observations and their source hashes; they are shown here as raw client-endpoint values (client :8090 vs launcher :8080, no engine-side request log) and stay excluded from the decode charts. Decode TPS is diagnostic only, not a default-selection score. Gyro-S speed uses the paired engine-log decode samples and observed full-prompt token ranges; client TG/PP metrics are not promoted. See data/gyro-s-speed-audit.json and data/gyro-m-speed-attribution-audit.json. Hosted performance is not comparable and remains withheld.')
section('THE FINAL SCORECARD — comprehensive daily-driver decision matrix')
how('Every test family contributes to the decision; no master score, weights or overall numeric ranking. Display order: recommended default, other low-effort stock options, higher-effort stock options, experimental setups. Coding/tools are effort-specific. Math/algebra, IFEval, Five Bugs and Needle are once-per-file samples repeated for context, NOT independent measurements at both efforts. Gyro-S current rc2 math/algebra/instructions/debug/recall are included; archived rc1 numbers are not mixed in. Gyro-S auxiliary tests were measured once at xhigh; repeating them beside both effort rows is context, not a new low-effort measurement. Missing matched evidence never becomes zero or a pass.')
figure('F10-scoreboard.png','Cell numbers remain raw percentages, not an aggregate. Orange = weaker and teal = stronger WITHIN each test column; five color bands span that column’s observed range (printed above). Narrow score differences are visually magnified, not proof of statistical significance or absolute pass/fail. Missing data stays gray. Hidden Bugs is shown as a percentage of /25. Speed is total wall time (code+debug+5-run tool), colored inverted so faster is stronger; its labels are minutes, not percentages. No model wins by summing colors.',big=True,tall=True)
cap=[];deploy=[]
for r in SB:
 s=r['sampled'] or r['legacy_sampled']; mark='†' if r['legacy_sampled'] else ''
 cap.append([name(r['key'])+' '+r['arm'],f'{f(r["coding_first"])} / {f(r["coding_final"])}',f'{f(r["tool_mean"])}±{f(r["tool_sd"])} / {f(r["tool_floor"])}',f(r['hardmode']),f'{f(s["gsm8k"],0)} / {f(s["algebra"],0)}'+mark,f'{f(s["ifeval_prompt"])} / {f(s["ifeval_instruction"])}'+mark,s['debug_visible']+' / '+s['debug_hidden']+mark,f'{f(s["needle_accuracy"],0)} @ {int(s["needle_context_tokens"]):,}'+mark,wall(r['total_wall_s'])])
 m=M[r['key']];t=m['trials'][r['arm']];tool_seconds=t.get('wall_s') or (m.get('trials_wall_min') or {}).get(r['arm'],0)*60
 deploy.append([name(r['key'])+' '+r['arm'],wall(r['coding_wall_s']),wall(s['debug_wall_s'])+mark,wall(tool_seconds) if tool_seconds else 'not retained',f(r['coding_output_tokens'],0),rating(t)+'; '+str(r['warning_count_last_trial'])+' warnings',r['verdict']+'; '+('rc2 tested; experimental fork' if r['key']=='gyro-s' else ('205 MiB free; raw gates failed; CTest/invalid slices' if r['key']=='gyro-m' else 'stock; long-duration stability not established'))])
table(['Setup / effort','Code first/final %','Tool mean±SD / floor','Hard Mode','Math / algebra %','IFE prompt / instr %','Bugs visible / hidden','Needle % @ tokens','Speed (Total Wall)'],cap,title='All-test capability matrix — no overall score')
call('Sample sizes matter: GSM8K 100 seeded questions; algebra first 100 abstract-algebra questions, not general MMLU; IFEval 50 prompts / 76 constraints; Five Bugs 25 sessions; Needle 25 cells. Retrieval 100% at 129,024 and 260,096 tokens is not equal context capacity. Gyro-S valid rc2 auxiliary results are shown, including a scored algebra run (79%) after the original dataset HTTP 429 was resolved offline. Failure/timeout/repair outcomes remain in coding scores. Shared per-file samples do not establish effort-specific performance. <b>Speed (Total Wall)</b> sums that setup’s measured code, debugging and five-run tool job times (whole-workload wall, lower is faster); it is not a per-token rate and excludes hosted rows. The heatmap colors Speed inverted so faster reads stronger within the column.')
table(['Setup / effort','Code wall','Debug wall','5-run tool wall','Code tokens','Stored rating / warnings','Decision / deployment caveat'],deploy,title='Time-to-result, reliability and deployment decision matrix')
call('Walls are whole-job observations, not time per token; debugging/plugin times may use different harness scope from coding. Safety warnings and stored last-trial rating are not a safety certificate or five-trial warning union. Gyro-M raw safety_gate.passed is false for both efforts; Gyro-S current rc2 sampled coverage is incomplete. All stock options also lack a controlled multi-day stability evaluation. No daily-driver model swap was performed.')
section('PICK YOUR FILE — one evidence-based recommendation','#14927a')
how('Recommended default: Swift IQ3_S (UkisAI), low, on the stock study setup. The recommendation uses breadth, reliable tool behavior, instruction following, hidden debugging, recall capacity and total wall time together. It is not guaranteed by a weighted master score. The default is mid-pack on speed (75m03s); if total workload time dominates, Swift Q2_0 low is fastest (55m54s) at the cost of sampled knowledge/debug depth. Different workload priorities justify the alternatives below; experimental results do not automatically become daily defaults.')
cards=[]
for rec in RECS:
 cards.append(card(rec['decision'],hlabel(rec['key'])+' '+rec['effort'],rec['reason']))
w('<div class="cards">'+''.join(cards)+'</div>')
section('FOR MAINTAINERS — measured findings and validity','#34495e')
bullet(['<b>Tool plumbing is build-specific:</b> rc1 forced calls were ignored with Gyro-S. rc2 live forced calls passed with Gyro-M on streaming and non-streaming; Gyro-S also has scored rc2 low/xhigh TEB arms. These are engine-qualified setup results, not model-only claims.',f'<b>Gyro-S scored-slice gate:</b> overall {escape(S_RUN["scoring_slice_audit"]["overall_outcome"])}; required HE+/TEB low/xhigh slices eligible={S_GATE["eligible"]}; invalid non-scoring slices={", ".join(S_RUN["scoring_slice_audit"]["invalid_non_scoring_slices"])}; controller error retained as {escape(S_RUN["controller_error"])}. Restoration closeout eligible={S_RUN["controller_closeout_audit"]["eligible"]}; no clean overall completion is claimed.',f'<b>Experimental fork fit:</b> {C["resource_samples"]} resource samples, minimum {C["min_free_vram_mib"]} MiB free, peak {C["max_temperature_c"]} C. '+', '.join(f'{k}={v}' for k,v in C['event_maxima'].items())+'. This is not long-term stability proof.',f'<b>Campaign status:</b> <code>{C["outcome"]}</code>, finished {escape(C["finished_at"])}. CTest {C["ctest"]["passed"]} pass / {C["ctest"]["skipped"]} skip / {C["ctest"]["failed"]} fail. ple_parity lacks its required Q2_0 PLE shard/capture fixtures; separate ple_reader_selftest passed.','<b>Audit bookkeeping:</b> both HumanEval+ task audits ran; a dictionary merge overwrote intended audit-step names, making done.json mark them absent. Xhigh validity logic also missed blank finals. Raw status/scores are preserved; no silently upgraded clean pass.', '<b>Monitoring:</b> corrected busy-vs-down logic after the first aborted attempt. Perf-low/needle static-counter warnings were not accompanied by engine-event counters. Standalone effort probe is excluded from scores; possible timing interference is disclosed.', '<b>Operations:</b> production Swift IQ3_S restoration passed health/chat/tool/vision and re-enabled guards/updater. Report revisions did not alter services.'])
section('METHOD & CORRECTIONS — reproducible without hiding caveats','#5b6770')
how('The PDF, GitHub README, scoreboard and recommendation registry are generated from the same extracted store. New data is integrated throughout, not isolated in a model-specific chapter. Source records and hashes are packaged; the portable compiler rebuilds from this folder.')
bullet(['<b>Coding:</b> BenchKit direct, 164 attempted tasks, greedy temperature 0, presence penalty 1.0, one repair. Gyro-M timeout 900 seconds, no injected client token cap. Serving protocol and engine changes qualify historical cross-cohort comparisons.',f'<b>Tools:</b> {C["manifest"]["teb_version"]}, {tl["n_scenarios"]} scenarios x five trials per effort; seed {tl["config"]["seed"]}, temperature {tl["config"]["temperature"]}, timeout {f(tl["config"]["timeout_seconds"],0)} s, turn cap {tl["config"]["max_turns"]}. Separate Swift v2.8.0 / 97-scenario data is NOT mixed in.','<b>Corrections:</b> historical partial coding denominators resolved by documented offline adjudication and a complete matched replacement run; category scores changed to five-run means; inflated raw prefill removed; ceiling-minus-floor gap labeled correctly; peak temperature corrected from 64 C summary to measured 78 C. These corrections, not an unmeasured model promotion, change the earlier recommendation framing.','<b>Hosted reference protocols:</b> Both GPT-6-luna MAX and GPT-6.1-SOL ultra are external OpenAI-hosted observations, not recommendations or local runs. Luna uses the Responses API at MAX; SOL uses Codex CLI 0.162.1 at ultra with a JSON-schema tool bridge, concurrency 4, and unenforced requested sampling controls. One trial-2 TC-56 timeout was recovered through a validated original-prefix continuation; the source detail retains both original and recovery records. '+escape(REF['sampling_caveat'])+' '+escape(SOL_REF['sampling_caveat'])+' SOL’s first-upload sampling and transport caveats are recorded here; its only published Five Bugs score is 20/25 hidden and 25/25 visible from the initial independent runs. No SOL decode-performance measures exist, and matched hosted rates remain withheld.', '<b>Not tested:</b> vision quality, multi-day agents, extra seeds, multi-GPU, other GPU sizes, full knowledge datasets, additional independent sampled-suite repeats, or Gyro-M beyond 131K. No automatic supported-memory or production-stability conclusion.'])
manifest=C['manifest']
table(['Provenance item','Pinned value'],[['Stock engine digest','34CDE150B21148E6'],['Fork rc2 run',manifest['campaign_id']],['Fork engine commit','<code>'+manifest['engine_commit']+'</code>'],['Fork binary SHA256','<code>'+manifest['engine_sha256']+'</code>'],['Gyro-M model revision','<code>'+manifest['model_revision']+'</code>'],['Gyro-M model SHA256','<code>'+manifest['model_sha256']+'</code>'],['Fork context / KV',f'{manifest["context"]:,} / {manifest["kv"]}'],['Rebuild','scripts/build_revision.py; data/ALL-METRICS.json; claims.json']],title='Version-pinned provenance')
links([('PR #1850',D['revision']['pr']),('Strata','https://github.com/Niko1221/Strata'),('AgentionAI fork','https://github.com/agentionai/Strata'),('Gyro source',SRC['AgentionAI']),('Tool-eval-bench','https://github.com/SeraphimSerapis/tool-eval-bench'),('EvalPlus','https://github.com/evalplus/evalplus'),('Five Bugs','https://github.com/vyang472/five-bugs'),('GPT-6.1-SOL first upload (a48292f)',SOL_REF['source_urls']['commit'])])
w(f'<div class="small">Revised {D["revision"]["date"]} · source-qualified observations · all recommendations use the same evidence-based decision registry, without numeric overall ranks. Numeric audit, whole-table pagination proof, source manifest and portable rebuild accompany the report.</div>')
w('</body></html>')
# Keep each section heading, explainer and first real result together. This
# prevents a heading-only page while still allowing long sections to flow.
packed=[]; idx=0
while idx < len(H):
 if not H[idx].startswith('<h2'):
  packed.append(H[idx]); idx+=1; continue
 block=[H[idx]]; idx+=1
 if idx<len(H) and H[idx].startswith('<div class="how">'):
  block.append(H[idx]);idx+=1
 if idx<len(H) and H[idx].startswith('<table'):
  while idx<len(H):
   item=H[idx];block.append(item);idx+=1
   if item.endswith('</table>'):break
 elif idx<len(H) and (H[idx].startswith('<figure') or H[idx].startswith('<div class="cards">') or H[idx].startswith('<ul>')):
  block.append(H[idx]);idx+=1
 packed.append('<div class="section-open">'+''.join(block)+'</div>')
H=packed
text='\n'.join(H)
assert not any(0x1f300<=ord(c)<=0x1faff or c=='★' for c in text)
used=re.findall(r'<img src="([^"]+)"',text)
assert len(used)==10 and all((BASE/p).exists() for p in used)
assert not re.search(r'<h2[^>]*>[^<]*Gyro',text)
(BASE/'community-updated.html').write_text(text)
(BASE/'data/TABLE-INVENTORY.json').write_text(json.dumps(TABLE_META,indent=2)+'\n')
# The readable README uses the identical generated HTML content in GitHub-safe HTML.
readme=text.split('</style></head><body>',1)[1].rsplit('</body>',1)[0]
# GitHub strips CSS, so keep semantic headings/tables and conventional chart paths.
readme=re.sub(r'<div class="hero">(.*?)</div></div>',lambda x:x.group(0),readme,flags=re.S)
(BASE/'README.md').write_text(f'# Community quant battery — RTX 5070 Ti 16 GB\n\n[Download the full PDF](strata-community.pdf) · Updated {D["revision"]["date"]} · 10 local files / 20 local effort arms; hosted references are external-only.\n\n'+readme+'\n\n## Rebuild and data\n\n`python scripts/build_revision.py` rebuilds all charts, PDF and this README from `data/ALL-METRICS.json`. `scripts/audit_revision.py` checks claims. Raw HumanEval+ task files may be gzip-compressed; use Python gzip/json or `gzip -dc`. The rig-side extractor is not required to reproduce figures.\n')
pdf=BASE/'strata-community.pdf'
if os.environ.get('REPORT_SKIP_PDF')=='1':
    print('PDF_RENDER_SKIPPED: REPORT_SKIP_PDF=1; render the generated HTML separately before finalizing docs/audits')
else:
    with tempfile.TemporaryDirectory(prefix='hermes-report-chromium-',dir=os.environ.get('TMPDIR','/tmp')) as chrome_profile:
        cmd=['chromium','--headless','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--disable-extensions','--no-first-run','--no-default-browser-check','--no-sandbox','--no-pdf-header-footer','--user-data-dir='+chrome_profile,f'--print-to-pdf={pdf}',(BASE/'community-updated.html').as_uri()]
        proc=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
    print('CHROMIUM_RC',proc.returncode)
    if proc.returncode:print(proc.stderr);raise SystemExit(proc.returncode)
    assert pdf.stat().st_size>100000
    print('PDF',pdf,'bytes',pdf.stat().st_size)
    info=subprocess.run(['pdfinfo',str(pdf)],capture_output=True,text=True,check=True)
    print('\n'.join(s for s in info.stdout.splitlines() if s.startswith(('Pages:','Page size:','CreationDate:'))))
print('SECTION ORDER',re.findall(r'<h2[^>]*>(.*?)</h2>',text))
print('DAILY_DRIVER_RECOMMENDATION',POLICY['default'])
print('TABLE_COUNT',TABLE_COUNT)
exec(compile((BASE/'scripts/write_public_docs.py').read_text(),str(BASE/'scripts/write_public_docs.py'),'exec'))
