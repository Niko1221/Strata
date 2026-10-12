"""Shared report narrative/tables: all models inline, coherent recommendations.
Executed by build_revision.py with its already-extracted metrics and chart helpers.
"""
H=[]; TABLE_COUNT=0; TABLE_META=[]
def w(s):H.append(s)
def section(title,color='#0b3d5c'):w(f'<h2 class="sec" style="background:{color}">{title}</h2>')
def how(s):w(f'<div class="how">{s}</div>')
def verdict(s):w(f'<div class="verdict">{s}</div>')
def call(s):w(f'<div class="callout">{s}</div>')
def figure(fn,caption,big=False):w(f'<figure class="figure {"big" if big else ""}"><img src="charts/{fn}"><figcaption>{caption}</figcaption></figure>')
def table(headers,rows,title='Results'):
 global TABLE_COUNT
 TABLE_COUNT+=1; tid=f'T{TABLE_COUNT:02d}'
 TABLE_META.append({'id':tid,'title':title,'rows':len(rows),'headers':headers})
 w(f'<table data-table-id="{tid}"><caption><b>{tid} · {escape(title)}</b></caption><thead><tr>'+''.join(f'<th>{s}</th>' for s in headers)+'</tr></thead><tbody>')
 for row in rows:w('<tr>'+''.join(f'<td>{s}</td>' for s in row)+'</tr>')
 w(f'</tbody><tfoot><tr><td colspan="{len(headers)}">{tid} · {len(rows)} rows · end of table</td></tr></tfoot></table>')
def links(items):w('<div class="links">'+''.join(f'<a href="{u}">{escape(l)}</a>' for l,u in items)+'</div>')
def bullet(items):w('<ul>'+''.join(f'<li>{s}</li>' for s in items)+'</ul>')
def card(title,body,warn=''):return f'<div class="card"><h4>{title}</h4>{body}'+(f'<div class="warn">{warn}</div>' if warn else '')+'</div>'
def eng(k):return {'stock':'0.1.40.3','rc1':'rc1','rc2':'rc2'}[M[k]['cohort']]
def note(k):return 'experimental' if k=='gyro-m' else ('historical fork' if k=='gyro-s' else 'reference')
def model_order(keys,value):
 return sorted(keys,key=lambda k:(M[k]['cohort']!='stock',-value(k)))
def arm_order(rr,value,complete=False):
 return sorted(rr,key=lambda x:(M[x[0]]['cohort']!='stock',not x[2]['audit']['complete_scoring'] if complete else False,-value(x[2])))
lo=G['heplus']['low'];hi=G['heplus']['xhigh'];tl=G['trials']['low'];th=G['trials']['xhigh']
alltools=[(k,e,t) for k in ORDER for e,t in (M[k].get('trials')or{}).items()]
stock_he=[(k,e,r) for k,e,r in FULL if M[k]['cohort']=='stock']
CODE_PICK=max(stock_he,key=lambda x:(x[2]['score'],x[2]['first_attempt_score']))
DEBUG_PICK=max(STOCK,key=lambda k:int(M[k]['fivebugs']['hid'].split('/')[0]))
RECS=[
 {'role':'all_rounder','key':TOP['key'],'effort':TOP['arm'],'rank':TOP['rank']},
 {'role':'reliability','key':BESTFLOOR[0],'effort':BESTFLOOR[1],'rank':rankmap[(BESTFLOOR[0],BESTFLOOR[1])]},
 {'role':'fast_coding','key':FAST[0],'effort':FAST[1],'rank':rankmap[(FAST[0],FAST[1])]},
 {'role':'coding_ceiling','key':CODE_PICK[0],'effort':CODE_PICK[1],'rank':rankmap[(CODE_PICK[0],CODE_PICK[1])]},
 {'role':'knowledge_debug','key':DEBUG_PICK,'effort':'xhigh','rank':rankmap[(DEBUG_PICK,'xhigh')]},
 {'role':'experimental_not_production','key':'gyro-m','effort':'low','rank':None},
]
claims['recommendations']=RECS
(BASE/'claims.json').write_text(json.dumps(claims,indent=2)+'\n')
css=(BASE/'scripts/style.css').read_text()
w('<!doctype html><html><head><meta charset="utf-8"><title>Qwen3.8-Flash-Next community quant study — 2026-10-12</title><style>'+css+'</style></head><body>')
w(f'<div class="hero"><h1>Qwen3.8-Flash-Next:<br>which quant should YOU run?</h1><div class="sub">{D["inventory"]["models"]} files · 3 labs · {D["inventory"]["effort_arms"]} effort arms · one measured RTX 5070 Ti (16 GB).</div><div class="chips"><span class="chip">Ryzen 9800X3D · 64 GB RAM · Windows + WSL2</span><span class="chip">Oct 7–12, 2026</span><span class="chip">Engine column: Strata 0.1.40.3 / AgentionAI rc1 / rc2</span><span class="chip">Blue = ISTA-DASLab · Orange = UkisAI · Green = AgentionAI</span><span class="chip">PR #1850 · revised {D["revision"]["date"]}</span></div></div>')
section('IN A HURRY? The study in six lines')
bullet([
 f'<b>Best measured stock all-rounder:</b> {hlabel(TOP["key"])} {TOP["arm"]} is #{TOP["rank"]} on the corrected eligible scoreboard ({f(TOP["composite"],0)}). This is a seven-axis choice aid, not a universal intelligence score.',
 f'<b>Pick the specialist when one axis matters.</b> Stock agent reliability: {hlabel(BESTFLOOR[0])} {BESTFLOOR[1]} ({f(BESTFLOOR[2]["pass_hat_5"])} floor). Fast coding: {hlabel(FAST[0])} {FAST[1]} ({wall(FAST[2]["total_time"])}). Those lower-composite picks do not contradict the all-rounder.',
 f'<b>Coding ceilings are close; costs are not.</b> Highest complete reported score {f(MAXHE)}%: '+', '.join(f'{hlabel(k)} {e}' for k,e in BESTHE)+'. Swift IQ3_S low reported the same percentage with only 163 scored tasks and is qualified, not tied on equal coverage.',
 f'<b>Tool support is a serving-stack fact.</b> Gyro-M / rc2 joined the same pinned tool evaluator at {f(tl["mean"])} low and {f(th["mean"])} xhigh. Gyro-S / rc1 remains n/a for tools. That is not an isolated model-only comparison.',
 f'<b>More thinking has mixed returns.</b> Gyro-M xhigh solved one additional coding task after repair at {f(hewallratio)}x wall and {f(hetokenratio)}x tokens. Its Hard Mode mean rose, but its floor fell from {f(tl["pass_hat_5"])} to {f(th["pass_hat_5"])}.',
 f'<b>Fit and validity are part of the recommendation.</b> Gyro-M is experimental on this 16 GB card: {C["min_free_vram_mib"]} MiB minimum free VRAM, {C["max_temperature_c"]} C peak, invalid-slice status. No daily-driver endorsement. Partial scoring and test failures remain disclosed.'
])
section('THE FOUR QUESTIONS — straight answers','#a1662f')
w('<div class="cards">'+card('Best overall on the measured stock rig?',f'<b>{hlabel(TOP["key"])} {TOP["arm"]} · #{TOP["rank"]} overall</b>. Same winner in the final board and first Pick Your File card.','Equal-weight composite; specialists can be better at the job you actually need.')+card('Fastest complete coding loop?',f'<b>{hlabel(FAST[0])} {FAST[1]}</b> · #{rankmap[(FAST[0],FAST[1])]} overall. {wall(FAST[2]["total_time"])} at {f(FAST[2]["score"])}%.','This is elapsed coding-run time, not the fastest decoder.')+card('Best stock agent reliability?',f'<b>{hlabel(BESTFLOOR[0])} {BESTFLOOR[1]}</b> · #{rankmap[(BESTFLOOR[0],BESTFLOOR[1])]} overall. Floor {f(BESTFLOOR[2]["pass_hat_5"])}.','Gyro-M low ties the observed floor, but is a safety-capped experimental fork; no stock-composite rank.')+card('Best debugging / knowledge sample?',f'<b>{hlabel(DEBUG_PICK)}</b>: hidden debugging {M[DEBUG_PICK]["fivebugs"]["hid"]}, algebra sample {f(M[DEBUG_PICK]["knowledge"]["mmlu"]["accuracy"],0)}%.','Low coding arm is partial/unranked; xhigh is #'+str(rankmap[(DEBUG_PICK,'xhigh')])+'. A specialist, not the all-round crown.')+'</div>')
links([('PR #1850',D['revision']['pr'])])
section('THE CAST — ten files, one base model, three lineages','#0072B2')
how('All files share Qwen3.8-Flash-Next ancestry (Qwen / Alibaba). Q- labels: ISTA-DASLab quants; S- labels: UkisAI Swift derivatives; Gyro: AgentionAI APR. File sizes are decimal GB, not VRAM requirements. Results are integrated in the same tables; the engine and validity columns preserve comparison boundaries.')
table(['File / lab','Quant','File GB','Engine','Status','Download'],[[name(k),M[k]['quant'],f(M[k]['size_gb']),eng(k),note(k),f'<a href="{SRC[M[k]["lab"]]}">HF repo</a>'] for k in ORDER],title='Model provenance and engine cohorts')
call('The eight reference files use stock Strata 0.1.40.3; Gyro-S uses historical AgentionAI rc1, Gyro-M rc2. Shared evaluator and prompts do not remove the engine/configuration variable. Fork rows are inline observations, not ranked stock-engine controls.')
section('THE SCORECARD — all models in one view','#14927a')
how('Tools: low-effort mean of five runs and empirical all-five floor. Coding: final percentage plus the actual solved/scored count. Asterisk = partial scoring. Knowledge/debugging are limited once-per-file tests. Stock rows sorted by floor; fork observations follow, with engine labels and no stock-composite rank.')
kk=model_order(ORDER,lambda k:(M[k].get('trials')or{}).get('low',{}).get('pass_hat_5',-1))
floors=[M[k]['trials']['low']['pass_hat_5'] for k in STOCK]
rows=[]
for k in kk:
 m=M[k];t=(m.get('trials')or{}).get('low');a=m['heplus']['low'];b=m['heplus']['xhigh']
 rows.append([name(k),eng(k),f'{f(t["mean"])} +/- {f(t["std"])}' if t else 'n/a',f(t['pass_hat_5'])+(' '+medal(t['pass_hat_5'],floors) if k in STOCK else '') if t else 'n/a',f'{f(a["score"])} ({a["passed"]}/{a["scored_total"]})'+('*' if not a['audit']['complete_scoring'] else ''),f'{f(b["score"])} ({b["passed"]}/{b["scored_total"]})'+('*' if not b['audit']['complete_scoring'] else ''),f(m['knowledge']['gsm8k']['accuracy'],0),f(m['knowledge']['mmlu']['accuracy'],0),m['fivebugs']['hid']])
table(['File / lab','Engine','Tool mean','Pass^5','HE+ low','HE+ xhigh','Math','Algebra','Debug'],rows,title='Integrated headline scorecard')
w('<div class="small">Medals show the top three distinct stock-reference score levels, ties share. Fork rows do not receive stock medals. n/a is an intentional unavailable measurement, not zero.</div>')
figure('F2-wall.png','Low-effort observed time-to-quality. Hollow points have partial scoring; green points use separate fork engines. No model-only causal speedup is claimed.',True)
section('RELIABILITY & SAFETY — averages are not guarantees','#b9770e')
how(f'Tool-eval-bench: {tl["n_scenarios"]} realistic scenarios x five identical-seed runs at each effort. Score 0 / 1 / 2; 2 = full marks. Pass@5: full marks at least once, the empirical capability ceiling. Pass^5: full marks in ALL five, the reliability floor. Partial credit every run never passes the floor. These summaries are not deployment guarantees.')
figure('F3-reliability.png','All measured low-effort files in the same reliability view. Engine labels preserve the stock/fork distinction. The line joins mean and floor; the table gap is Pass@5 minus Pass^5.')
rr=sorted(alltools,key=lambda x:(M[x[0]]['cohort']!='stock',-x[2]['pass_hat_5']))
floorvals=[t['pass_hat_5'] for _,_,t in stocktools];meanvals=[t['mean'] for _,_,t in stocktools]
table(['Arm / lab','Mean +/- SD','95% CI','Pass@5','Pass^5 floor','Ceiling-floor gap','Last-trial rating','Warnings','Engine'],[[name(k)+' '+e,f'{f(t["mean"])} +/- {f(t["std"])}'+(' '+medal(t['mean'],meanvals) if k in STOCK else ''),f'{f(t["ci95"][0])}–{f(t["ci95"][1])}',f(t['pass_at_5']),f(t['pass_hat_5'])+(' '+medal(t['pass_hat_5'],floorvals) if k in STOCK else ''),f(t['gap']),rating(t),str(t['n_warnings']),eng(k)] for k,e,t in rr],title='Integrated five-trial tool reliability')
countcap=sum('safety-capped' in t['rating'] for _,_,t in stocktools)
w(f'<div class="small">{countcap}/{len(stocktools)} stock arms are capped on the stored last-trial rating. Gyro-M adds one capped and one Good rating. Not capped is NOT the same as no warnings or production safety. Warning lists/rating describe the final primary trial, not a five-run union.</div>')
figure('F9-decomposition.png','The all-five floor is the green count / 92. All tool-capable arms appear; fork rows are labeled rc2. A higher mean can coexist with more run-to-run flakes.',True)
verdict(f'Gyro-M illustrates the trade-off: always-full scenarios {tl["decomposition"]["solid"]} low / {th["decomposition"]["solid"]} xhigh; flaky {tl["decomposition"]["flaky"]} / {th["decomposition"]["flaky"]}. Its mean confidence intervals overlap, so the observed mean gain is modest. No causal mechanism is inferred from the aggregate alone.')
links([('Tool-eval-bench','https://github.com/SeraphimSerapis/tool-eval-bench')])
section('AGENTIC TOOL USE — hard chains reward extra thought','#4a5fc1')
how('Hard Mode (P) stresses long chains and recovery. Every category figure uses the five-run mean, not the last-trial score. Effort comparisons stay within a measured file/build. The engine column prevents the fork data from masquerading as a stock quant A/B.')
figure('F4-hardmode.png','All measured arms in one Hard Mode chart. Stock ranking and rc2 observational rows are distinguished by their labels.',True)
rr=sorted(alltools,key=lambda x:(M[x[0]]['cohort']!='stock',-x[2]['cats']['P']))
pvals=[t['cats']['P'] for _,_,t in stocktools]
table(['Arm / lab','Hard Mode mean','SD across five','Turn budget','Engine'],[[name(k)+' '+e,f(t['cats']['P'])+(' '+medal(t['cats']['P'],pvals) if k in STOCK else ''),f(t['cats_std']['P']),str(t['config']['max_turns']),eng(k)] for k,e,t in rr],title='Multi-step tool-chain results')
bullet([f'Gyro-M Hard Mode rises from {f(tl["cats"]["P"])}% to {f(th["cats"]["P"])}%, while its all-five floor falls. Optimizing only one of those axes is not a reliable workload decision.','Warnings include premature notifications, missed dependency ordering and claims that an invoice is paid while funds remain held. They are scripted failure traps, not permissions.','rc2 live forced tool calls passed streaming and non-streaming with Gyro-M; Gyro-S rc1 remains unavailable for tool-eval. This does not establish Gyro-S behavior on rc2.'])
section('CODE — every denominator visible','#0b5c8a')
how('164 function-writing tasks via BenchKit / EvalPlus, greedy generation, one verifier-guided repair. First = one-shot score; final includes the replacement. Scored timeout/length/blank outputs remain failures. Excluded harness errors can reduce the denominator; those rows retain observed percentages but are qualified and excluded from full-coverage/composite claims.')
figure('F1-coding.png','All ten files; hollow starred bars use 163 or 162 scored tasks. Other ordinary failures remain in the 164-task denominator.')
rr=[(k,e,M[k]['heplus'][e]) for k in ORDER for e in ['low','xhigh']]
rr=arm_order(rr,lambda r:r['score'],True)
good=[r['score'] for k,e,r in rr if k in STOCK and r['audit']['complete_scoring']];walls=[r['total_time'] for k,e,r in rr if k in STOCK and r['audit']['complete_scoring']]
table(['Arm / lab','First %','Final %','Solved / scored','Repairs','Wall','tok/s','Blank finals','Infra excluded','Engine'],[[name(k)+' '+e,f(r['first_attempt_score']),f(r['score'])+('*' if not r['audit']['complete_scoring'] else (' '+medal(r['score'],good) if k in STOCK else '')),f'{r["passed"]}/{r["scored_total"]}',f'{r["repair_successes"]}/{r["repair_attempted"]}',wall(r['total_time'])+(' '+medal(r['total_time'],walls,True) if k in STOCK and r['audit']['complete_scoring'] else ''),f(r['tok_s']),str(len(r['audit']['blank_final_ids'])),str(r.get('harness_errors',0)),eng(k)] for k,e,r in rr],title='Integrated coding results — full and qualified coverage')
call('Retrospective coverage correction: Q-IQ3_XXS low 155/163; Swift IQ3_S low 157/163; Gyro-S xhigh 153/162 after two HTTP 502 harness errors. Their observed percentages are preserved, not silently rescored. Gyro-M xhigh blank finals on HumanEval/39 and HumanEval/99 ended on length and remain scored failures; 157/164 is not a clean-response certificate.')
bullet([f'Gyro-M xhigh costs {f(hewallratio)}x low wall and {f(hetokenratio)}x tokens for one additional final pass. Hard Mode benefits do not imply a coding-efficiency benefit.',f'Gyro-M low vs historical Gyro-S low has {f(lo["first_attempt_score"]-M["gyro-s"]["heplus"]["low"]["first_attempt_score"])} points more first-attempt score, {f(abs(lo["score"]-M["gyro-s"]["heplus"]["low"]["score"]))} points less final score. Engines/configurations differ; this is not isolated quant improvement.'])
links([('EvalPlus','https://github.com/evalplus/evalplus'),('BenchKit','https://github.com/DogukanUrker/BenchKit')])
section('KNOWLEDGE & DEBUGGING — the test we actually ran','#6c3483')
how('GSM8K: 100 seeded-shuffled questions, 8-shot. MMLU: first 100, 5-shot, entirely abstract algebra — NOT a 57-subject exam. IFEval: first 50 prompts / 76 constraints. Five Bugs: five buggy Python files x five sessions, visible plus local hidden checks. All are limited samples; do not turn them into a whole-model intelligence ranking.')
kk=model_order(ORDER,lambda k:int(M[k]['fivebugs']['hid'].split('/')[0]));hidden=[int(M[k]['fivebugs']['hid'].split('/')[0]) for k in STOCK]
table(['File / lab','GSM8K /100','Algebra /100','IFEval prompt / instruction %','Bugs visible','Bugs hidden','Needle % @ tokens','Engine'],[[name(k),f(M[k]['knowledge']['gsm8k']['accuracy'],0),f(M[k]['knowledge']['mmlu']['accuracy'],0),f'{f(M[k]["knowledge"]["ifeval"]["prompt_accuracy"])} / {f(M[k]["knowledge"]["ifeval"]["instruction_accuracy"])}',M[k]['fivebugs']['vis'],M[k]['fivebugs']['hid']+(' '+medal(int(M[k]['fivebugs']['hid'].split('/')[0]),hidden) if k in STOCK else ''),f'{f(M[k]["needle"]["retrieval_accuracy"],0)} @ {int(M[k]["needle"]["effective_context"]):,}',eng(k)] for k in kk],title='Integrated sampled knowledge, debugging and long-context recall')
figure('F5-debug.png','Every model in the same visible/hidden debugging plot. Fork results retain engine-qualified observational status.')
verdict(f'{hlabel(DEBUG_PICK)} remains the debugging specialist at {M[DEBUG_PICK]["fivebugs"]["hid"]} hidden sessions and leads this algebra sample at {f(M[DEBUG_PICK]["knowledge"]["mmlu"]["accuracy"],0)}%. Its low coding arm is partial and cannot claim the stock all-round crown.')
how(f'Needle: 25 cells (five lengths x five insertion depths). Effective retrieved lengths are {int(M["iq3_s"]["needle"]["effective_context"]):,} for stock and {int(G["needle"]["effective_context"]):,} for both fork rows. Gyro-M configured context {C["manifest"]["context"]:,}; no 260K claim for it.')
links([('Five Bugs upstream','https://github.com/vyang472/five-bugs'),('TEB knowledge / needle plugins','https://github.com/SeraphimSerapis/tool-eval-bench')])
section('REASONING ECONOMICS — what you wait for','#009e73')
how('Total output tokens include reasoning, repairs and failures. Wall includes request/harness overhead. Failure time share and passing-task cost come from task-level response times, not wall subtraction. Different engine/protocol rows remain descriptive.')
figure('F6-tokens.png','Completion-token totals for all attempted tasks, all files, both efforts. Partial-scoring arms retain their coverage qualifier.')
rr=[(k,e,M[k]['heplus'][e]) for k in ORDER for e in ['low','xhigh']]
rr.sort(key=lambda x:(M[x[0]]['cohort']!='stock',x[2]['total_time']))
table(['Arm / lab','Output tokens','Passing tokens','Passing seconds','Failed response time %','Wall','Engine'],[[name(k)+' '+e,f(r['total_output_tokens'],0),f(r['audit']['passing_tokens'],0),f(r['audit']['passing_response_s'],0),f(r['audit']['failing_time_share_pct'])+'%',wall(r['total_time']),eng(k)] for k,e,r in rr],title='Integrated time-to-result economics')
verdict('Pick effort for the task: more deliberation can improve long tool chains while making code loops slower. The overall ranking averages capability axes, not latency or safety; the specialist cards below say exactly which axis overrides that ranking.')
section('SPEED — real decode; corrected first-token timing','#c0392b')
how('llama-benchy 0.4.0: prompt 2,048, generation 256, concurrency 1, three repeats/depth/effort. Raw pp_throughput in the old PDF was inflated by early SSE role chunks and is retired. Here prompt-to-first-token rate = (prompt + background context) / e2e TTFT, including network/queue/reasoning. It is NOT an engine-kernel prefill benchmark. Largest depths differ: stock 131,072; rc2 122,880.')
figure('F7-decode-all.png','All finite low-effort decode profiles. Lightest bar = deepest measured point; row tables state the exact token depths. Gyro-S malformed profiles remain n/a.')
figure('F8-decode-effort.png','Decode at zero background context for both efforts, all measurable files. A fork-engine observation is not a same-engine model A/B.')
for effort in ['low','xhigh']:
 kk=model_order(ORDER,lambda k:((M[k].get('perf')or{}).get(effort)or{}).get('0',{}).get('tg_mean',-1));rows=[]
 for k in kk:
  pp=(M[k].get('perf')or{}).get(effort)
  if not pp:rows.append([name(k),eng(k),'n/a','n/a','n/a','n/a']);continue
  ps=sorted(pp.values(),key=lambda r:r['context_tokens'])
  rows.append([name(k),eng(k),' / '.join(f(r['context_tokens'],0) for r in ps),' / '.join(f(r['tg_mean']) for r in ps),' / '.join(f(r['prefill_e2e_tps'],0) for r in ps),f(ps[-1]['e2e_ttft_ms']/1000,2)])
 table(['File / lab','Engine','Context tokens: 0 / mid / deep','Decode tok/s: 0 / mid / deep','First-token rate: 0 / mid / deep','Deep TTFT s'],rows,title=f'Integrated {effort}-effort performance profiles')
call('Gyro-S synthetic streaming produced million-token/s decode artifacts and null depth cells. Those profiles are excluded, not recycled as a valid speed baseline. Every printed finite profile has three stored measurements per depth; both efforts appear in the same format.')
section('THE FINAL SCOREBOARD — one message, with explicit scope')
how(f'OVERALL averages seven min-max-normalized axes within the {len(el)} eligible stock-engine arms: coding final, agent floor, Hard Mode mean, algebra sample, math sample, hidden debugging, zero-context decode. Full scoring only. Forks and partial arms appear inline but are unranked. This crowns the best measured STOCK ALL-ROUNDER, not every workload winner; safety and latency are not averaged into that score.')
figure('F10-scoreboard.png','Same ranks and scope used by every recommendation below. Asterisk = partial coding denominator. Not capped is a final-trial rating status, not a safety guarantee.',True)
verdict(f'<b>All-rounder:</b> {hlabel(TOP["key"])} {TOP["arm"]} · #{TOP["rank"]} overall ({f(TOP["composite"],0)}). <b>Specialists:</b> {hlabel(BESTFLOOR[0])} {BESTFLOOR[1]} · #{rankmap[(BESTFLOOR[0],BESTFLOOR[1])]} owns the stock floor; {hlabel(FAST[0])} {FAST[1]} · #{rankmap[(FAST[0],FAST[1])]} owns the fastest complete coding wall. Pick Your File translates THIS board; it does not introduce a second overall winner.')
bridge_roles={'all_rounder':('Best stock all-rounder','Balance the seven measured capability axes'),'reliability':('Reliability specialist','Prioritize the empirical tool floor'),'fast_coding':('Coding-latency specialist','Prioritize complete-run wall time, outside OVERALL'),'coding_ceiling':('Coding-ceiling specialist','Prioritize the complete coding score'),'knowledge_debug':('Knowledge / debugging specialist','Prioritize independent sampled tests; low is partial/unranked'),'experimental_not_production':('Experimental observation only','Different engine and narrow 16 GB margin; no production pick')}
table(['Decision','File / effort','OVERALL rank','Why this agrees with the board'],[[bridge_roles[r['role']][0],name(r['key'])+' '+r['effort'],('#'+str(r['rank'])) if r['rank'] else 'unranked',bridge_roles[r['role']][1]] for r in RECS],title='One recommendation registry — all-rounder versus specialists')
section('PICK YOUR FILE — the scoreboard translated into jobs','#14927a')
how('<b>One decision rule:</b> use the #1 stock all-rounder if you balance the seven measured axes. Override that composite only when your workload has a named priority — reliability, coding latency, coding ceiling, or debugging. Lower-ranked specialists are deliberate exceptions, not alternative overall crowns. Unranked experimental/partial rows are not eligible production recommendations.')
cards=[]
cards.append(card('Balanced workload · best stock all-rounder',f'<b>{hlabel(TOP["key"])} {TOP["arm"]} · #{TOP["rank"]} overall</b><br>Score {f(TOP["composite"],0)} across the seven eligible stock axes. Also leads stock tool mean at {f(BESTMEAN[2]["mean"])}.','Composite is equal-weighted, on one rig and limited knowledge samples; not a universal quality/safety claim.'))
cards.append(card('Reliability-first agent · specialist override',f'<b>{hlabel(BESTFLOOR[0])} {BESTFLOOR[1]} · #{rankmap[(BESTFLOOR[0],BESTFLOOR[1])]} overall</b><br>Stock floor {f(BESTFLOOR[2]["pass_hat_5"])} is highest. Choose this lower-ranked arm when consistent full-mark tool behavior outweighs the other axes.','Still has scripted safety warnings. Gyro-M low ties the observed floor but is unranked/experimental, not an equivalent production pick.'))
cards.append(card('Fast coding loop · specialist override',f'<b>{hlabel(FAST[0])} {FAST[1]} · #{rankmap[(FAST[0],FAST[1])]} overall</b><br>Complete coding wall {wall(FAST[2]["total_time"])} at {f(FAST[2]["score"])}%.','You are prioritizing latency, which is outside the composite; this is not a new all-round winner.'))
cards.append(card('Highest stock coding ceiling · specialist override',f'<b>{hlabel(CODE_PICK[0])} {CODE_PICK[1]} · #{rankmap[(CODE_PICK[0],CODE_PICK[1])]} overall</b><br>Complete reported coding {f(CODE_PICK[2]["score"])}% ({CODE_PICK[2]["passed"]}/{CODE_PICK[2]["scored_total"]}).','Scored blank/timeout failures remain. Gyro-S low ties this reported ceiling on rc1, but is not a tool-capable stock control.'))
cards.append(card('Knowledge / debugging · specialist override',f'<b>{hlabel(DEBUG_PICK)}</b><br>Hidden checks {M[DEBUG_PICK]["fivebugs"]["hid"]}, algebra sample {f(M[DEBUG_PICK]["knowledge"]["mmlu"]["accuracy"],0)}%. Low unranked (partial coding); xhigh #{rankmap[(DEBUG_PICK,"xhigh")]} overall.','The partial low coding result prevents an overall crown. Specialist evidence is from separate debugging/knowledge samples; choose it only for that priority.'))
cards.append(card('Experimental fork · no production recommendation',f'<b>Gyro-M (AgentionAI) low · unranked rc2</b><br>HE+ {f(lo["score"])}%, observed floor {f(tl["pass_hat_5"])}, live tools. The board displays its results but does not authorize a stock rank.','Minimum '+str(C['min_free_vram_mib'])+' MiB free VRAM, '+f(C['manifest']['context'],0)+' context, invalid-slice status. Not a supported 16 GB daily-driver pick. Gyro-S rc1 has no valid tools score.'))
w('<div class="cards">'+''.join(cards)+'</div>')
section('FOR MAINTAINERS — measured findings and validity','#34495e')
bullet(['<b>Tool plumbing is build-specific:</b> rc1 forced calls were ignored with Gyro-S. rc2 live forced calls passed with Gyro-M on streaming and non-streaming. The historical finding stays scoped to rc1; Gyro-S on rc2 was not tested.',f'<b>Experimental fork fit:</b> {C["resource_samples"]} resource samples, minimum {C["min_free_vram_mib"]} MiB free, peak {C["max_temperature_c"]} C. '+', '.join(f'{k}={v}' for k,v in C['event_maxima'].items())+'. This is not long-term stability proof.',f'<b>Campaign status:</b> <code>{C["outcome"]}</code>, finished {escape(C["finished_at"])}. CTest {C["ctest"]["passed"]} pass / {C["ctest"]["skipped"]} skip / {C["ctest"]["failed"]} fail. ple_parity lacks its required Q2_0 PLE shard/capture fixtures; separate ple_reader_selftest passed.','<b>Audit bookkeeping:</b> both HumanEval+ task audits ran; a dictionary merge overwrote intended audit-step names, making done.json mark them absent. Xhigh validity logic also missed blank finals. Raw status/scores are preserved; no silently upgraded clean pass.', '<b>Monitoring:</b> corrected busy-vs-down logic after the first aborted attempt. Perf-low/needle static-counter warnings were not accompanied by engine-event counters. Standalone effort probe is excluded from scores; possible timing interference is disclosed.', '<b>Operations:</b> production Swift IQ3_S restoration passed health/chat/tool/vision and re-enabled guards/updater. Report revisions did not alter services.'])
section('METHOD & CORRECTIONS — reproducible without hiding caveats','#5b6770')
how('The PDF, GitHub README, scoreboard and recommendation registry are generated from the same extracted store. New data is integrated throughout, not isolated in a model-specific chapter. Source records and hashes are packaged; the portable compiler rebuilds from this folder.')
bullet(['<b>Coding:</b> BenchKit direct, 164 attempted tasks, greedy temperature 0, presence penalty 1.0, one repair. Gyro-M timeout 900 seconds, no injected client token cap. Serving protocol and engine changes qualify historical cross-cohort comparisons.',f'<b>Tools:</b> {C["manifest"]["teb_version"]}, {tl["n_scenarios"]} scenarios x five trials per effort; seed {tl["config"]["seed"]}, temperature {tl["config"]["temperature"]}, timeout {f(tl["config"]["timeout_seconds"],0)} s, turn cap {tl["config"]["max_turns"]}. Separate Swift v2.8.0 / 97-scenario data is NOT mixed in.','<b>Corrections:</b> historical partial coding denominators qualified; category scores changed to five-run means; inflated raw prefill removed; ceiling-minus-floor gap labeled correctly; peak temperature corrected from 64 C summary to measured 78 C. These corrections, not an unmeasured model promotion, change the earlier recommendation framing.','<b>Not tested:</b> vision quality, multi-day agents, extra seeds, multi-GPU, other GPU sizes, full knowledge datasets, Gyro-M beyond 131K or Gyro-S on rc2. No automatic supported-memory or production-stability conclusion.'])
manifest=C['manifest']
table(['Provenance item','Pinned value'],[['Stock engine digest','34CDE150B21148E6'],['Fork rc2 run',manifest['campaign_id']],['Fork engine commit','<code>'+manifest['engine_commit']+'</code>'],['Fork binary SHA256','<code>'+manifest['engine_sha256']+'</code>'],['Gyro-M model revision','<code>'+manifest['model_revision']+'</code>'],['Gyro-M model SHA256','<code>'+manifest['model_sha256']+'</code>'],['Fork context / KV',f'{manifest["context"]:,} / {manifest["kv"]}'],['Rebuild','scripts/build_revision.py; data/ALL-METRICS.json; claims.json']],title='Version-pinned provenance')
links([('PR #1850',D['revision']['pr']),('Strata','https://github.com/Niko1221/Strata'),('AgentionAI fork','https://github.com/agentionai/Strata'),('Gyro source',SRC['AgentionAI']),('Tool-eval-bench','https://github.com/SeraphimSerapis/tool-eval-bench'),('EvalPlus','https://github.com/evalplus/evalplus'),('Five Bugs','https://github.com/vyang472/five-bugs')])
w(f'<div class="small">Revised {D["revision"]["date"]} · source-qualified observations · all recommendations use the same registry and ranks. Numeric audit, whole-table pagination proof, source manifest and portable rebuild accompany the report.</div>')
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
(BASE/'README.md').write_text('# Community quant battery — RTX 5070 Ti 16 GB\n\n[Download the full PDF](strata-community.pdf) · Updated 2026-10-12 · 10 files / 20 effort arms.\n\n'+readme+'\n\n## Rebuild and data\n\n`python scripts/build_revision.py` rebuilds all charts, PDF and this README from `data/ALL-METRICS.json`. `scripts/audit_revision.py` checks claims. Raw HumanEval+ task files may be gzip-compressed; use Python gzip/json or `gzip -dc`. The rig-side extractor is not required to reproduce figures.\n')
pdf=BASE/'strata-community.pdf'
proc=subprocess.run(['chromium','--headless','--disable-gpu','--no-sandbox','--no-pdf-header-footer',f'--print-to-pdf={pdf}',(BASE/'community-updated.html').as_uri()],capture_output=True,text=True,timeout=150)
print('CHROMIUM_RC',proc.returncode)
if proc.returncode:print(proc.stderr);raise SystemExit(proc.returncode)
assert pdf.stat().st_size>100000
print('PDF',pdf,'bytes',pdf.stat().st_size)
info=subprocess.run(['pdfinfo',str(pdf)],capture_output=True,text=True,check=True)
print('\n'.join(s for s in info.stdout.splitlines() if s.startswith(('Pages:','Page size:','CreationDate:'))))
print('SECTION ORDER',re.findall(r'<h2[^>]*>(.*?)</h2>',text))
print('COMPOSITE_LEADER',TOP['key'],TOP['arm'],TOP['composite'])
print('TABLE_COUNT',TABLE_COUNT)
exec(compile((BASE/'scripts/write_public_docs.py').read_text(),str(BASE/'scripts/write_public_docs.py'),'exec'))
