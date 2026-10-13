#!/usr/bin/env python3
"""Generate GitHub README and PR draft from the same recommendation registry."""
from pathlib import Path
import json,re,subprocess
BASE=Path(__file__).resolve().parent.parent
D=json.loads((BASE/'data/ALL-METRICS.json').read_text());M=D['models'];C=D['gyro_m_campaign'];REF=D['hosted_api_reference']
HR=REF['human_eval_plus'];TR=REF['tool_eval_bench'];HFB=REF['five_bugs'];HKS=REF['sampled_knowledge_and_recall'];HPERF=REF['performance']
SB=json.loads((BASE/'data/FINAL-SCOREBOARD.json').read_text());CL=json.loads((BASE/'claims.json').read_text());RECS=CL['recommendations']
TOP=CL['stock_composite_leader']; ORDER=list(M);FORK='https://github.com/Zauberio/Strata';BRANCH='study/quant-battery-rtx5070ti';REL='bench/results/2026-10-10-community-quant-study-rtx5070ti'
def lab(k):return f"{M[k]['short']} ({M[k]['lab']})"
HOSTED_LABEL='GPT-6-luna MAX (OpenAI) · hosted API reference'
HOSTED_ENGINE='OpenAI Codex Responses API · max'
def f(x,n=1):return f'{x:,.{n}f}' if isinstance(x,(float,int)) else 'n/a'
def engine(k):return {'stock':'0.1.40.3','rc1':'rc1','rc2':'rc2'}[M[k]['cohort']]
def wall(x):v=round(x);return f'{v//60}m{v%60:02d}s'
def table(h,rows):return '\n'.join(['| '+' | '.join(map(str,h))+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(str(v).replace('|','\\|').replace('\n',' ') for v in r)+' |' for r in rows])
def rating(t):return t['rating'].replace('★',str(t['rating'].count('★'))+'/5',1).replace('★','') if t else 'n/a'
def arm(k,e):return f'{lab(k)} {e}'
info=subprocess.run(['pdfinfo',str(BASE/'strata-community.pdf')],capture_output=True,text=True,check=True).stdout;pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
roles={'all_rounder':('Best stock all-rounder','Balance seven measured axes; equal-weight composite, not universal intelligence.'),'reliability':('Reliability specialist','Highest stock all-five tool floor; warnings still exist.'),'fast_coding':('Coding latency specialist','Fastest complete coding wall; latency is outside OVERALL.'),'coding_ceiling':('Coding ceiling specialist','Highest complete stock coding score; scored failures remain.'),'knowledge_debug':('Knowledge / debugging specialist','Independent limited samples; low coding partial/unranked, xhigh rank shown.'),'experimental_not_production':('Experimental observation, not a production pick','rc2 fork, narrow 16 GB margin, invalid-slice campaign status.')}
picks=[[roles[r['role']][0],arm(r['key'],r['effort']),f"#{r['rank']}" if r['rank'] else 'unranked',roles[r['role']][1]] for r in RECS]
H=[]
def w(x):H.append(x)
w(f'# Qwen3.8-Flash-Next: which quant should you run?\n\nUpdated **{D["revision"]["date"]}** · **10 local files / 3 quant or finetune labs / 20 local effort arms / 18 local tool arms**. The separate **{HOSTED_LABEL}** is outside those counts and anchors score-relative suite comparisons; it is not a recommendation.\n\n[Download the designed PDF](strata-community.pdf) ({pages} pages) · [Canonical metrics](data/ALL-METRICS.json) · [Hosted reference detail](data/hosted-api-reference.json) · [Claim registry](claims.json) · [Audit](qa/AUDIT.json)\n\nMeasured on **RTX 5070 Ti 16 GB, Ryzen 9800X3D, 64 GB RAM, Windows + WSL2**. Eight stock files use Strata 0.1.40.3; Gyro-S uses historical rc1, Gyro-M rc2. Shared evaluators do not isolate model-only stock/fork differences. The separate GPT-6-luna MAX row is an OpenAI-hosted baseline/reference, not a recommendation or local RTX run.')
w(f'## Straight answer\n\n**{arm(TOP["key"],TOP["arm"])} is #{TOP["rank"]}, the best measured stock all-rounder** ({f(TOP["composite"],0)} overall). Reliability, latency, coding ceiling and debugging specialists can be lower on that board because their chosen priority outweighs its other axes. The final scoreboard and Pick Your File use the exact same registry and ranks. Experimental fork rows and partial coding arms remain visible but unranked.')
w('## The cast\n\nThe ten local files all descend from the same Qwen3.8-Flash-Next base model; file sizes are decimal GB, not VRAM requirements. The GPT-6-luna MAX hosted API reference is not a local file and does not enter this inventory or the recommendation registry. Every common result table includes Gyro-M inline with an engine label. There is no standalone Gyro-M chapter.\n\n'+table(['File / lab','Quant','File GB','Engine','Status'],[[lab(k),M[k]['quant'],f(M[k]['size_gb']),engine(k),'experimental on this 16 GB rig' if k=='gyro-m' else ('historical fork' if k=='gyro-s' else 'reference')] for k in ORDER]))
w('Downloads: [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) · [UkisAI Swift](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) · [AgentionAI Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF).')
w('## Reliability and agentic tools\n\n[Tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench): pinned `'+C['manifest']['teb_version']+'`, 92 scenarios x five runs per effort; 0/1/2 scoring. **Pass@5** = full marks at least once (capability ceiling); **Pass^5** = full marks in ALL five (reliability floor). Category values use all-five means. Warning counts describe the last primary trial, not a five-run union. The table is ordered by Pass^5 floor (descending; mean breaks ties). Evaluator rating is omitted because the hosted artifact provides no comparable rating. Safety-capped is not inferred merely from the warning count. Gyro-S rc1 has no usable tool baseline; it is n/a, not zero.')
tr=[(k,e,t) for k in ORDER for e,t in (M[k].get('trials')or{}).items()]+[('hosted-reference','max',TR)]
tr.sort(key=lambda r:(-(TR['pass_hat_5_percent'] if r[0]=='hosted-reference' else r[2]['pass_hat_5']),-(TR['mean'] if r[0]=='hosted-reference' else r[2]['mean'])))
tr_rows=[]
for k,e,t in tr:
 if k=='hosted-reference':tr_rows.append([HOSTED_LABEL,HOSTED_ENGINE,f'{f(TR["mean"])} +/- {f(TR["stddev"])}',f(TR['pass_at_5_percent']),f(TR['pass_hat_5_percent']),f(TR['reliability_gap_percent']),f(TR['hard_mode_mean_percent']),TR['final_trial_safety_warnings']])
 else:tr_rows.append([arm(k,e),engine(k),f'{f(t["mean"])} +/- {f(t["std"])}',f(t['pass_at_5']),f(t['pass_hat_5']),f(t['gap']),f(t['cats']['P']),t['n_warnings']])
w(table(['Arm / lab','Engine','Mean +/- SD','Pass@5','Pass^5','Ceiling-floor gap','Hard Mode mean','Warnings (last trial)'],tr_rows))
w('95% CI and evaluator rating are intentionally omitted: the hosted artifact does not publish comparable values, so no proxy is inferred. GPT-6-luna MAX is ordered by its measured score but remains an external OpenAI API baseline, not a local-engine arm.')
w('GPT-6-luna MAX sampling caveat: '+REF['sampling_caveat']+' Treat it as an observed reference run, not an identical seed/temperature-controlled local trial.')
for fn in ['F3-reliability.png','F9-decomposition.png','F4-hardmode.png']:w(f'![All-model tool results](charts/{fn})')
w('Gyro-M Hard Mode gains do not imply reliability gains: its floor drops at xhigh and mean confidence intervals overlap. Engine/configuration differences prevent a model-only causal comparison.')
w('## Coding\n\n[EvalPlus](https://github.com/evalplus/evalplus) through BenchKit: 164 attempted function-writing tasks, greedy generation plus one verifier-guided replacement. First = one-shot score; final includes replacement. Blank/length-limited outputs remain scored failures. Excluded harness errors reduce coverage; those rows are qualified and excluded from complete-coverage medals/composites.\n\n![Coding](charts/F1-coding.png)\n\n![Time to quality](charts/F2-wall.png)')
he=[(k,e,r) for k in ORDER for e,r in M[k]['heplus'].items()]
he_entries=[(k,e,r,False) for k,e,r in he]+[('hosted-reference','max',HR,True)]
def he_order_key(x):
 k,e,r,hosted=x
 complete=(HR['harness_errors']==0 and len(HR['tasks'])==164) if hosted else r['audit']['complete_scoring']
 return (not complete,-(HR['final_percent'] if hosted else r['score']))
he_entries.sort(key=he_order_key)
he_rows=[]
for k,e,r,hosted in he_entries:
 if hosted:he_rows.append([HOSTED_LABEL,HOSTED_ENGINE,f(HR['first_percent']),f(HR['final_percent']),f'{HR["final_passes"]}/{len(HR["tasks"])}',f'{HR["repair_successes"]}/{HR["repair_attempted"]}',wall(HR['wall_seconds']),f(HR['provider_reported_tokens_per_second'])+'†',HR['blank_final_count'],HR['harness_errors']])
 else:he_rows.append([arm(k,e),engine(k),f(r['first_attempt_score']),f(r['score'])+(' *' if not r['audit']['complete_scoring'] else ''),f'{r["passed"]}/{r["scored_total"]}',f'{r["repair_successes"]}/{r["repair_attempted"]}',wall(r['total_time']),f(r['tok_s']),len(r['audit']['blank_final_ids']),r.get('harness_errors',0)])
w('Rows are ordered by final score, with complete-coverage runs first and partial-coverage runs after them.')
w(table(['Arm / lab','Engine','First %','Final %','Solved/scored','Repairs','Wall','tok/s','Blank finals','Infra excluded'],he_rows))
w(f'† Hosted {f(HR["provider_reported_tokens_per_second"])} tok/s is provider-reported completion usage including hidden reasoning tokens. It is not visible-text throughput, local decode speed, or a matched speed result; see performance caveat below.')
w('`*` Partial: Q-IQ3_XXS low 155/163; Swift IQ3_S low 157/163; Gyro-S xhigh 153/162 after two HTTP 502 errors. Gyro-M xhigh blank HumanEval/39 and /99 ended on length and remain failures in 157/164. No raw score/status was silently upgraded.')
w('## Sampled knowledge, debugging and recall\n\nGSM8K 100 shuffled questions, 8-shot; MMLU first 100 entirely abstract algebra, 5-shot (NOT full MMLU); IFEval first 50 prompts / 76 constraints. [Five Bugs](https://github.com/vyang472/five-bugs): five files x five sessions with local hidden checks. Needle: 25 cells; stock and fork effective contexts differ.')
knowledge_entries=[('local',k) for k in ORDER]+[('hosted','hosted-reference')]
knowledge_entries.sort(key=lambda x:-(HFB['hidden_passes'] if x[0]=='hosted' else int(M[x[1]]['fivebugs']['hid'].split('/')[0])))
knowledge_rows=[]
for kind,k in knowledge_entries:
 if kind=='hosted':knowledge_rows.append([HOSTED_LABEL,HOSTED_ENGINE,f(HKS['gsm8k']['accuracy_percent'],0),f(HKS['mmlu']['accuracy_percent'],0),f'{f(HKS["ifeval"]["prompt_accuracy_percent"])} / {f(HKS["ifeval"]["instruction_accuracy_percent"])}',f'{HFB["visible_passes"]}/{HFB["visible_total"]}',f'{HFB["hidden_passes"]}/{HFB["hidden_total"]}',f'{f(HKS["needle"]["accuracy_percent"],0)} @ {HKS["needle"]["effective_context_tokens"]:,}'])
 else:knowledge_rows.append([lab(k),engine(k),f(M[k]['knowledge']['gsm8k']['accuracy'],0),f(M[k]['knowledge']['mmlu']['accuracy'],0),f'{f(M[k]["knowledge"]["ifeval"]["prompt_accuracy"])} / {f(M[k]["knowledge"]["ifeval"]["instruction_accuracy"])}',M[k]['fivebugs']['vis'],M[k]['fivebugs']['hid'],f'{f(M[k]["needle"]["retrieval_accuracy"],0)} @ {M[k]["needle"]["effective_context"]:,}'])
w('Rows are ordered by hidden Five Bugs score (descending); Luna is shown in score position as a hosted reference, not a local candidate.')
w(table(['File / lab','Engine','GSM8K','Algebra','IFEval prompt/instruction','Bugs visible','Bugs hidden','Needle % @ effective tokens'],knowledge_rows))
w('![Debugging](charts/F5-debug.png)\n\nThese are small samples, not a whole-model intelligence ranking or long-session guarantee.')
w('## Reasoning economics\n\nCompletion tokens include thinking, repair and failures. Passing/failed task costs come from response times, not from subtracting job wall time.\n\n![Completion tokens](charts/F6-tokens.png)\n\n'+table(['Arm / lab','Engine','Output tokens','Passing tokens','Passing seconds','Failed response-time share %','Wall'],[[arm(k,e),engine(k),f(r['total_output_tokens'],0),f(r['audit']['passing_tokens'],0),f(r['audit']['passing_response_s'],0),f(r['audit']['failing_time_share_pct']),wall(r['total_time'])] for k,e,r in he]))
w(f'## Measured performance\n\nllama-benchy 0.4.0: prompt 2,048, generation 256, concurrency 1, three repeats/depth/effort. Retired inflated raw prompt-throughput values. **Prompt-to-first-token rate** = total input tokens / end-to-end TTFT, including queue/network/reasoning; not kernel prefill. Deep contexts differ: stock 131,072, rc2 122,880. Gyro-S malformed million-token/s/null profiles are excluded and labeled n/a. GPT-6-luna MAX matched decode comparison is {HPERF["status"].replace("_", " ")}: requested {"/".join(str(v) for v in HPERF["requested_repeats_per_context"])} repeats at {" / ".join(f(v,0) for v in HPERF["contexts_tokens"])} tokens, with only {"/".join(str(v) for v in HPERF["visible_text_rate_samples_per_context"])} valid visible-rate samples. All hosted rates are withheld; its provider-reported HE+ token rate is not a speed substitute.\n\n![Low-effort profiles](charts/F7-decode-all.png)\n\n![Both efforts](charts/F8-decode-effort.png)')
for e in ['low','xhigh']:
 rows=[]
 for k in ORDER:
  pp=(M[k].get('perf')or{}).get(e)
  if not pp:rows.append([lab(k),engine(k),'n/a','n/a','n/a','n/a']);continue
  ps=sorted(pp.values(),key=lambda r:r['context_tokens'])
  rows.append([lab(k),engine(k),' / '.join(f(r['context_tokens'],0) for r in ps),' / '.join(f(r['tg_mean']) for r in ps),' / '.join(f(r['prefill_e2e_tps'],0) for r in ps),f(ps[-1]['e2e_ttft_ms']/1000,2)])
 w(f'### {e}\n\n'+table(['File / lab','Engine','Contexts 0/mid/deep','Decode tok/s 0/mid/deep','First-token rate 0/mid/deep','Deep TTFT seconds'],rows))
w('## The final scoreboard\n\nSeven equally weighted min-max-normalized axes: coding final, tool floor, Hard Mode mean, algebra, math, hidden debugging, zero-context decode. Normalization uses only complete eligible stock-engine arms. Safety and wall latency are not composite axes; fork and partial rows are unranked. This is an all-rounder choice aid on this rig, not a universal quality or safety score.\n\n![The final scoreboard](charts/F10-scoreboard.png)')
w(table(['Rank','Arm / lab','Engine','OVERALL','Coding','Floor','Hard Mode','Algebra','Math','Debug','Decode','Eligibility'],[[r['rank'] or '—',arm(r['key'],r['arm']),engine(r['key']),f(r['composite'],0),f(r['coding']),f(r['floor']),f(r['hardmode']),f(r['mmlu'],0),f(r['gsm8k'],0),r['debug'],f(r['decode']),'eligible' if r['eligible'] else r['reason']] for r in SB]))
w('## Pick Your File — this same board translated into jobs\n\nThe #1 stock all-rounder is the default for a balanced seven-axis workload. Lower-ranked specialists are explicit priority overrides, not alternate overall crowns. Experimental/partial rows do not become production picks.\n\n'+table(['Job','Pick / lab','Board rank','Rationale / caveat'],picks))
w('## Maintainer findings, validity and corrections\n\n- rc1 forced tools were ignored with Gyro-S; rc2 live forced calls passed with Gyro-M in streaming/non-streaming. Gyro-S on rc2 was not tested. These are serving-stack findings, not model-only quality changes.\n- The Gyro-M full campaign finished `'+C['finished_at']+'`, status **`'+C['outcome']+'`**. Its '+str(C['resource_samples'])+' resource samples recorded minimum **'+str(C['min_free_vram_mib'])+' MiB free VRAM**, peak **'+str(C['max_temperature_c'])+' C** (correcting the earlier 64 C summary). Experimental on this 16 GB rig; no supported daily-driver claim.\n- CTest: '+str(C['ctest']['passed'])+' passed / '+str(C['ctest']['skipped'])+' skipped / '+str(C['ctest']['failed'])+' failed (`ple_parity`, missing Q2_0 PLE/capture fixtures). Separate `ple_reader_selftest` passed.\n- Both coding audits ran; a dictionary merge overwrote audit step names in done.json. Xhigh validity logic missed length-limited blank finals. Original status/results and logs are preserved.\n- Historical partial denominators now qualify ranking/recommendations; category comparisons use five-run means; inflated prompt rate is retired; ceiling-minus-floor gap is correctly named.\n- Separate Swift TEB v2.8.0 / 97-scenario runs are not mixed into this 92-scenario cohort. Production restoration passed; this document update made no production/model changes.')
w('## What we did NOT test\n\nVision quality, multi-day agents, other GPUs/VRAM sizes, multi-GPU, additional seeds, full knowledge datasets, Gyro-M beyond 131K, or Gyro-S on rc2. Same-evaluator historical observations are not a same-day model-only stock/fork control.')
w('## Provenance\n\n'+table(['Item','Value'],[['Stock version / digest','0.1.40.3 / 34CDE150B21148E6'],['Fork engine commit',C['manifest']['engine_commit']],['Fork binary SHA256',C['manifest']['engine_sha256']],['Gyro-M revision',C['manifest']['model_revision']],['Gyro-M SHA256',C['manifest']['model_sha256']],['Fork configured context / KV',f"{C['manifest']['context']:,} / {C['manifest']['kv']}"],['Tool evaluator',C['manifest']['teb_version']]]))
w('## Rebuild and verify\n\nPython, Chromium and Poppler are required. Install the pinned Python report requirements in your own venv:\n\n```bash\npython -m pip install -r requirements-report.txt\npython -I scripts/build_revision.py\nmkdir -p qa\npdftoppm -png -r 50 strata-community.pdf qa/page\npython -I scripts/audit_revision.py\npython -I scripts/audit_package.py\n```\n\nThe compiler reads only packaged data and rewrites PDF, charts, README, scoreboard and the shared recommendation registry. The numeric audit independently recomputes HumanEval+ task counts and all 92 x 5 tool arrays, and verifies every table starts/ends on one page using T01-style identifiers. `SOURCE-MANIFEST.json` hashes the committed files; rebuilds change PDF timestamps, so verify that manifest **before** rebuilding. Raw coding files are gzip-compressed JSON with all task records retained. See [TRIMMED.md](TRIMMED.md) for exclusions and redaction boundaries.')
(BASE/'README.md').write_text('\n\n'.join(H)+'\n')
raw=f'{FORK}/raw/{BRANCH}/{REL}';blob=f'{FORK}/blob/{BRANCH}/{REL}'
body=f'''### What this is

Updated community quality battery requested via Xavier: **10 local files, 3 quant/finetune labs, 20 local effort arms, 18 local tool arms**, on **RTX 5070 Ti 16 GB / Ryzen 9800X3D / 64 GB RAM / Windows + WSL2**. Eight stock files use **0.1.40.3 (`34CDE150B21148E6`)**; Gyro-S uses historical AgentionAI rc1 and Gyro-M rc2. Shared evaluators do not isolate model-only stock/fork differences. A separate GPT-6-luna MAX (OpenAI-hosted API reference) anchors the suite results; it is not a recommendation or a local RTX run.

**[Download the full PDF]({raw}/strata-community.pdf)** ({pages} pages) · **[GitHub report]({blob}/README.md)** · **[Canonical metrics]({blob}/data/ALL-METRICS.json)** · **[Hosted API reference detail]({blob}/data/hosted-api-reference.json)** · **[Rebuild/audit scripts]({FORK}/tree/{BRANCH}/{REL}/scripts)**.

### What changed in this revision

- Gyro-M is integrated into the common coding, tools, knowledge/debugging, economics, speed and final-board views — **no standalone model chapter**.
- GPT-6-luna MAX is integrated as a **separate OpenAI-hosted reference row/bar, score-ordered where comparable, with no medal decoration**; it is excluded from local file/arm counts, recommendations, ranks, composites and local performance claims.
- Whole tables stay on one page; section headings/explainers stay with their first result. Every table has an automated PDF-page proof.
- Final Scoreboard and Pick Your File share one generated registry: **{arm(TOP['key'],TOP['arm'])} is #{TOP['rank']} stock all-rounder**. Lower-composite reliability/latency/debugging choices are explicitly named specialists, not alternate overall winners.
- Historical partial coding denominators (155/163, 157/163, 153/162) are qualified and excluded from complete-coverage medals/composites. Five-run category means replace last-trial category values; inflated prompt throughput is retired.

### Measured headlines, with scope

- GPT-6-luna MAX external reference: HumanEval+ {f(HR['final_percent'])}% final ({f(HR['first_percent'])}% first); TEB {f(TR['mean'])} +/- {f(TR['stddev'])}, Pass@5 {f(TR['pass_at_5_percent'])}%, Pass^5 {f(TR['pass_hat_5_percent'])}%, Hard Mode {f(TR['hard_mode_mean_percent'])}%; FiveBugs {HFB['visible_passes']}/{HFB['visible_total']} visible / {HFB['hidden_passes']}/{HFB['hidden_total']} hidden. Sampled suites and full protocols are in the linked detail/README; not a local run or recommendation.
- Matched hosted decode performance is unavailable; invalid/partial rates are withheld. Provider-reported HE+ {f(HR['provider_reported_tokens_per_second'])} tok/s includes hidden reasoning and is not visible-text speed or a local decode measurement.
- Gyro-M HE+: {M['gyro-m']['heplus']['low']['passed']}/164 low ({f(M['gyro-m']['heplus']['low']['score'])}%), {M['gyro-m']['heplus']['xhigh']['passed']}/164 xhigh ({f(M['gyro-m']['heplus']['xhigh']['score'])}%). Two xhigh length-limited blank finals remain failures.
- Same pinned TEB cohort: mean {f(M['gyro-m']['trials']['low']['mean'])} +/- {f(M['gyro-m']['trials']['low']['std'])} low / {f(M['gyro-m']['trials']['xhigh']['mean'])} +/- {f(M['gyro-m']['trials']['xhigh']['std'])} xhigh; floor {f(M['gyro-m']['trials']['low']['pass_hat_5'])} / {f(M['gyro-m']['trials']['xhigh']['pass_hat_5'])}. Hard Mode improves while empirical reliability falls; no model-only causal conclusion.
- The highest stock floor remains {arm(CL['stock_best_floor']['key'],CL['stock_best_floor']['effort'])} ({f(CL['stock_best_floor']['floor'])}); Gyro-M low ties the observed floor on an experimental fork, not a stock-composite rank.
- Minimum {C['min_free_vram_mib']} MiB free VRAM, peak {C['max_temperature_c']} C. This 16 GB Gyro-M result is experimental, not a supported daily-driver recommendation.

### Validity and engine findings

Campaign **`{C['outcome']}`**: CTest {C['ctest']['passed']} passed / {C['ctest']['skipped']} skipped / {C['ctest']['failed']} failed (`ple_parity`, missing fixtures); separate reader selftest passed. Both HE audits exist, but a step-name merge defect made done.json mark them absent; blank-final validity handling was too permissive. Raw status and failures are preserved.

Gyro-S rc1 has no valid tool-eval baseline and malformed speed profiles remain n/a. Gyro-M rc2 live forced tools passed streaming/non-streaming; **Gyro-S on rc2 was not tested**. Separate Swift v2.8.0 / 97-scenario tool data is not blended in. No engine/server/default code is changed by this PR.

### What we did NOT test

Vision quality; multi-GPU; other GPUs/VRAM sizes; multi-day agents; additional seeds; full knowledge datasets; Gyro-M beyond 131K; Gyro-S on rc2. Results are same-rig historical observations, not same-day model-only stock/fork controls. For the hosted API reference, {REF['sampling_caveat']} The matched decode profile is incomplete/unavailable and all hosted throughput rates are withheld.

### Reproduce and audit

`data/ALL-METRICS.json -> scripts/build_revision.py -> PDF + README + ten charts + claims.json + FINAL-SCOREBOARD.json`. Task JSONs, all 18 raw local five-trial tool arms, measured local perf profiles, the sanitized hosted reference detail, source manifest and validation scripts are included. Hosted detail contains audited scores/protocol/source hashes only — no prompts, model outputs, thoughts or credentials. Coding raw data is losslessly gzip-compressed; secrets/rig addresses are redacted. Scope stays one report folder plus the existing community-benchmarks row. See [TRIMMED.md]({blob}/TRIMMED.md) for exclusions.

If committing a PDF is undesirable, I can move it to a release/attachment without changing the measured report.
'''
(BASE/'PR-BODY.md').write_text(body)
(BASE/'PR-TITLE.txt').write_text('Community report: 10-file quant battery on RTX 5070 Ti 16 GB (ISTA vs Swift vs Gyro-S/M)\n')
print('GITHUB_DOCS',len(RECS),'shared recommendations',pages,'PDF pages')
