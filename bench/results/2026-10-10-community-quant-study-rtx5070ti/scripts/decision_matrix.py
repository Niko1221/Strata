"""Comprehensive evidence matrix. No aggregate score, weights or numeric ranks."""
DEFAULT = ('swift-iq3_s', 'low')
RECOMMENDATIONS = [
    {'role':'all_rounder','key':'swift-iq3_s','effort':'low','decision':'Recommended daily-driver default','reason':'Broad stock-cohort evidence: strong coding, tools, sampled knowledge, instructions and hidden debugging. Total workload wall is mid-pack (75m03s), not the fastest; chosen for breadth, not speed.'},
    {'role':'fastest_total_wall','key':'swift-q2_0','effort':'low','decision':'Fastest total workload time','reason':'Lowest total wall (code+debug+5-run tool) at 55m54s and the best stock all-five floor; sampled math/algebra and hidden debugging trail the recommended default.'},
    {'role':'reliability','key':'swift-q2_0','effort':'low','decision':'Tool-reliability specialist','reason':'Highest observed stock all-five floor, and it doubles as the fastest total wall (55m54s); weaker sampled math/algebra and hidden debugging than the recommended default.'},
    {'role':'fast_coding','key':'swift-iq3_xxs','effort':'low','decision':'Coding-latency specialist','reason':'Shortest complete coding job; total workload wall 62m31s. Not the strongest broad sampled capability profile.'},
    {'role':'coding_ceiling','key':'iq3_s','effort':'xhigh','decision':'Coding-score specialist','reason':'Highest observed stock final coding score; substantially longer total wall (160m) than low-effort Swift options.'},
    {'role':'knowledge_debug','key':'swift-iq3_s','effort':'low','decision':'Sampled knowledge / debugging choice','reason':'Highest local sampled algebra accuracy and hidden debugging score; limited samples, not a whole-model knowledge exam.'},
    {'role':'experimental_not_production','key':'gyro-m','effort':'low','decision':'Experimental comparison, not a default pick','reason':'Scored quality is useful, but narrow 16 GB memory margin, raw safety-gate failures, CTest/campaign caveats and the slowest total wall (172m at xhigh) prevent a daily-driver endorsement.'},
]

def build_matrix(data):
    rows=[]
    for key,m in data['models'].items():
        legacy=key=='gyro-s' and m.get('auxiliary_cohort')=='rc1'
        for effort in ['low','xhigh']:
            he=m['heplus'][effort];tool=m['trials'][effort]
            tool_wall=tool.get('wall_s') or (m.get('trials_wall_min') or {}).get(effort,0)*60
            total_wall=he['total_time']+m['fivebugs']['wall_s']+tool_wall
            sampled={'gsm8k':m['knowledge']['gsm8k']['accuracy'],'algebra':m['knowledge']['mmlu']['accuracy'],'ifeval_prompt':m['knowledge']['ifeval']['prompt_accuracy'],'ifeval_instruction':m['knowledge']['ifeval']['instruction_accuracy'],'debug_visible':m['fivebugs']['vis'],'debug_hidden':m['fivebugs']['hid'],'debug_wall_s':m['fivebugs']['wall_s'],'needle_accuracy':m['needle']['retrieval_accuracy'],'needle_context_tokens':m['needle']['effective_context']}
            verdict='Recommended default' if (key,effort)==DEFAULT else ('Experimental; not default' if m['cohort']!='stock' else ('Stock alternative' if effort=='low' else 'Task-specific effort'))
            rows.append({'key':key,'arm':effort,'label':f"{m['short']} ({m['lab']})",'cohort':m['cohort'],'coding_first':he['first_attempt_score'],'coding_final':he['score'],'coding_passed':he['passed'],'coding_scored':he['scored_total'],'coding_wall_s':he['total_time'],'coding_output_tokens':he['total_output_tokens'],'tool_wall_s':tool_wall,'total_wall_s':total_wall,'tool_mean':tool['mean'],'tool_sd':tool['std'],'tool_floor':tool['pass_hat_5'],'tool_ceiling':tool['pass_at_5'],'hardmode':tool['cats']['P'],'stored_rating':tool['rating'],'warning_count_last_trial':tool['n_warnings'],'raw_safety_gate_passed':tool.get('raw_safety_gate_passed'),'sampled':None if legacy else sampled,'legacy_sampled':sampled if legacy else None,'sampled_scope':'legacy rc1; not evidence for current rc2 setup' if legacy else ('rc2 xhigh sample once per file; repeated for context, not a low-effort measurement' if key=='gyro-s' else 'one per-file sample; not repeated separately at both efforts'),'source_files':{k:v['_file'] for k,v in m['knowledge'].items()}|{'needle':m['needle']['_file']},'verdict':verdict,'deployment_note':'Only 205 MiB minimum free VRAM; CTest ple_parity failure; completed_with_invalid_slices; decode unverified' if key=='gyro-m' else ('rc2 math/instructions/debug/recall/algebra verified; experimental fork, not daily-default endorsement' if key=='gyro-s' else 'Stock study setup; no long-duration stability or comprehensive safety certification')})
    # Recommendation first, then stock alternatives, higher-effort options, experimental setups.
    # This is a display order, explicitly NOT a quality rank.
    rows.sort(key=lambda r:(0 if (r['key'],r['arm'])==DEFAULT else (3 if r['cohort']!='stock' else (1 if r['arm']=='low' else 2)),r['key'],r['arm']))
    assert len(rows)==20 and len({(r['key'],r['arm']) for r in rows})==20
    return rows

POLICY = {'method':'Evidence-based daily-driver decision matrix; no master score or numeric rank','default':{'key':DEFAULT[0],'effort':DEFAULT[1]},'weights':None,'recommendation_confidence':'provisional, benchmark-backed recommendation for this rig; not long-term stability certification','coverage_policy':'Use every tested family; never mix Gyro-S rc1 sampled suites into rc2 evidence, impute missing data, double-count shared per-file samples as effort repetitions, or equate tool warnings with safety certification. TPS stays in separate diagnostics.','deployment_policy':'A capability result alone cannot override observed memory, campaign-validity, safety-gate and matched-coverage caveats. Existing stock evidence is preferred over experimental fork-only wins for the default recommendation. This report does not change the deployed model.'}
