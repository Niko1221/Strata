# Executable acceptance test matrix

The IDs below are **mandatory real-backend/failure gates**; they are not implemented by the bundled toy or reference store. `scripts/verify.py --gate G1` and later fail closed until every named test is discoverable, executes without skip/xfail, and gate evidence is present. Running a test that contains only `assert True` is not conformance; the independent-oracle and fault-injection requirements in `docs/certification.md` apply.

G0 is independently covered by the 9 pre-existing toy tests, 32 logical-store tests, and 10 numerical-comparison tests.

## G1 — 27 mandatory cases

| ID | Required `unittest` method | What it must assert |
|---|---|---|
| G1-001 | `test_g1_001_fresh_suffix_a` | Each token of suffix A matches a fresh uncached P||A[:i] next-logit vector. |
| G1-002 | `test_g1_002_fresh_suffix_b` | Each token of distinct suffix B matches uncached P||B[:i]; not A||B. |
| G1-003 | `test_g1_003_empty_suffix_frontier` | Snapshot P next logits agree with fresh P before any suffix is added. |
| G1-004 | `test_g1_004_every_accepted_boundary` | After each accepted generated/forced token, consumed length, positions and next logits align. |
| G1-005 | `test_g1_005_split_prefill_associativity` | Prefill P in one batch or chunks, with identical fixed tokens, yields equivalent state. |
| G1-006 | `test_g1_006_partial_page_boundary` | Fork at page_len-1, page_len, page_len+1; both children match fresh oracle. |
| G1-007 | `test_g1_007_repeated_token_history` | Same repeated token or last token in different histories cannot collide in prefix-cache identity. |
| G1-008 | `test_g1_008_same_last_block_different_prefix` | Two histories sharing the same final KV block contents but different earlier tokens never incorrectly share states. |
| G1-009 | `test_g1_009_absolute_positions` | Positions and attention visibility match a fresh reference at long and offset prefixes. |
| G1-010 | `test_g1_010_rope_and_yarn_contract` | RoPE/YaRN config changes are incompatible unless an explicitly proven converter exists. |
| G1-011 | `test_g1_011_tokenizer_and_template_contract` | Same displayed text with different exact tokenization/template identity cannot share incompatible history. |
| G1-012 | `test_g1_012_attention_mask_visibility` | Causal/window/multimodal visibility metadata is included and validated. |
| G1-013 | `test_g1_013_different_interleave_order` | Processing branches A and B in reverse/interleaved order cannot contaminate histories/logits. |
| G1-014 | `test_g1_014_pending_sampled_token` | Saving immediately after sample cannot produce sampled-but-unconsumed cache frontier. |
| G1-015 | `test_g1_015_mid_generation_rng_replay` | Seven accepted tokens, save, twenty more, restore, repeat: tokens and draws agree. |
| G1-016 | `test_g1_016_sampler_multiple_draws` | Controller stores actual RNG draw cursor if sampler uses variable draws per token. |
| G1-017 | `test_g1_017_grammar_mid_production` | Mid-grammar restoration preserves legal-token state and never bypasses grammar. |
| G1-018 | `test_g1_018_stop_progress_and_eos` | Stop matcher and EOS status survive restore; stopped scope remains stopped. |
| G1-019 | `test_g1_019_chunked_generate_equivalence` | Generate(7)+Generate(20) equals Generate(27) under same scope. |
| G1-020 | `test_g1_020_sibling_rng_independence` | One branch consuming 100 extra RNG draws cannot affect sibling stream. |
| G1-021 | `test_g1_021_alternate_stream_not_forced_different` | Different stream IDs may lead to identical tokens; API does not assert otherwise. |
| G1-022 | `test_g1_022_unsupported_hybrid_state` | Backend refuses incomplete recurrent/convolution state save rather than claiming support. |
| G1-023 | `test_g1_023_incompatible_model_rejected` | Different weights/adapters/architecture cannot import old numerical checkpoints. |
| G1-024 | `test_g1_024_execution_profile_mismatch` | A numerically incompatible checkpoint representation is rejected or safely recomputed. |
| G1-025 | `test_g1_025_true_uncached_oracle` | Independent oracle performs full-history forward with no SUT cache hits. |
| G1-026 | `test_g1_026_near_tie_numerics` | Logit comparison uses frozen tolerances and does not equate text similarity with correctness. |
| G1-027 | `test_g1_027_stream_publication_boundary` | A resumable streamed token has a durable, fully consumed matching snapshot. |

## G2 — 19 mandatory cases

| ID | Required `unittest` method | What it must assert |
|---|---|---|
| G2-001 | `test_g2_001_cold_process_reopen` | Persist prefix/branch, close process and reopen, then continue same exact history. |
| G2-002 | `test_g2_002_remove_all_kv_checkpoints` | Delete every cached numeric state and reconstruct logits/tokens from retained inputs. |
| G2-003 | `test_g2_003_checkpoint_payload_corruption` | Flip checkpoint payload byte; detect checksum mismatch and refuse/rebuild. |
| G2-004 | `test_g2_004_checkpoint_manifest_corruption` | Corrupt consumed count, model fingerprint or position map; reject/check compatibility. |
| G2-005 | `test_g2_005_partial_checkpoint_write` | Interrupt staged checkpoint write before rename; incomplete object is never discoverable. |
| G2-006 | `test_g2_006_crash_before_snapshot_publish` | Kill before snapshot publish; branch pointer remains on prior fully durable history. |
| G2-007 | `test_g2_007_crash_before_branch_rename` | Kill after staging refs+receipt before rename; old head remains intact. |
| G2-008 | `test_g2_008_crash_after_branch_rename` | Kill after refs rename before ack; retry same operation ID returns a single committed result. |
| G2-009 | `test_g2_009_idempotency_payload_mismatch` | Reusing operation ID with different parameters returns conflict, not silent success. |
| G2-010 | `test_g2_010_aba_pointer_revision` | CAS with stale revision fails after P→A→P movement. |
| G2-011 | `test_g2_011_racing_cas_processes` | Independent workers contending on stale revision yield exactly one winner. |
| G2-012 | `test_g2_012_kill_during_decode` | Fault after partial backend token ingest discards lease; no partially consumed public snapshot. |
| G2-013 | `test_g2_013_out_of_memory_transaction` | Backend allocation failure does not corrupt published history or other branches. |
| G2-014 | `test_g2_014_eviction_vs_deletion` | Cache eviction can never delete retained history; historical deletion requires distinct authorization. |
| G2-015 | `test_g2_015_checkpoint_from_wrong_model` | Checkpoint from another model/positional profile cannot be silently loaded. |
| G2-016 | `test_g2_016_checkpoint_restore_grammar_rng` | After cold restore controller state includes RNG, grammar, penalties and stop state. |
| G2-017 | `test_g2_017_missing_input_artifact` | Lost required multimodal/input artifact yields explicit failure, not fabricated replay. |
| G2-018 | `test_g2_018_external_effects_not_replayed` | Restoring saved conversation does not rerun external tools or irreversible side effects. |
| G2-019 | `test_g2_019_power_loss_scope_honest` | Fault evidence explicitly distinguishes process kill from power-loss durability. |

## G3 — 16 mandatory cases

| ID | Required `unittest` method | What it must assert |
|---|---|---|
| G3-001 | `test_g3_001_actual_prefix_work_shared` | Second hot fork shows zero duplicate full-prefix prefill work with internal counters. |
| G3-002 | `test_g3_002_unique_kv_page_accounting` | Unique prefix pages do not grow k-fold as k branches fork the same root. |
| G3-003 | `test_g3_003_partial_tail_cow` | Fork at each partial-page position: sibling append copies/makes private tail only. |
| G3-004 | `test_g3_004_shared_page_never_written` | Byte/checksum audit proves shared pages remain immutable across child generation. |
| G3-005 | `test_g3_005_quantization_scale_isolation` | Shared quantizer metadata/outlier state cannot be mutated by sibling. |
| G3-006 | `test_g3_006_recurrent_buffer_isolation` | When applicable, recurrent state is isolated or capability blocked. |
| G3-007 | `test_g3_007_evict_one_branch_keep_other` | Eviction of one numerical state leaves siblings and root reconstructible. |
| G3-008 | `test_g3_008_cpu_ram_ssd_roundtrip` | If supported, exact persisted state offload and restore agree with fresh oracle. |
| G3-009 | `test_g3_009_memory_pressure_recovery` | Forced allocation pressure uses fallback or error without branch corruption. |
| G3-010 | `test_g3_010_tail_copy_bounded` | Measured tail-copy work is bounded by page size, not the whole prefix. |
| G3-011 | `test_g3_011_fork_latency_and_work` | Fork costs O(1) logical reference work; classify physical lease activation separately. |
| G3-012 | `test_g3_012_allocation_leak_cycles` | Many fork/release/rewind cycles return allocations to stable baseline. |
| G3-013 | `test_g3_013_stress_interleaving_and_eviction` | Randomized concurrent branch updates, COW and cache eviction preserve the oracle. |
| G3-014 | `test_g3_014_corrupt_page_refusal` | Corrupt/offline cache page is never fed to attention; replay or explicit error. |
| G3-015 | `test_g3_015_counters_are_direct` | Instrumented counters reflect actual backend work, not guessed from response time. |
| G3-016 | `test_g3_016_long_context_resource_profile` | Capacity limits and scaling measured at short, medium and longest supported contexts. |

## G4 — 11 mandatory cases

| ID | Required `unittest` method | What it must assert |
|---|---|---|
| G4-001 | `test_g4_001_three_future_demonstration` | P forks into three recorded alternative suffixes that each match fresh oracle. |
| G4-002 | `test_g4_002_rewind_and_new_fourth_future` | Rewind one branch to P and take another suffix; old future stays reachable. |
| G4-003 | `test_g4_003_evaluator_read_only` | Evaluation cannot modify model history, controller, other candidates or refs. |
| G4-004 | `test_g4_004_evaluator_reproducibility` | Same evaluator code, fixture, seed and snapshot produce same declared score. |
| G4-005 | `test_g4_005_evaluator_side_effects_sandbox` | Evaluator uses sandboxed or mocked external effects and never reruns irreversible tools. |
| G4-006 | `test_g4_006_winner_adoption_cas` | Validated winner adoption moves one branch pointer exactly once. |
| G4-007 | `test_g4_007_adoption_race_loses_cleanly` | Racing stale winner adoption fails without rewriting any candidate. |
| G4-008 | `test_g4_008_evaluation_records_are_immutable` | Scores and evaluator source identity are immutable provenance records. |
| G4-009 | `test_g4_009_winning_branch_cold_replay` | Selected branch survives all numerical eviction and matches fresh oracle. |
| G4-010 | `test_g4_010_evaluation_does_not_consume_rng` | Scoring a candidate does not perturb branch-local generation draws. |
| G4-011 | `test_g4_011_retained_loser_accessible` | Nonwinning futures remain reproducibly accessible until explicit history GC. |

## G5 — 12 mandatory cases

| ID | Required `unittest` method | What it must assert |
|---|---|---|
| G5-001 | `test_g5_001_supported_matrix` | Run every advertised supported model/backend/precision/profile combination; report unsupported. |
| G5-002 | `test_g5_002_randomized_state_machine` | At least 1000 reproducible action sequences compare SUT to independent model-free and real oracles. |
| G5-003 | `test_g5_003_replay_after_upgrade` | Checkpoint migration refuses incompatible old layouts; retained inputs cold-replay under compatible model. |
| G5-004 | `test_g5_004_long_soak` | Long-duration release soak with memory/throughput/corruption stats and recorded hardware. |
| G5-005 | `test_g5_005_sanitizer_diagnostics` | Run applicable ASan/UBSan/TSan or memory-safety tools and record unsupported tools. |
| G5-006 | `test_g5_006_access_control_snapshot_ids` | Knowing a content hash does not bypass authorization to read/fork/evaluate. |
| G5-007 | `test_g5_007_encrypted_sensitive_persistence` | Access permissions/at-rest policy for potentially sensitive cache and history checked. |
| G5-008 | `test_g5_008_retention_and_gc` | Reachability-aware GC never deletes live heads or actively leased snapshots. |
| G5-009 | `test_g5_009_concurrent_gc_and_load` | GC under concurrent materialize, fork, rewind and eval cannot create dangling state. |
| G5-010 | `test_g5_010_restart_with_journal_ambiguity` | Ambiguous acknowledgements after process termination recovered by operation receipts. |
| G5-011 | `test_g5_011_negative_mutant_campaign` | All 12 specified failure mutants are detected by at least one independent test. |
| G5-012 | `test_g5_012_release_rollback` | Rollback leaves authoritative history and branch revision identities valid. |

## A concrete test pattern

```python
def test_g1_001_fresh_suffix_a(self):
    P, A = self.fixed_ids("prefix"), self.fixed_ids("suffix_a")
    branch = self.sut.create_snapshot(P)
    for i in range(len(A) + 1):
        expected = self.oracle.fresh_no_cache(P + A[:i]).next_logits
        actual = self.sut.materialize_and_get_next_logits(branch)
        assert_logits_close(actual, expected, atol=self.frozen_atol, rtol=self.frozen_rtol)
        if i < len(A):
            branch = self.sut.feed_and_commit(branch, (A[i],))
    self.assertEqual(self.sut.history(branch), P + A)
```

Do **not** silently skip a test when a model lacks a capability. Declare the **profile unsupported**, and block a gate that advertised the capability. Run a separate supported-profile matrix; inability to pass some profiles is not a reason to weaken the test.

