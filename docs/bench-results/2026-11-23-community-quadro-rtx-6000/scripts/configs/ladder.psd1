# Context ladder configs for run_strata_speed.ps1 -Mode ladder.
# Vision is off in every config: the llama-server runs in MoE loaded no image projector, and the
# picture reader takes VRAM that Strata would otherwise use for cached experts.
@{
    Configs = @(
        @{ Label = 'dflash2-int8'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8') }
        @{ Label = 'mtp-int8';     Args = @('--vision', 'off', '--drafting', 'mtp', '--kv', 'int8') }
    )
}
