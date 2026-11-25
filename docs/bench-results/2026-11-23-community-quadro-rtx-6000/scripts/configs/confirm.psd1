# Full-fill confirmation at 256K (run_strata_speed.ps1 -Mode ladder -Contexts 262144) of the screen's winners.
@{
    Configs = @(
        @{ Label = 'base';                Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8') }
        @{ Label = 'graphprio';           Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1') }
        @{ Label = 'graphprio-q4';        Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'q4_0', '--set', 'STRATA_GRAPH_PRIO=1') }
        @{ Label = 'graphprio-q4-bf16tc'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'q4_0', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
    )
}
