# 3-run check at 256K (-Mode short -ShortContext 262144, DDR4-3200) of the screen's two best single-run decode rows, on the tuned start.
@{
    Configs = @(
        @{ Label = 'tuned-r1'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
        @{ Label = 'q4-r1'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'q4_0', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
        @{ Label = 'dc05-r1'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1', '--set', 'draft-conf=0.5') }
        @{ Label = 'dc05-r2'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1', '--set', 'draft-conf=0.5') }
        @{ Label = 'q4-r2'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'q4_0', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
        @{ Label = 'tuned-r2'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
        @{ Label = 'tuned-r3'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
        @{ Label = 'q4-r3'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'q4_0', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
        @{ Label = 'dc05-r3'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1', '--set', 'draft-conf=0.5') }
    )
}
