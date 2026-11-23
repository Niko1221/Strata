# Repeat pass for the community report: default vs tuned start, three runs each, order A B B A A B.
# Run with -Mode ladder -Contexts 32768,131072,262144. Each label is one run at each of the three sizes.
@{
    Configs = @(
        @{ Label = 'default-r1'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8') }
        @{ Label = 'tuned-r1';   Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
        @{ Label = 'tuned-r2';   Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
        @{ Label = 'default-r2'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8') }
        @{ Label = 'default-r3'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8') }
        @{ Label = 'tuned-r3';   Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
    )
}
