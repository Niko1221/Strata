# Context ladder at DDR4-3200: default and tuned start. -Mode ladder (32K, 64K, 128K, 192K, 256K).
@{
    Configs = @(
        @{ Label = 'default'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8') }
        @{ Label = 'tuned';   Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
    )
}
