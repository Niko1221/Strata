# 16K run at DDR4-3200: DFlash2 vs MTP (as before) plus the tuned start. -Mode short.
@{
    Configs = @(
        @{ Label = 'dflash2-int8'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8') }
        @{ Label = 'mtp-int8';     Args = @('--vision', 'off', '--drafting', 'mtp', '--kv', 'int8') }
        @{ Label = 'dflash2-tuned'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
    )
}
