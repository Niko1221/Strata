# One-change screen at 256K, DDR4-3200, plus the tuned start as the last row. -Mode short -ShortContext 262144.
# Base = DFlash2, int8 KV, vision off (the ladder's winner). Each row adds one change.
@{
    Configs = @(
        @{ Label = 'base';        Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8') }
        @{ Label = 'sm75perf';    Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_SM75_PERF=1') }
        @{ Label = 'graphprio';   Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1') }
        @{ Label = 'bf16tc-simt'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
        @{ Label = 'kv-q4_0';     Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'q4_0') }
        @{ Label = 'spec4';       Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'spec=4') }
        @{ Label = 'spec8';       Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'spec=8') }
        @{ Label = 'draftconf05'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'draft-conf=0.5') }
        @{ Label = 'reserve1088'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'vram-reserve-mib=1088') }
        @{ Label = 'kvres16k';    Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'kv-resident=16384') }
        @{ Label = 'kvres64k';    Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'kv-resident=65536') }
        @{ Label = 'tuned';       Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1') }
    )
}
