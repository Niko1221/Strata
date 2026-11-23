# The tuned setting for This PC: (Quadro RTX 6000 24 GB, 64 GB RAM): int8 KV, DFlash2, vision off,
# STRATA_GRAPH_PRIO=1 and the Turing BF16 prompt switches. Run at 256K with -Mode ladder -Contexts 262144.
@{
    Configs = @(
        @{ Label = 'tuned'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8', '--set', 'STRATA_GRAPH_PRIO=1', '--set', 'STRATA_BF16_TC=1', '--set', 'STRATA_SELECT_SIMT=1', '--save-preset', 'tuned-256k') }
    )
}
