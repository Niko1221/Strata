# Re-run of the default start's ladder rows lost to the #2065 slowdown (-Mode ladder -Contexts 131072,196608,262144).
@{
    Configs = @(
        @{ Label = 'default-rerun'; Args = @('--vision', 'off', '--drafting', 'dflash2', '--kv', 'int8') }
    )
}
