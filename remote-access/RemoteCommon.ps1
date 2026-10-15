function Get-StrataFunnelOwnership {
    [CmdletBinding()]
    param([object]$Configuration, [int]$Port)
    if ($null -eq $Configuration) { return 'empty' }
    if ($Configuration -isnot [System.Management.Automation.PSCustomObject] -and
        $Configuration -isnot [System.Collections.IDictionary]) { return 'other' }
    foreach ($field in @('Web', 'TCP', 'Foreground')) {
        $value = $Configuration.$field
        if ($null -ne $value -and $value -isnot [System.Management.Automation.PSCustomObject] -and
            $value -isnot [System.Collections.IDictionary]) { return 'other' }
    }
    if ($null -ne $Configuration.Foreground) {
        foreach ($child in ([pscustomobject]$Configuration.Foreground).PSObject.Properties) {
            if ((Get-StrataFunnelOwnership -Configuration $child.Value -Port $Port) -ne 'empty') { return 'other' }
        }
    }
    $tcp = ([pscustomobject]$Configuration.TCP).'443'
    $bindings = @()
    if ($null -ne $Configuration.Web) {
        $bindings = @(([pscustomobject]$Configuration.Web).PSObject.Properties | Where-Object { $_.Name.EndsWith(':443') })
    }
    if ($null -ne $tcp -and (!$tcp.HTTPS -or $tcp.TCPForward)) { return 'other' }
    if ($bindings.Count -eq 0) {
        if ($null -ne $tcp) { return 'other' }
        return 'empty'
    }
    $target = 'http://127.0.0.1:' + $Port
    foreach ($binding in $bindings) {
        if ($null -eq $binding.Value.Handlers) { return 'other' }
        $handlers = @($binding.Value.Handlers.PSObject.Properties)
        if ($handlers.Count -ne 1 -or $handlers[0].Name -ne '/' -or $handlers[0].Value.Proxy -ne $target) {
            return 'other'
        }
    }
    return 'owned'
}
