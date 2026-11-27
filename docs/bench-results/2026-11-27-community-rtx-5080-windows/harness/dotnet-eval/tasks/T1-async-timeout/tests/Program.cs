using System.Diagnostics;
using Eval;

var checks = new List<(string name, bool ok, string detail)>();
void Check(string name, bool ok, string detail = "") => checks.Add((name, ok, detail));

var fetcher = new Fetcher();

// 1) rychla load vrati vysledok
try
{
    var v = await fetcher.LoadAsync(_ => Task.FromResult("ok"), TimeSpan.FromSeconds(5), CancellationToken.None);
    Check("fast load vrati vysledok", v == "ok", $"got={v ?? "<null>"}");
}
catch (Exception ex)
{
    Check("fast load vrati vysledok", false, ex.GetType().Name);
}

// 2) timeout -> TimeoutException, rychlo
try
{
    var sw = Stopwatch.StartNew();
    await fetcher.LoadAsync(
        async c => { await Task.Delay(3000, c); return "x"; },
        TimeSpan.FromMilliseconds(100),
        CancellationToken.None);
    Check("timeout hodi TimeoutException", false, "vratilo sa bez vynimky");
}
catch (TimeoutException)
{
    Check("timeout hodi TimeoutException", true);
}
catch (Exception ex)
{
    Check("timeout hodi TimeoutException", false, ex.GetType().Name);
}

// 3) uz zruseny token -> OperationCanceledException
try
{
    using var cts = new CancellationTokenSource();
    cts.Cancel();
    await fetcher.LoadAsync(_ => Task.FromResult("x"), TimeSpan.FromSeconds(5), cts.Token);
    Check("zruseny token -> OperationCanceledException", false, "vratilo sa bez vynimky");
}
catch (OperationCanceledException)
{
    Check("zruseny token -> OperationCanceledException", true);
}
catch (Exception ex)
{
    Check("zruseny token -> OperationCanceledException", false, ex.GetType().Name);
}

// 4) timeout musi zrusit beziacu load
var observed = false;
try
{
    await fetcher.LoadAsync(
        async c =>
        {
            try { await Task.Delay(3000, c); }
            catch (OperationCanceledException) { observed = true; throw; }
            return "x";
        },
        TimeSpan.FromMilliseconds(80),
        CancellationToken.None);
}
catch (TimeoutException) { }
catch (Exception) { }
await Task.Delay(250);
Check("timeout zrusi beziacu load", observed);

// 5) vynimka z load sa propaguje
try
{
    await fetcher.LoadAsync(
        async _ => { await Task.Yield(); throw new InvalidOperationException("boom"); },
        TimeSpan.FromSeconds(5),
        CancellationToken.None);
    Check("vynimka z load sa propaguje", false, "vratilo sa bez vynimky");
}
catch (InvalidOperationException)
{
    Check("vynimka z load sa propaguje", true);
}
catch (Exception ex)
{
    Check("vynimka z load sa propaguje", false, ex.GetType().Name);
}

foreach (var (name, ok, detail) in checks)
{
    Console.WriteLine((ok ? "PASS " : "FAIL ") + name + (detail.Length > 0 ? " :: " + detail : ""));
}
var passed = checks.Count(c => c.ok);
Console.WriteLine($"RESULT: {passed}/{checks.Count}");
return passed == checks.Count ? 0 : 1;
