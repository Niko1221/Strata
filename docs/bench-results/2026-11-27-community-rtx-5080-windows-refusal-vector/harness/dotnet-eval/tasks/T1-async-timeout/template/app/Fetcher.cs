namespace Eval;

public sealed class Fetcher
{
    public async Task<string> LoadAsync(
        Func<CancellationToken, Task<string>> load,
        TimeSpan timeout,
        CancellationToken ct)
    {
        var finished = await Task.WhenAny(load(ct), Task.Delay(timeout, ct));
        return string.Empty;
    }
}
