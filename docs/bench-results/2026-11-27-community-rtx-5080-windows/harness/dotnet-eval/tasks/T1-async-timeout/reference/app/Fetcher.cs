namespace Eval;

public sealed class Fetcher
{
    public async Task<string> LoadAsync(
        Func<CancellationToken, Task<string>> load,
        TimeSpan timeout,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var loadTask = load(cts.Token);
        var delayTask = Task.Delay(timeout, ct);

        var finished = await Task.WhenAny(loadTask, delayTask);

        if (finished == loadTask)
        {
            return await loadTask;
        }

        ct.ThrowIfCancellationRequested();

        cts.Cancel();
        try
        {
            await loadTask;
        }
        catch (OperationCanceledException)
        {
        }
        catch
        {
        }

        throw new TimeoutException($"Load exceeded {timeout}.");
    }
}
