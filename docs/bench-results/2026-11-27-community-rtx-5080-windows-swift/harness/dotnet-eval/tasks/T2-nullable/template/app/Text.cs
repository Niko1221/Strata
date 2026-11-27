namespace Eval;

public sealed record Customer(int Id, string Name);

public static class Text
{
    public static string Normalize(string? input)
        => input!.Trim().ToLowerInvariant();

    public static Customer? FindByName(IEnumerable<Customer> list, string? name)
        => list.FirstOrDefault(c => c.Name == name);
}
