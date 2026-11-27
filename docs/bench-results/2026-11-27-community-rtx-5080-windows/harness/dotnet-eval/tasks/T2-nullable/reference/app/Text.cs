namespace Eval;

public sealed record Customer(int Id, string Name);

public static class Text
{
    public static string Normalize(string? input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            throw new ArgumentNullException(nameof(input));
        }

        return input.Trim().ToLowerInvariant();
    }

    public static Customer? FindByName(IEnumerable<Customer> list, string? name)
    {
        ArgumentNullException.ThrowIfNull(list);

        if (string.IsNullOrWhiteSpace(name))
        {
            return null;
        }

        var wanted = name.Trim();
        return list.FirstOrDefault(c => string.Equals(c.Name, wanted, StringComparison.OrdinalIgnoreCase));
    }
}
