using Eval;

var checks = new List<(string name, bool ok, string detail)>();
void Check(string name, bool ok, string detail = "") => checks.Add((name, ok, detail));

var people = new List<Customer>
{
    new(1, "Alena"),
    new(2, "Bob"),
    new(3, "Cyril")
};

Check("Normalize oreze a zmensi", Text.Normalize("  AB ") == "ab", Text.Normalize("  AB ") ?? "<null>");

try
{
    _ = Text.Normalize(null);
    Check("Normalize(null) -> ArgumentNullException", false, "vratilo sa bez vynimky");
}
catch (ArgumentNullException)
{
    Check("Normalize(null) -> ArgumentNullException", true);
}
catch (Exception ex)
{
    Check("Normalize(null) -> ArgumentNullException", false, ex.GetType().Name);
}

try
{
    _ = Text.Normalize("   ");
    Check("Normalize(medzery) -> ArgumentNullException", false, "vratilo sa bez vynimky");
}
catch (ArgumentNullException)
{
    Check("Normalize(medzery) -> ArgumentNullException", true);
}
catch (Exception ex)
{
    Check("Normalize(medzery) -> ArgumentNullException", false, ex.GetType().Name);
}

var found = Text.FindByName(people, " alena ");
Check("FindByName ignoruje medzery", found?.Id == 1, found?.Name ?? "<null>");

var upper = Text.FindByName(people, "ALENA");
Check("FindByName ignoruje velkost pismen", upper?.Id == 1, upper?.Name ?? "<null>");

Check("FindByName(null) -> null", Text.FindByName(people, null) is null);
Check("FindByName(medzery) -> null", Text.FindByName(people, "  ") is null);
Check("FindByName(neexistuje) -> null", Text.FindByName(people, "Zoe") is null);
Check("FindByName(prazdny zoznam) -> null", Text.FindByName(new List<Customer>(), "Alena") is null);

foreach (var (name, ok, detail) in checks)
{
    Console.WriteLine((ok ? "PASS " : "FAIL ") + name + (detail.Length > 0 ? " :: " + detail : ""));
}
var passed = checks.Count(c => c.ok);
Console.WriteLine($"RESULT: {passed}/{checks.Count}");
return passed == checks.Count ? 0 : 1;
