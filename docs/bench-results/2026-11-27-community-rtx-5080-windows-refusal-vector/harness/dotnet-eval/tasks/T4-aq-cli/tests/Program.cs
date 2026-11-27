using System.Diagnostics;
using System.Text.Json;

var checks = new List<(string name, bool ok, string detail)>();
void Check(string name, bool ok, string detail = "") => checks.Add((name, ok, detail));

// cesta k app projektu je posledny argument (dotnet run vie preposlat aj vlastne prepinace)
var appDir = args.Length > 0 ? args[^1] : (Environment.GetEnvironmentVariable("EVAL_APP") ?? Path.Combine("..", "app"));

// fixture: 6 hodin, jedna null, vsetky pasma (good / moderate / poor / extremely poor)
const string Fixture = """
{"hourly":{"time":["2026-11-27T00:00","2026-11-27T01:00","2026-11-27T02:00","2026-11-27T03:00","2026-11-27T04:00","2026-11-27T05:00"],"pm10":[10.0,20.5,null,30.0,5.0,100.0],"pm2_5":[8.0,15.0,null,22.0,4.0,80.0],"european_aqi":[10,45,50,65,5,120]}}
""";

var fixturePath = Path.Combine(Path.GetTempPath(), "aq-fixture-" + Guid.NewGuid().ToString("N") + ".json");
File.WriteAllText(fixturePath, Fixture);

(int code, string stdout, string stderr) Run(params string[] a)
{
    var psi = new ProcessStartInfo("dotnet")
    {
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false
    };
    psi.ArgumentList.Add("run");
    psi.ArgumentList.Add("--project");
    psi.ArgumentList.Add(appDir);
    psi.ArgumentList.Add("--no-build");
    psi.ArgumentList.Add("--");
    foreach (var x in a) psi.ArgumentList.Add(x);

    using var p = Process.Start(psi)!;
    var o = p.StandardOutput.ReadToEnd();
    var e = p.StandardError.ReadToEnd();
    p.WaitForExit(60000);
    return (p.ExitCode, o, e);
}

JsonElement? AsJson(string s)
{
    var i = s.IndexOf('{');
    if (i < 0) return null;
    try
    {
        using var d = JsonDocument.Parse(s.Substring(i));
        return d.RootElement.Clone();
    }
    catch
    {
        return null;
    }
}

// pri zlyhani ukaz stdout, a ked je prazdny, aspon stderr (inak sa nic nedozvieme)
string D((int code, string stdout, string stderr) r)
    => (r.stdout.Length > 0 ? r.stdout : ("[exit " + r.code + "] " + r.stderr)).Trim();

double N(JsonElement root, string key)
    => root.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : double.NaN;

// 1-2) zaklad: 6 hodin, priemer pm10 = (10+20.5+30+5+100)/5 = 33.1
var r1 = Run("--file", fixturePath, "--json", "--hours", "6");
var j1 = AsJson(r1.stdout);
Check("json --hours 6 -> hours=6", j1 is not null && N(j1.Value, "hours") == 6, D(r1));
Check("json --hours 6 -> avg_pm10=33.1 (null sa nerata)", j1 is not null && Math.Abs(N(j1.Value, "avg_pm10") - 33.1) < 0.001, j1 is null ? D(r1) : N(j1.Value, "avg_pm10").ToString());

// 3-4) --hours berie POSLEDNE hodiny, nie prve
var r2 = Run("--file", fixturePath, "--json", "--hours", "3");
var j2 = AsJson(r2.stdout);
Check("--hours 3 -> first=03:00 (posledne hodiny)", j2 is not null && j2.Value.TryGetProperty("first", out var f2) && f2.GetString() == "2026-11-27T03:00", j2 is null ? D(r2) : (j2.Value.TryGetProperty("first", out var ff) ? ff.GetString() ?? "" : ""));
Check("--hours 3 -> avg_pm10=45.0", j2 is not null && Math.Abs(N(j2.Value, "avg_pm10") - 45.0) < 0.001, j2 is null ? D(r2) : N(j2.Value, "avg_pm10").ToString());

// 5) viac hodin ako existuje -> vsetky
var r3 = Run("--file", fixturePath, "--json", "--hours", "99");
var j3 = AsJson(r3.stdout);
Check("--hours 99 -> hours=6 (orezane)", j3 is not null && N(j3.Value, "hours") == 6, D(r3));

// 6) maximum
var r4 = Run("--file", fixturePath, "--json");
var j4 = AsJson(r4.stdout);
Check("default -> max_aqi=120", j4 is not null && N(j4.Value, "max_aqi") == 120, j4 is null ? D(r4) : N(j4.Value, "max_aqi").ToString());

// 7-8) tabulka a pasma
var r5 = Run("--file", fixturePath, "--hours", "6");
Check("tabulka obsahuje 'extremely poor' a 'good'", r5.stdout.Contains("extremely poor") && r5.stdout.Contains("good"), r5.stdout.Trim().Split('\n')[0]);
var line3 = r5.stdout.Split('\n').FirstOrDefault(l => l.Contains("03:00")) ?? "";
Check("riadok 03:00 (aqi 65) je 'poor', nie 'extremely'", line3.Contains("poor") && !line3.Contains("extremely"), line3.Trim());

// 9) --hours 0 -> exit 2
var r7 = Run("--file", fixturePath, "--hours", "0");
Check("--hours 0 -> exit 2", r7.code == 2, "code=" + r7.code);

// 10) chybajuci subor -> exit 2
var missing = Path.Combine(Path.GetTempPath(), "aq-missing-" + Guid.NewGuid().ToString("N") + ".json");
var r8 = Run("--file", missing, "--json");
Check("chybajuci subor -> exit 2", r8.code == 2, "code=" + r8.code);

try { File.Delete(fixturePath); } catch { }

foreach (var (name, ok, detail) in checks)
{
    Console.WriteLine((ok ? "PASS " : "FAIL ") + name + (string.IsNullOrEmpty(detail) ? "" : " :: " + detail));
}
var passed = checks.Count(c => c.ok);
Console.WriteLine($"RESULT: {passed}/{checks.Count}");
return passed == checks.Count ? 0 : 1;
