using System.Text.Json;

string? file = null;
int hours = 24;
bool asJson = false;

for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--file") { file = args[i + 1]; }
    else if (args[i] == "--hours") { hours = int.Parse(args[i + 1]); }
    else if (args[i] == "--json") { asJson = true; }
}

var payload = file is null
    ? "{}"
    : File.ReadAllText(file);

using var doc = JsonDocument.Parse(payload);
var hourly = doc.RootElement.GetProperty("hourly");
var times = hourly.GetProperty("time");

Console.WriteLine("time pm10 pm2_5 aqi band");
foreach (var t in times.EnumerateArray())
{
    Console.WriteLine(t.GetString());
}
