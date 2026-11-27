using System.Text.Json;

string? file = null;
int hours = 24;
bool asJson = false;

for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--file")
    {
        if (i + 1 >= args.Length) { Console.Error.WriteLine("Error: --file requires a path argument"); Environment.Exit(2); }
        file = args[i + 1];
    }
    else if (args[i] == "--hours")
    {
        if (i + 1 >= args.Length) { Console.Error.WriteLine("Error: --hours requires a numeric argument"); Environment.Exit(2); }
        if (!int.TryParse(args[i + 1], out hours)) { Console.Error.WriteLine("Error: --hours must be an integer"); Environment.Exit(2); }
    }
    else if (args[i] == "--json")
    {
        asJson = true;
    }
    else
    {
        Console.Error.WriteLine($"Error: unknown argument '{args[i]}'");
        Environment.Exit(2);
    }
}

if (hours <= 0) { Console.Error.WriteLine("Error: --hours must be greater than 0"); Environment.Exit(2); }

string payload;
if (file is not null)
{
    if (!File.Exists(file)) { Console.Error.WriteLine($"Error: file '{file}' not found"); Environment.Exit(2); }
    payload = File.ReadAllText(file);
}
else
{
    using var http = new HttpClient();
    payload = http.GetStringAsync("https://air-quality-api.open-meteo.com/v1/air-quality?latitude=50.0755&longitude=14.4378&hourly=pm10,pm2_5,european_aqi&timezone=Europe%2FPrague&forecast_days=2").GetAwaiter().GetResult();
}

JsonDocument doc;
try { doc = JsonDocument.Parse(payload); }
catch { Console.Error.WriteLine("Error: invalid JSON"); Environment.Exit(2); return; }

var hourly = doc.RootElement.GetProperty("hourly");
var times = hourly.GetProperty("time").EnumerateArray().ToList();
var pm10Arr = hourly.GetProperty("pm10").EnumerateArray().ToList();
var pm25Arr = hourly.GetProperty("pm2_5").EnumerateArray().ToList();
var aqiArr = hourly.GetProperty("european_aqi").EnumerateArray().ToList();

int total = times.Count;
int start = Math.Max(0, total - hours);

var selected = new List<(string time, double? pm10, double? pm25, int? aqi)>();
for (int i = start; i < total; i++)
{
    double? pm10 = i < pm10Arr.Count && pm10Arr[i].ValueKind != JsonValueKind.Null ? pm10Arr[i].GetDouble() : null;
    double? pm25 = i < pm25Arr.Count && pm25Arr[i].ValueKind != JsonValueKind.Null ? pm25Arr[i].GetDouble() : null;
    int? aqi = i < aqiArr.Count && aqiArr[i].ValueKind != JsonValueKind.Null ? aqiArr[i].GetInt32() : null;
    selected.Add((times[i].GetString()!, pm10, pm25, aqi));
}

double? avgPm10 = null, avgPm25 = null, avgAqi = null;
double? maxPm10 = null, maxPm25 = null;
int? maxAqi = null;

var pm10Vals = selected.Where(s => s.pm10.HasValue).Select(s => s.pm10!.Value).ToList();
var pm25Vals = selected.Where(s => s.pm25.HasValue).Select(s => s.pm25!.Value).ToList();
var aqiVals = selected.Where(s => s.aqi.HasValue).Select(s => (double)s.aqi!.Value).ToList();

if (pm10Vals.Count > 0) { avgPm10 = Math.Round(pm10Vals.Average(), 1); maxPm10 = Math.Round(pm10Vals.Max(), 1); }
if (pm25Vals.Count > 0) { avgPm25 = Math.Round(pm25Vals.Average(), 1); maxPm25 = Math.Round(pm25Vals.Max(), 1); }
if (aqiVals.Count > 0) { avgAqi = Math.Round(aqiVals.Average(), 1); maxAqi = (int)Math.Round(aqiVals.Max()); }

string Band(int? aqi)
{
    if (aqi is null) return "-";
    if (aqi <= 20) return "good";
    if (aqi <= 40) return "fair";
    if (aqi <= 60) return "moderate";
    if (aqi <= 80) return "poor";
    if (aqi <= 100) return "very poor";
    return "extremely poor";
}

string Fmt(double? v) => v.HasValue ? v.Value.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture) : "-";
string FmtInt(int? v) => v.HasValue ? v.Value.ToString() : "-";

if (asJson)
{
    var obj = new Dictionary<string, object?>
    {
        ["hours"] = selected.Count,
        ["avg_pm10"] = avgPm10,
        ["avg_pm2_5"] = avgPm25,
        ["avg_aqi"] = avgAqi,
        ["max_pm10"] = maxPm10,
        ["max_pm2_5"] = maxPm25,
        ["max_aqi"] = maxAqi,
        ["first"] = selected.Count > 0 ? selected[0].time : null,
        ["last"] = selected.Count > 0 ? selected[^1].time : null
    };
    Console.WriteLine(JsonSerializer.Serialize(obj, new JsonSerializerOptions { WriteIndented = false }));
}
else
{
    Console.WriteLine("time pm10 pm2_5 aqi band");
    foreach (var (time, pm10, pm25, aqi) in selected)
    {
        Console.WriteLine($"{time} {Fmt(pm10)} {Fmt(pm25)} {FmtInt(aqi)} {Band(aqi)}");
    }
    Console.WriteLine($"avg {Fmt(avgPm10)} {Fmt(avgPm25)} {Fmt(avgAqi)} -");
    Console.WriteLine($"max {Fmt(maxPm10)} {Fmt(maxPm25)} {FmtInt(maxAqi)} -");
}
