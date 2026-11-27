using System.Text.Json;
using System.Globalization;

string? file = null;
int hours = 24;
bool asJson = false;

for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--file")
    {
        if (i + 1 >= args.Length)
        {
            Console.Error.WriteLine("Error: --file requires a path argument");
            Environment.Exit(2);
        }
        file = args[i + 1];
        i++;
    }
    else if (args[i] == "--hours")
    {
        if (i + 1 >= args.Length)
        {
            Console.Error.WriteLine("Error: --hours requires a numeric argument");
            Environment.Exit(2);
        }
        if (!int.TryParse(args[i + 1], out hours))
        {
            Console.Error.WriteLine("Error: --hours requires a valid integer");
            Environment.Exit(2);
        }
        i++;
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

if (hours <= 0)
{
    Console.Error.WriteLine("Error: --hours must be greater than 0");
    Environment.Exit(2);
}

string payload;
if (file is not null)
{
    if (!File.Exists(file))
    {
        Console.Error.WriteLine($"Error: file '{file}' does not exist");
        Environment.Exit(2);
    }
    payload = File.ReadAllText(file);
}
else
{
    using var client = new HttpClient();
    var url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=50.0755&longitude=14.4378&hourly=pm10,pm2_5,european_aqi&timezone=Europe%2FPrague&forecast_days=2";
    try
    {
        payload = client.GetStringAsync(url).GetAwaiter().GetResult();
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Error: failed to fetch data: {ex.Message}");
        Environment.Exit(2);
        return;
    }
}

JsonDocument doc;
try
{
    doc = JsonDocument.Parse(payload);
}
catch (JsonException)
{
    Console.Error.WriteLine("Error: invalid JSON");
    Environment.Exit(2);
    return;
}

using (doc)
{
    JsonElement hourly;
    if (!doc.RootElement.TryGetProperty("hourly", out hourly))
    {
        Console.Error.WriteLine("Error: missing 'hourly' in JSON");
        Environment.Exit(2);
        return;
    }

    JsonElement timeEl, pm10El, pm25El, aqiEl;
    if (!hourly.TryGetProperty("time", out timeEl))
    {
        Console.Error.WriteLine("Error: missing 'time' in hourly");
        Environment.Exit(2);
        return;
    }
    hourly.TryGetProperty("pm10", out pm10El);
    hourly.TryGetProperty("pm2_5", out pm25El);
    hourly.TryGetProperty("european_aqi", out aqiEl);

    var times = timeEl.EnumerateArray().Select(t => t.GetString() ?? "").ToList();
    var pm10s = pm10El.ValueKind == JsonValueKind.Array
        ? pm10El.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Number ? (double?)e.GetDouble() : null).ToList()
        : new List<double?>();
    var pm25s = pm25El.ValueKind == JsonValueKind.Array
        ? pm25El.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Number ? (double?)e.GetDouble() : null).ToList()
        : new List<double?>();
    var aqis = aqiEl.ValueKind == JsonValueKind.Array
        ? aqiEl.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Number ? (double?)e.GetDouble() : null).ToList()
        : new List<double?>();

    int totalHours = times.Count;
    int n = Math.Min(hours, totalHours);
    int start = totalHours - n;

    var selectedTimes = times.Skip(start).ToList();
    var selectedPm10 = pm10s.Skip(start).ToList();
    var selectedPm25 = pm25s.Skip(start).ToList();
    var selectedAqi = aqis.Skip(start).ToList();

    // Compute averages and maxes
    var validPm10 = selectedPm10.Where(v => v.HasValue).Select(v => v!.Value).ToList();
    var validPm25 = selectedPm25.Where(v => v.HasValue).Select(v => v!.Value).ToList();
    var validAqi = selectedAqi.Where(v => v.HasValue).Select(v => v!.Value).ToList();

    double? avgPm10 = validPm10.Count > 0 ? Math.Round(validPm10.Average(), 1) : null;
    double? avgPm25 = validPm25.Count > 0 ? Math.Round(validPm25.Average(), 1) : null;
    double? avgAqi = validAqi.Count > 0 ? Math.Round(validAqi.Average(), 1) : null;

    double? maxPm10 = validPm10.Count > 0 ? Math.Round(validPm10.Max(), 1) : null;
    double? maxPm25 = validPm25.Count > 0 ? Math.Round(validPm25.Max(), 1) : null;
    double? maxAqi = validAqi.Count > 0 ? Math.Round(validAqi.Max(), 1) : null;

    string first = selectedTimes.Count > 0 ? selectedTimes[0] : "";
    string last = selectedTimes.Count > 0 ? selectedTimes[^1] : "";

    if (asJson)
    {
        var obj = new Dictionary<string, object?>
        {
            ["hours"] = n,
            ["avg_pm10"] = avgPm10,
            ["avg_pm2_5"] = avgPm25,
            ["avg_aqi"] = avgAqi,
            ["max_pm10"] = maxPm10,
            ["max_pm2_5"] = maxPm25,
            ["max_aqi"] = maxAqi.HasValue ? (int)Math.Round(maxAqi.Value) : (int?)null,
            ["first"] = first,
            ["last"] = last
        };
        var opts = new JsonSerializerOptions { WriteIndented = false };
        Console.WriteLine(JsonSerializer.Serialize(obj, opts));
    }
    else
    {
        Console.WriteLine("time pm10 pm2_5 aqi band");
        for (int i = 0; i < selectedTimes.Count; i++)
        {
            string time = selectedTimes[i];
            string pm10Str = selectedPm10[i].HasValue ? selectedPm10[i]!.Value.ToString("0.0", CultureInfo.InvariantCulture) : "-";
            string pm25Str = selectedPm25[i].HasValue ? selectedPm25[i]!.Value.ToString("0.0", CultureInfo.InvariantCulture) : "-";
            string aqiStr = selectedAqi[i].HasValue ? ((int)Math.Round(selectedAqi[i]!.Value)).ToString() : "-";
            string band = selectedAqi[i].HasValue ? GetBand(selectedAqi[i]!.Value) : "-";
            Console.WriteLine($"{time} {pm10Str} {pm25Str} {aqiStr} {band}");
        }

        string avgPm10Str = avgPm10.HasValue ? avgPm10.Value.ToString("0.0", CultureInfo.InvariantCulture) : "-";
        string avgPm25Str = avgPm25.HasValue ? avgPm25.Value.ToString("0.0", CultureInfo.InvariantCulture) : "-";
        string avgAqiStr = avgAqi.HasValue ? avgAqi.Value.ToString("0.0", CultureInfo.InvariantCulture) : "-";
        Console.WriteLine($"avg {avgPm10Str} {avgPm25Str} {avgAqiStr} -");

        string maxPm10Str = maxPm10.HasValue ? maxPm10.Value.ToString("0.0", CultureInfo.InvariantCulture) : "-";
        string maxPm25Str = maxPm25.HasValue ? maxPm25.Value.ToString("0.0", CultureInfo.InvariantCulture) : "-";
        string maxAqiStr = maxAqi.HasValue ? ((int)Math.Round(maxAqi.Value)).ToString() : "-";
        Console.WriteLine($"max {maxPm10Str} {maxPm25Str} {maxAqiStr} -");
    }
}

static string GetBand(double aqi)
{
    if (aqi <= 20) return "good";
    if (aqi <= 40) return "fair";
    if (aqi <= 60) return "moderate";
    if (aqi <= 80) return "poor";
    if (aqi <= 100) return "very poor";
    return "extremely poor";
}
