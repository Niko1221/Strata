using System.Globalization;
using System.Text.Json;

const string DefaultUrl = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=50.0755&longitude=14.4378&hourly=pm10,pm2_5,european_aqi&timezone=Europe%2FPrague&forecast_days=2";

string? file = null;
int hours = 24;
bool asJson = false;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--file":
            if (i + 1 >= args.Length) { Console.Error.WriteLine("--file requires a path"); return 2; }
            file = args[++i];
            break;
        case "--hours":
            if (i + 1 >= args.Length || !int.TryParse(args[++i], NumberStyles.Integer, CultureInfo.InvariantCulture, out hours))
            {
                Console.Error.WriteLine("--hours requires a number");
                return 2;
            }
            break;
        case "--json":
            asJson = true;
            break;
        default:
            Console.Error.WriteLine("unknown argument: " + args[i]);
            return 2;
    }
}

if (hours <= 0)
{
    Console.Error.WriteLine("--hours must be positive");
    return 2;
}

string payload;
try
{
    if (file is null)
    {
        using var http = new HttpClient();
        payload = await http.GetStringAsync(DefaultUrl);
    }
    else
    {
        payload = await File.ReadAllTextAsync(file);
    }
}
catch (Exception ex)
{
    Console.Error.WriteLine("cannot read data: " + ex.Message);
    return 2;
}

List<Row> rows;
try
{
    rows = Parse(payload);
}
catch (Exception ex)
{
    Console.Error.WriteLine("invalid data: " + ex.Message);
    return 2;
}

int take = Math.Min(hours, rows.Count);
var window = rows.Skip(rows.Count - take).ToList();

var avgPm10 = Round1(Average(window.Select(r => r.Pm10)));
var avgPm25 = Round1(Average(window.Select(r => r.Pm25)));
var avgAqi = Round1(Average(window.Select(r => r.Aqi is int a ? (double?)a : null)));
var maxPm10 = Round1(Maximum(window.Select(r => r.Pm10)));
var maxPm25 = Round1(Maximum(window.Select(r => r.Pm25)));
var maxAqiRaw = Maximum(window.Select(r => r.Aqi is int a ? (double?)a : null));
int? maxAqi = maxAqiRaw is double m ? (int?)Math.Round(m) : null;
int? avgAqiInt = avgAqi is double aq ? (int?)Math.Round(aq) : null;

if (asJson)
{
    var o = new Dictionary<string, object?>
    {
        ["hours"] = window.Count,
        ["avg_pm10"] = avgPm10,
        ["avg_pm2_5"] = avgPm25,
        ["avg_aqi"] = avgAqi,
        ["max_pm10"] = maxPm10,
        ["max_pm2_5"] = maxPm25,
        ["max_aqi"] = maxAqi,
        ["first"] = window.Count > 0 ? window[0].Time : null,
        ["last"] = window.Count > 0 ? window[^1].Time : null
    };
    Console.WriteLine(JsonSerializer.Serialize(o));
    return 0;
}

Console.WriteLine("time pm10 pm2_5 aqi band");
foreach (var r in window)
{
    Console.WriteLine($"{r.Time} {Num(r.Pm10)} {Num(r.Pm25)} {NumAqi(r.Aqi)} {Band(r.Aqi)}");
}
Console.WriteLine($"avg {Num(avgPm10)} {Num(avgPm25)} {NumAqi(avgAqiInt)} {Band(avgAqiInt)}");
Console.WriteLine($"max {Num(maxPm10)} {Num(maxPm25)} {NumAqi(maxAqi)} {Band(maxAqi)}");
return 0;

static List<Row> Parse(string payload)
{
    using var doc = JsonDocument.Parse(payload);
    if (!doc.RootElement.TryGetProperty("hourly", out var hourly))
    {
        throw new InvalidOperationException("missing 'hourly'");
    }

    var times = hourly.TryGetProperty("time", out var t)
        ? t.EnumerateArray().Select(e => e.GetString() ?? string.Empty).ToList()
        : new List<string>();

    var pm10 = Numbers(hourly, "pm10");
    var pm25 = Numbers(hourly, "pm2_5");
    var aqi = Numbers(hourly, "european_aqi");

    if (times.Count == 0)
    {
        throw new InvalidOperationException("no hours");
    }

    var rows = new List<Row>();
    for (int i = 0; i < times.Count; i++)
    {
        var a = At(aqi, i);
        rows.Add(new Row(times[i], At(pm10, i), At(pm25, i), a is double v ? (int?)Math.Round(v, MidpointRounding.AwayFromZero) : null));
    }

    return rows;
}

static List<double?> Numbers(JsonElement hourly, string name)
{
    var list = new List<double?>();
    if (!hourly.TryGetProperty(name, out var arr) || arr.ValueKind != JsonValueKind.Array)
    {
        return list;
    }

    foreach (var e in arr.EnumerateArray())
    {
        list.Add(e.ValueKind == JsonValueKind.Number ? e.GetDouble() : null);
    }

    return list;
}

static double? At(List<double?> list, int i) => i < list.Count ? list[i] : null;

static double? Average(IEnumerable<double?> values)
{
    var xs = values.Where(v => v.HasValue).Select(v => v!.Value).ToList();
    return xs.Count == 0 ? null : xs.Average();
}

static double? Maximum(IEnumerable<double?> values)
{
    var xs = values.Where(v => v.HasValue).Select(v => v!.Value).ToList();
    return xs.Count == 0 ? null : xs.Max();
}

static double? Round1(double? v) => v is double d ? Math.Round(d, 1) : null;

static string Num(double? v) => v is double d ? d.ToString("0.0", CultureInfo.InvariantCulture) : "-";

static string NumAqi(int? v) => v is int i ? i.ToString(CultureInfo.InvariantCulture) : "-";

static string Band(int? aqi) => aqi switch
{
    null => "-",
    <= 20 => "good",
    <= 40 => "fair",
    <= 60 => "moderate",
    <= 80 => "poor",
    <= 100 => "very poor",
    _ => "extremely poor"
};

record Row(string Time, double? Pm10, double? Pm25, int? Aqi);
