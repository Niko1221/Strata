using System.Net;
using System.Net.Http;
using System.Text;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Logging;

var checks = new List<(string name, bool ok, string detail)>();
void Check(string name, bool ok, string detail = "") => checks.Add((name, ok, detail));

WebApplication? app = null;
try
{
    var builder = WebApplication.CreateBuilder();
    builder.Logging.ClearProviders();
    builder.WebHost.UseUrls("http://127.0.0.1:0");

    app = builder.Build();
    Eval.TodoApi.MapTodoEndpoints(app);
    await app.StartAsync();

    var address = app.Urls.First();
    using var http = new HttpClient { BaseAddress = new Uri(address) };

    async Task<(HttpStatusCode code, string body, string? location)> Send(HttpMethod method, string path, string? json = null)
    {
        using var req = new HttpRequestMessage(method, path);
        if (json is not null)
        {
            req.Content = new StringContent(json, Encoding.UTF8, "application/json");
        }

        using var resp = await http.SendAsync(req);
        var body = await resp.Content.ReadAsStringAsync();
        return (resp.StatusCode, body, resp.Headers.Location?.ToString());
    }

    // 1) prazdny zoznam
    var r = await Send(HttpMethod.Get, "/todos");
    Check("GET /todos prazdne -> 200 []", r.code == HttpStatusCode.OK && r.body.Trim() == "[]", $"{r.code} {r.body}");

    // 2) vytvorenie
    r = await Send(HttpMethod.Post, "/todos", "{\"title\":\"Nakup\"}");
    Check("POST /todos -> 201 + Location", r.code == HttpStatusCode.Created && (r.location ?? "").Contains("/todos/"), $"{r.code} {r.location}");

    // 3) zoznam obsahuje polozku
    r = await Send(HttpMethod.Get, "/todos");
    Check("GET /todos obsahuje Nakup", r.code == HttpStatusCode.OK && r.body.Contains("Nakup"), $"{r.code} {r.body}");

    // 4) detail
    r = await Send(HttpMethod.Get, "/todos/1");
    Check("GET /todos/1 -> 200 done=false", r.code == HttpStatusCode.OK && r.body.Contains("\"done\":false"), $"{r.code} {r.body}");

    // 5) oznacenie done
    r = await Send(HttpMethod.Put, "/todos/1/done");
    Check("PUT /todos/1/done -> 204", r.code == HttpStatusCode.NoContent, $"{r.code}");

    // 6) stav po oznaceni
    r = await Send(HttpMethod.Get, "/todos/1");
    Check("GET /todos/1 -> done=true", r.code == HttpStatusCode.OK && r.body.Contains("\"done\":true"), $"{r.code} {r.body}");

    // 7) neexistujuce id
    r = await Send(HttpMethod.Get, "/todos/999");
    Check("GET /todos/999 -> 404", r.code == HttpStatusCode.NotFound, $"{r.code}");

    // 8) prazdny title
    r = await Send(HttpMethod.Post, "/todos", "{\"title\":\"\"}");
    Check("POST prazdny title -> 400", r.code == HttpStatusCode.BadRequest, $"{r.code}");

    // 9) delete
    r = await Send(HttpMethod.Delete, "/todos/1");
    var after = await Send(HttpMethod.Get, "/todos/1");
    Check("DELETE /todos/1 -> 204 a potom 404", r.code == HttpStatusCode.NoContent && after.code == HttpStatusCode.NotFound, $"{r.code} / {after.code}");
}
catch (Exception ex)
{
    Check("aplikacia sa spustila", false, ex.GetType().Name + ": " + ex.Message.Split('\n')[0]);
}
finally
{
    if (app is not null)
    {
        try { await app.StopAsync(); } catch { }
    }
}

foreach (var (name, ok, detail) in checks)
{
    Console.WriteLine((ok ? "PASS " : "FAIL ") + name + (detail.Length > 0 ? " :: " + detail : ""));
}
var passed = checks.Count(c => c.ok);
Console.WriteLine($"RESULT: {passed}/{checks.Count}");
return passed == checks.Count ? 0 : 1;
