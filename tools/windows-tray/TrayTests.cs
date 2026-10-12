using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Windows.Forms;

internal static class TrayTests
{
    [DllImport("user32.dll")] private static extern IntPtr GetThreadDpiAwarenessContext();
    [DllImport("user32.dll")] private static extern bool AreDpiAwarenessContextsEqual(IntPtr a, IntPtr b);
    private static int checks;
    private static void Check(bool pass, string label) { checks++; if (!pass) throw new Exception(label); }
    private static string Body(string state) { return "{\"data\":[{\"id\":\"PRIVATE_MODEL_SENTINEL\",\"status\":{\"value\":\"" + state + "\"},\"secret\":\"SECRET_SENTINEL\"}]}"; }
    private static Snapshot Fixture(int status, string body, string extra)
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        var server = Task.Run(delegate {
            using (var connection = listener.AcceptTcpClient())
            using (var stream = connection.GetStream())
            {
                stream.ReadTimeout = 5000;
                var request = new byte[4096];
                stream.Read(request, 0, request.Length);
                byte[] payload = Encoding.UTF8.GetBytes(body);
                byte[] header = Encoding.ASCII.GetBytes("HTTP/1.1 " + status + " Fixture\r\nConnection: close\r\nContent-Length: " + payload.Length + "\r\n" + extra + "\r\n");
                stream.Write(header, 0, header.Length);
                stream.Write(payload, 0, payload.Length);
            }
        });
        try
        {
            using (var probe = new Probe())
            {
                var result = probe.Read(port, CancellationToken.None).GetAwaiter().GetResult();
                server.GetAwaiter().GetResult();
                Check(result.Checked.HasValue, "timestamp on every completed probe");
                return result;
            }
        }
        finally { listener.Stop(); }
    }
    [STAThread]
    private static int Main(string[] args)
    {
        Application.EnableVisualStyles();
        using (var menu = new ContextMenuStrip()) { menu.CreateControl(); }
        SynchronizationContext.SetSynchronizationContext(null);
        Check(AreDpiAwarenessContextsEqual(GetThreadDpiAwarenessContext(), new IntPtr(-4)), "actual PerMonitorV2 process");
        int port;
        using (var artwork = TrayContext.LoadIcon()) Check(artwork.Width > 0 && artwork.Height > 0, "embedded favicon loads after resource stream closes");
        Check(Program.TryPort(new string[0], out port) && port == 8095, "official default port");
        Check(Program.TryPort(new[] { "--port", "43210" }, out port) && port == 43210, "custom port");
        foreach (string bad in new[] { "0", "65536", "-1", "localhost", "8095/path" })
            Check(!Program.TryPort(new[] { "--port", bad }, out port), "reject non-port: " + bad);
        Check(!Program.TryPort(new[] { "--host", "0.0.0.0" }, out port), "no arbitrary host input");
        foreach (string state in new[] { "loaded", "unloaded", "loading", "failed" })
        {
            var result = Snapshot.Parse(Body(state));
            Check(result.Api == "available" && result.Model == state, "model state " + state);
            Check(!result.Diagnostics(8095).Contains("SENTINEL"), "omit raw model IDs and other response fields");
        }
        foreach (string bad in new[] { "not JSON", "null", "{}", Body("made-up"), "{\"data\":[{}]}" })
            Check(Snapshot.Parse(bad).Api == "invalid response", "reject invalid model response");
        Check(Snapshot.Parse("{\"data\":[]}").Model == "not listed", "empty official list is not unloaded or failure");
        Check(Snapshot.Parse("{\"data\":[{\"id\":\"a\",\"status\":{\"value\":\"loaded\"}},{\"id\":\"alias\",\"alias_of\":\"a\"}]}").Model == "loaded", "official aliases accepted");
        using (var row = new StatusItem("API: available")) Check(!row.Enabled, "status rows stay noninteractive");
        Check(new Snapshot().Diagnostics(8095).Contains("not checked"), "honest initial diagnostics");
        Check(Fixture(200, Body("loaded"), "").Model == "loaded", "HTTP parse");
        Check(Fixture(200, "<html>login</html>", "").Api == "invalid response", "HTML is not healthy API");
        foreach (int code in new[] { 401, 403 })
        {
            var result = Fixture(code, "SECRET_SENTINEL", "");
            Check(result.Api == "auth required" && result.Model == "unknown", "authentication boundary");
            Check(!result.Diagnostics(8095).Contains("SECRET_SENTINEL"), "omit error response body");
        }
        Check(Fixture(302, "", "Location: http://127.0.0.1:1/\r\n").Api == "redirect (not followed)", "no redirect follow");
        Check(Fixture(503, "", "").Api == "unavailable", "HTTP failure");
        Check(Fixture(200, new string('x', 65537), "").Api == "invalid response", "response size limit");
        using (var probe = new Probe())
        using (var stop = new CancellationTokenSource())
        {
            stop.Cancel();
            Check(probe.Read(8095, stop.Token).GetAwaiter().GetResult().Api == "timeout", "cancellation bounded");
        }
        if (args.Length == 1)
        {
            using (var probe = new Probe())
            {
                var live = probe.Read(int.Parse(args[0]), CancellationToken.None).GetAwaiter().GetResult();
                Check(live.Api == "available" && live.Model == "loaded", "official mock server compatibility");
            }
        }
        Console.WriteLine(checks + " Tray checks passed.");
        return 0;
    }
}
