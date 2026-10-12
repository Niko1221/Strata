using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal static class Program
{
    [STAThread]
    private static int Main(string[] args)
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        int port;
        if (!TryPort(args, out port))
        {
            MessageBox.Show("Usage: StrataTray.exe [--port 8095]\nConnects to an existing server on 127.0.0.1.", "Strata Tray");
            return 2;
        }
        bool first;
        using (var mutex = new Mutex(true, @"Local\Strata.ReadOnlyTray." + port, out first))
        {
            if (!first) return 0;
            using (var context = new TrayContext(port)) Application.Run(context);
        }
        return 0;
    }

    internal static bool TryPort(string[] args, out int port)
    {
        port = 8095;
        return args.Length == 0 || (args.Length == 2 && args[0] == "--port" &&
            int.TryParse(args[1], NumberStyles.None, CultureInfo.InvariantCulture, out port) && port > 0 && port <= 65535);
    }
}

internal sealed class Snapshot
{
    internal string Api = "checking", Model = "unknown";
    internal int? HttpCode;
    internal DateTime? Checked;

    internal static Snapshot Parse(string body)
    {
        var result = new Snapshot { Api = "invalid response" };
        try
        {
            var root = new JavaScriptSerializer { MaxJsonLength = 65536, RecursionLimit = 16 }
                .DeserializeObject(body) as Dictionary<string, object>;
            object value;
            if (root == null || !root.TryGetValue("data", out value)) return result;
            var data = value as object[];
            if (data == null) return result;
            if (data.Length == 0) return new Snapshot { Api = "available", Model = "not listed" };
            var model = data[0] as Dictionary<string, object>;
            if (model == null || !model.TryGetValue("id", out value) || !(value is string) || ((string)value).Length == 0) return result;
            string canonical = (string)value;
            for (int i = 1; i < data.Length; i++)
            {
                var alias = data[i] as Dictionary<string, object>;
                if (alias == null || !alias.TryGetValue("alias_of", out value) || !canonical.Equals(value)) return result;
            }
            if (!model.TryGetValue("status", out value)) return result;
            var status = value as Dictionary<string, object>;
            if (status == null || !status.TryGetValue("value", out value)) return result;
            string state = value as string;
            if (state != "loaded" && state != "unloaded" && state != "loading" && state != "failed") return result;
            result.Api = "available";
            result.Model = state;
        }
        catch (ArgumentException) { }
        catch (InvalidOperationException) { }
        return result;
    }

    internal string Diagnostics(int port)
    {
        return "Strata Tray diagnostics v1\r\n" +
            "Captured UTC: " + DateTime.UtcNow.ToString("o") + "\r\n" +
            "Tray mode: read-only; server lifecycle is managed separately\r\n" +
            "Endpoint: http://127.0.0.1:" + port + "\r\n" +
            "API: " + Api + "\r\nModel state: " + Model + "\r\n" +
            "HTTP: " + (HttpCode.HasValue ? HttpCode.Value.ToString(CultureInfo.InvariantCulture) : "unknown") + "\r\n" +
            "Checked UTC: " + (Checked.HasValue ? Checked.Value.ToString("o") : "not checked") + "\r\n" +
            "Anonymous /v1/models probe; Web UI and inference are not verified.\r\n" +
            "No API key is read or sent. Protected servers report auth required.\r\n";
    }
}

internal sealed class Probe : IDisposable
{
    private readonly HttpClient client = new HttpClient(new HttpClientHandler {
        AllowAutoRedirect = false, UseCookies = false, UseDefaultCredentials = false, UseProxy = false
    }) { Timeout = TimeSpan.FromSeconds(3) };

    internal async Task<Snapshot> Read(int port, CancellationToken cancellation)
    {
        var result = new Snapshot();
        using (var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellation))
        {
            timeout.CancelAfter(TimeSpan.FromSeconds(3));
            try
            {
                using (var response = await client.GetAsync("http://127.0.0.1:" + port + "/v1/models",
                    HttpCompletionOption.ResponseHeadersRead, timeout.Token))
                {
                    int code = (int)response.StatusCode;
                    if (code == 200)
                    {
                        using (var stream = await response.Content.ReadAsStreamAsync())
                        using (var body = new MemoryStream())
                        {
                            var buffer = new byte[4096];
                            int count;
                            while ((count = await stream.ReadAsync(buffer, 0, buffer.Length, timeout.Token)) > 0)
                            {
                                if (body.Length + count > 65536) { result.Api = "invalid response"; break; }
                                body.Write(buffer, 0, count);
                            }
                            if (result.Api != "invalid response") result = Snapshot.Parse(Encoding.UTF8.GetString(body.ToArray()));
                        }
                    }
                    else result.Api = code == 401 || code == 403 ? "auth required" :
                        code >= 300 && code < 400 ? "redirect (not followed)" : "unavailable";
                    result.HttpCode = code;
                }
            }
            catch (OperationCanceledException) { result.Api = "timeout"; }
            catch (HttpRequestException) { result.Api = "unavailable"; }
            catch (IOException) { result.Api = "unavailable"; }
            result.Checked = DateTime.UtcNow;
            return result;
        }
    }
    public void Dispose() { client.Dispose(); }
}

internal sealed class StatusItem : ToolStripMenuItem
{
    internal StatusItem(string text) : base(text) { Enabled = false; }
}

internal sealed class StatusRenderer : ToolStripProfessionalRenderer
{
    protected override void OnRenderMenuItemBackground(ToolStripItemRenderEventArgs e)
    {
        if (!(e.Item is StatusItem)) { base.OnRenderMenuItemBackground(e); return; }
        using (var brush = new SolidBrush(SystemColors.Menu))
            e.Graphics.FillRectangle(brush, new Rectangle(Point.Empty, e.Item.Size));
    }
    protected override void OnRenderItemText(ToolStripItemTextRenderEventArgs e)
    {
        if (!(e.Item is StatusItem)) { base.OnRenderItemText(e); return; }
        TextRenderer.DrawText(e.Graphics, e.Text, e.TextFont, e.TextRectangle, SystemColors.MenuText, e.TextFormat);
    }
}

internal sealed class TrayContext : ApplicationContext
{
    private readonly int port;
    private readonly Probe probe = new Probe();
    private readonly CancellationTokenSource shutdown = new CancellationTokenSource();
    private readonly System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer { Interval = 5000 };
    private readonly ContextMenuStrip menu = new ContextMenuStrip { Renderer = new StatusRenderer(), ShowItemToolTips = true };
    private readonly StatusItem api = new StatusItem("API: checking"), model = new StatusItem("Model: unknown");
    private readonly NotifyIcon icon;
    private Snapshot snapshot = new Snapshot();
    private bool polling, closing;

    internal TrayContext(int serverPort)
    {
        port = serverPort;
        menu.Items.Add(new StatusItem("Strata — 127.0.0.1:" + port));
        menu.Items.Add(api);
        menu.Items.Add(model);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add("Open Web UI", null, delegate { OpenWeb(); });
        menu.Items.Add("Refresh status", null, async delegate { await Refresh(); });
        menu.Items.Add("Copy diagnostics", null, delegate { CopyDiagnostics(); });
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add("Exit Tray (keep server running)", null, delegate { ExitThread(); });
        icon = new NotifyIcon { Icon = SystemIcons.Application, Text = "Strata — checking", ContextMenuStrip = menu, Visible = true };
        icon.DoubleClick += delegate { OpenWeb(); };
        timer.Tick += async delegate { await Refresh(); };
        timer.Start();
        // First poll starts from the UI message loop, so continuations stay on that thread.
        timer.Interval = 100;
    }

    private async Task Refresh()
    {
        if (polling || closing) return;
        polling = true;
        timer.Interval = 5000;
        try
        {
            var next = await probe.Read(port, shutdown.Token);
            if (closing) return;
            snapshot = next;
            api.Text = "API: " + snapshot.Api;
            model.Text = "Model: " + snapshot.Model;
            api.ToolTipText = "Anonymous check: " + snapshot.Checked.Value.ToString("u");
            icon.Text = "Strata — " + snapshot.Api;
        }
        finally { polling = false; }
    }

    private void OpenWeb()
    {
        try { Process.Start(new ProcessStartInfo("http://127.0.0.1:" + port + "/") { UseShellExecute = true }); }
        catch (Exception error)
        {
            if (!(error is System.ComponentModel.Win32Exception) && !(error is InvalidOperationException)) throw;
            Notice("Could not open the browser. Open http://127.0.0.1:" + port + "/ manually.");
        }
    }
    private void CopyDiagnostics()
    {
        try { Clipboard.SetText(snapshot.Diagnostics(port)); }
        catch (System.Runtime.InteropServices.ExternalException) { Notice("Clipboard is busy. Try Copy diagnostics again."); }
    }
    private void Notice(string message) { icon.ShowBalloonTip(4000, "Strata Tray", message, ToolTipIcon.Info); }
    protected override void ExitThreadCore()
    {
        closing = true;
        timer.Stop();
        shutdown.Cancel();
        icon.Visible = false;
        icon.Dispose();
        menu.Dispose();
        timer.Dispose();
        probe.Dispose();
        base.ExitThreadCore();
    }
}
