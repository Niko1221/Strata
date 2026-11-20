using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal sealed class ManagedTray : ApplicationContext
{
    // Same choices and 1024-token units as the official App's native catalog.
    internal static readonly int[] ContextChoices = { 8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288 };
    private readonly JavaScriptSerializer json = new JavaScriptSerializer { MaxJsonLength = 1048576 };
    private readonly ManagedMenu view;
    private readonly System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer { Interval = 500 };
    private readonly HttpClient appClient = new HttpClient();
    private readonly CancellationTokenSource appShutdown = new CancellationTokenSource();
    private AppUpdateState appState;
    private bool appChecking;
    private readonly Dictionary<string, string> paths = new Dictionary<string, string>();
    private readonly Process child;
    private readonly ProcessJob job = new ProcessJob();
    private readonly NotifyIcon icon;
    private readonly Icon artwork = TrayContext.LoadIcon();
    private readonly string configPath, webUrl, logs, desktopApp;
    private string latest, selectedProfile;
    private Dictionary<string, object> snapshot;
    private bool closing, controllerEnded;

    internal static string Quote(string value)
    {
        // Windows CommandLineToArgvW escaping, including trailing backslashes.
        var result = new StringBuilder("\"");
        int slash = 0;
        foreach (char c in value)
        {
            if (c == '\\') { slash++; continue; }
            if (c == '"') result.Append('\\', slash * 2 + 1);
            else result.Append('\\', slash);
            result.Append(c); slash = 0;
        }
        return result.Append('\\', slash * 2).Append('"').ToString();
    }

    internal ManagedTray(string path)
    {
        configPath = Path.GetFullPath(path);
        var config = json.Deserialize<Dictionary<string, object>>(File.ReadAllText(configPath));
        string directory = Path.GetDirectoryName(configPath);
        string python = Path.GetFullPath(Path.Combine(directory, (string)config["python"]));
        int port = config.ContainsKey("port") ? Convert.ToInt32(config["port"]) : 8095;
        if (port < 1 || port > 65535) throw new ArgumentException("Invalid port");
        webUrl = "http://127.0.0.1:" + port + "/";
        logs = Path.Combine(directory, Path.GetFileNameWithoutExtension(configPath) + ".state", "logs");
        object configuredApp;
        if (config.TryGetValue("desktop_app", out configuredApp))
        {
            if (!(configuredApp is string) || String.IsNullOrWhiteSpace((string)configuredApp))
                throw new ArgumentException("Invalid App Executable");
            desktopApp = Path.GetFullPath(Path.Combine(directory, (string)configuredApp));
            if (!File.Exists(desktopApp) || !desktopApp.EndsWith(".exe", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("App Executable Is Missing");
        }
        foreach (var pair in (Dictionary<string, object>)config["profiles"])
        {
            string name = pair.Key;
            paths.Add(name, Path.GetFullPath(Path.Combine(directory, (string)pair.Value)));
        }
        view = new ManagedMenu(paths.Keys, desktopApp != null);
        foreach (ToolStripMenuItem item in view.Profiles.DropDownItems)
        {
            string name = item.Text;
            item.Click += delegate { Select("profile", name); };
        }
        foreach (ToolStripMenuItem item in view.Contexts.DropDownItems)
        {
            int choice = (int)item.Tag;
            item.Click += delegate { Select("context", choice); };
        }
        foreach (ToolStripMenuItem item in view.Idles.DropDownItems)
        {
            int choice = (int)item.Tag;
            item.Click += delegate { Select("idle", choice); };
        }
        foreach (var item in view.Actions)
        {
            string action = (string)item.Tag;
            item.Click += delegate { Send(action); };
        }
        view.OpenApp.Click += delegate { OpenDesktop(); };
        view.OpenWeb.Click += delegate { Open(webUrl); };
        view.EngineCheck.Click += delegate { Send("check"); };
        view.EngineInstall.Click += delegate { Send("update"); };
        view.Automatic.Click += delegate { Send(new Dictionary<string, object> { { "action", "auto" }, { "enabled", !view.Automatic.Checked } }); };
        view.AppCheck.Click += async delegate { await CheckApp(); };
        view.AppInstall.Click += delegate { OpenDesktop(); };
        view.AppNotes.Click += delegate { Open("https://github.com/Niko1221/Strata/releases/latest"); };
        view.OpenConfig.Click += delegate { if (selectedProfile != null && paths.ContainsKey(selectedProfile)) Open(paths[selectedProfile]); };
        view.TrayConfig.Click += delegate { Open(configPath); };
        view.CopyDiagnostics.Click += delegate {
            try { Clipboard.SetText(Diagnostics(snapshot)); }
            catch (System.Runtime.InteropServices.ExternalException) { Notice("Clipboard is busy. Try again."); }
        };
        view.OpenLogs.Click += delegate { Open(logs); };
        view.Exit.Click += delegate { if (child.HasExited) { ExitThread(); return; } Send("exit"); closing = true; view.Exit.Enabled = false; };
        icon = new NotifyIcon { Icon = artwork, Text = "Strata — starting", ContextMenuStrip = view.Menu, Visible = true };
        icon.DoubleClick += delegate { if (desktopApp == null) Open(webUrl); else OpenDesktop(); };
        var start = new ProcessStartInfo(python, "-B -u " + Quote(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "managed", "controller.py")) + " " + Quote(configPath)) {
            UseShellExecute = false, CreateNoWindow = true, RedirectStandardInput = true,
            RedirectStandardOutput = true, RedirectStandardError = true, StandardOutputEncoding = Encoding.UTF8,
            WorkingDirectory = directory
        };
        start.EnvironmentVariables["PYTHONUTF8"] = "1";
        child = new Process { StartInfo = start };
        child.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data != null && e.Data.Length <= 65536) System.Threading.Interlocked.Exchange(ref latest, e.Data); };
        child.ErrorDataReceived += delegate { }; // Raw errors may contain config paths or credentials.
        try
        {
            child.Start();
            job.Add(child);
            child.BeginOutputReadLine(); child.BeginErrorReadLine();
            child.StandardInput.WriteLine("START"); child.StandardInput.Flush();
        }
        catch
        {
            try { if (!child.HasExited) child.Kill(); } catch (InvalidOperationException) { }
            job.Dispose(); icon.Dispose(); artwork.Dispose(); view.Dispose(); child.Dispose();
            throw;
        }
        timer.Tick += delegate { Refresh(); };
        timer.Start();
    }

    internal static string ContextLabel(int tokens) { return tokens % 1024 == 0 ? (tokens / 1024) + "K" : tokens + " Tokens"; }

    private void OpenDesktop()
    {
        if (desktopApp == null) return;
        try
        {
            var start = new ProcessStartInfo(desktopApp) { UseShellExecute = false, CreateNoWindow = true,
                WorkingDirectory = Path.GetDirectoryName(desktopApp) };
            start.EnvironmentVariables["STRATA_SERVER_URL"] = webUrl.TrimEnd('/');
            Process.Start(start);
        }
        catch (System.ComponentModel.Win32Exception) { Notice("Could Not Open Strata App."); }
    }

    private async System.Threading.Tasks.Task CheckApp()
    {
        if (appChecking || desktopApp == null) return;
        appChecking = true;
        try
        {
            string current = AppUpdatePolicy.Installed(desktopApp);
            appState = new AppUpdateState { Current = current, Phase = "checking" };
            view.AppUpdate.Text = AppUpdatePolicy.Describe(appState);
            appState = await AppUpdatePolicy.Check(current, appClient, appShutdown.Token);
        }
        catch (OperationCanceledException) { if (!appShutdown.IsCancellationRequested && appState != null) appState.Phase = "failed"; }
        catch (Exception e)
        {
            if (!(e is HttpRequestException) && !(e is FormatException) && !(e is IOException) &&
                !(e is System.ComponentModel.Win32Exception) && !(e is ArgumentException)) throw;
            if (appState != null) appState.Phase = "failed";
        }
        finally
        {
            appChecking = false;
            if (!appShutdown.IsCancellationRequested) view.AppUpdate.Text = AppUpdatePolicy.Describe(appState);
        }
    }

    private void Select(string key, object value)
    {
        Send(new Dictionary<string, object> { { "action", "select" }, { "selection", new Dictionary<string, object> { { key, value } } } });
    }
    private void Send(string action) { Send(new Dictionary<string, object> { { "action", action } }); }
    private void Send(Dictionary<string, object> command)
    {
        try { child.StandardInput.WriteLine(json.Serialize(command)); child.StandardInput.Flush(); }
        catch (Exception e) { if (!(e is IOException) && !(e is InvalidOperationException)) throw; Notice("Controller unavailable. Restart Tray to reconnect."); }
    }
    internal static string Text(Dictionary<string, object> value, string key)
    {
        object item;
        return value != null && value.TryGetValue(key, out item) && item != null ? Convert.ToString(item, System.Globalization.CultureInfo.InvariantCulture) : "unknown";
    }
    private void Refresh()
    {
        string raw = System.Threading.Interlocked.Exchange(ref latest, null);
        if (raw != null)
        {
            try { snapshot = json.Deserialize<Dictionary<string, object>>(raw); }
            catch (ArgumentException) { Notice("Invalid controller status."); }
        }
        // Output callbacks may deliver the final sanitized error after process exit.
        // Apply it on every tick, including ticks after controllerEnded.
        if (snapshot != null)
        {
            view.ApplyError(snapshot);
        }
        if (child.HasExited)
        {
            if (controllerEnded) return;
            controllerEnded = true;
            if (closing) { ExitThread(); return; }
            view.Api.Text = "API: controller stopped";
            view.Lifecycle.Text = "Lifecycle: restart Tray to reconnect";
            view.SetControls(false);
            view.Exit.Enabled = true;
            closing = true;
            view.Exit.Text = "Exit Tray";

            return;
        }
        if (snapshot == null) return;
        view.Apply(snapshot, closing);
        object value;
        bool busy = snapshot.TryGetValue("busy", out value) && value is bool && (bool)value;
        var selected = snapshot.ContainsKey("selected") ? snapshot["selected"] as Dictionary<string, object> : null;
        if (selected != null)
        {
            selectedProfile = Text(selected, "profile");
        }
        view.OpenConfig.Enabled = selectedProfile != null && paths.ContainsKey(selectedProfile) && File.Exists(paths[selectedProfile]);
        if (closing && !busy && view.Error.Visible) { closing = false; view.Exit.Enabled = true; }
        icon.Text = "Strata — " + Text(snapshot, "phase");
    }
    internal static string Diagnostics(Dictionary<string, object> state)
    {
        var output = new StringBuilder("Strata Tray diagnostics v2\r\nMode: managed\r\n");
        foreach (string key in new[] { "captured", "phase", "api", "api_checked", "model", "web", "active", "version", "update", "latest", "update_checked" })
            output.Append(key).Append(": ").Append(Text(state, key)).Append("\r\n");
        foreach (string selection in new[] { "selected", "effective" })
        {
            object value;
            var fields = state != null && state.TryGetValue(selection, out value) ? value as Dictionary<string, object> : null;
            // Profile labels and model IDs can contain private information; omit them.
            foreach (string key in new[] { "context", "idle", "auto" }) output.Append(selection).Append(' ').Append(key).Append(": ").Append(Text(fields, key)).Append("\r\n");
        }
        return output.Append("Status verifies the owned API; signed-in Web UI and inference are not verified.\r\nPaths, profile labels, credentials and raw errors are omitted.\r\n").ToString();
    }
    private void Open(string path)
    {
        try { Process.Start(new ProcessStartInfo(path) { UseShellExecute = true }); }
        catch (System.ComponentModel.Win32Exception) { Notice("Could not open the file, folder or browser. Check its default application."); }
    }
    private void Notice(string message) { icon.ShowBalloonTip(4000, "Strata Tray", message, ToolTipIcon.Info); }
    protected override void ExitThreadCore()
    {
        appShutdown.Cancel(); appClient.Dispose();
        timer.Stop(); timer.Dispose();
        child.StandardInput.Close();
        job.Dispose(); child.Dispose();
        icon.Visible = false; icon.Dispose(); artwork.Dispose(); view.Dispose();
        base.ExitThreadCore();
    }
}
