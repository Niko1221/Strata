using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal sealed class ManagedTray : ApplicationContext
{
    private readonly JavaScriptSerializer json = new JavaScriptSerializer { MaxJsonLength = 1048576 };
    private readonly ContextMenuStrip menu = new ContextMenuStrip { Renderer = new StatusRenderer() };
    private readonly Timer timer = new Timer { Interval = 500 };
    private readonly StatusItem api = new StatusItem("API: checking"), model = new StatusItem("Model: unknown"),
        lifecycle = new StatusItem("Lifecycle: starting"), web = new StatusItem("Web UI: not checked"), error = new StatusItem("");
    private readonly ToolStripMenuItem profiles = new ToolStripMenuItem("Model profile"),
        contexts = new ToolStripMenuItem("Context"), idles = new ToolStripMenuItem("Unload after idle"),
        update = new ToolStripMenuItem("Updates: checking"), automatic = new ToolStripMenuItem("Automatically upgrade when idle"),
        openConfig = new ToolStripMenuItem("Open configuration"), exit = new ToolStripMenuItem("Exit Tray and stop server");
    private readonly Dictionary<string, string> paths = new Dictionary<string, string>();
    private readonly List<ToolStripMenuItem> actions = new List<ToolStripMenuItem>();
    private readonly Process child;
    private readonly ProcessJob job = new ProcessJob();
    private readonly NotifyIcon icon;
    private readonly Icon artwork = TrayContext.LoadIcon();
    private readonly string configPath, webUrl, logs;
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
        foreach (var pair in (Dictionary<string, object>)config["profiles"])
        {
            string name = pair.Key;
            paths.Add(name, Path.GetFullPath(Path.Combine(directory, (string)pair.Value)));
            profiles.DropDownItems.Add(name, null, delegate { Select("profile", name); });
        }
        foreach (int tokens in new[] { 65536, 131072, 262144 })
        {
            int choice = tokens;
            contexts.DropDownItems.Add((tokens / 1024) + "k", null, delegate { Select("context", choice); });
        }
        foreach (int seconds in new[] { 0, 300, 600, 1800 })
        {
            int choice = seconds;
            idles.DropDownItems.Add(seconds == 0 ? "Never" : (seconds / 60) + " min", null, delegate { Select("idle", choice); });
        }
        menu.Items.AddRange(new ToolStripItem[] { api, model, lifecycle, web, error, new ToolStripSeparator() });
        menu.Items.Add("Open Web UI", null, delegate { Open(webUrl); });
        menu.Items.AddRange(new ToolStripItem[] { profiles, contexts, idles });
        AddAction("Start server", "start");
        AddAction("Apply configuration / Restart", "restart");
        AddAction("Stop server", "stop");
        AddAction("Preload model", "load");
        AddAction("Unload model now", "unload");
        update.DropDownItems.Add("Check now", null, delegate { Send("check"); });
        update.DropDownItems.Add("Install available update", null, delegate { Send("update"); });
        automatic.Click += delegate { Send(new Dictionary<string, object> { { "action", "auto" }, { "enabled", !automatic.Checked } }); };
        update.DropDownItems.Add(automatic);
        menu.Items.Add(update);
        openConfig.Click += delegate { if (selectedProfile != null && paths.ContainsKey(selectedProfile)) Open(paths[selectedProfile]); };
        openConfig.ToolTipText = "Opens the selected original profile. Save, then use Apply configuration / Restart.";
        openConfig.Enabled = false;
        menu.Items.Add(openConfig);
        menu.Items.Add("Open Tray configuration", null, delegate { Open(configPath); });
        menu.Items.Add("Copy diagnostics", null, delegate {
            try { Clipboard.SetText(Diagnostics(snapshot)); }
            catch (System.Runtime.InteropServices.ExternalException) { Notice("Clipboard is busy. Try again."); }
        });
        menu.Items.Add("Open logs", null, delegate { Open(logs); });
        menu.Items.Add(new ToolStripSeparator());
        exit.Click += delegate { if (child.HasExited) { ExitThread(); return; } Send("exit"); closing = true; exit.Enabled = false; };
        menu.Items.Add(exit);
        icon = new NotifyIcon { Icon = artwork, Text = "Strata — starting", ContextMenuStrip = menu, Visible = true };
        icon.DoubleClick += delegate { Open(webUrl); };
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
            job.Dispose(); icon.Dispose(); artwork.Dispose(); menu.Dispose(); child.Dispose();
            throw;
        }
        timer.Tick += delegate { Refresh(); };
        timer.Start();
    }

    private void AddAction(string label, string action)
    {
        var item = new ToolStripMenuItem(label);
        item.Click += delegate { Send(action); };
        menu.Items.Add(item); actions.Add(item);
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
            error.Text = Text(snapshot, "error"); error.Visible = error.Text != "" && error.Text != "unknown";
        }
        if (child.HasExited)
        {
            if (controllerEnded) return;
            controllerEnded = true;
            if (closing) { ExitThread(); return; }
            api.Text = "API: controller stopped";
            lifecycle.Text = "Lifecycle: restart Tray to reconnect";
            foreach (var item in actions) item.Enabled = false;
            profiles.Enabled = contexts.Enabled = idles.Enabled = update.Enabled = false;
            exit.Enabled = true;
            closing = true;
            exit.Text = "Exit Tray";

            return;
        }
        if (snapshot == null) return;
        api.Text = "API: " + Text(snapshot, "api");
        web.Text = "Web UI: " + Text(snapshot, "web") + " (anonymous probe)";
        model.Text = "Model state: " + Text(snapshot, "model");
        lifecycle.Text = "Lifecycle: " + Text(snapshot, "phase") + "; admitted requests " + Text(snapshot, "active");
        object value;
        bool busy = snapshot.TryGetValue("busy", out value) && value is bool && (bool)value;
        var selected = snapshot.ContainsKey("selected") ? snapshot["selected"] as Dictionary<string, object> : null;
        var effective = snapshot.ContainsKey("effective") ? snapshot["effective"] as Dictionary<string, object> : null;
        if (selected != null)
        {
            selectedProfile = Text(selected, "profile");
            profiles.Text = "Profile: " + selectedProfile + " (effective " + Text(effective, "profile") + ")";
            contexts.Text = "Context: " + Text(selected, "context") + " (effective " + Text(effective, "context") + ")";
            idles.Text = "Idle unload: " + Text(selected, "idle") + " s (effective " + Text(effective, "idle") + ")";
            automatic.Checked = Text(selected, "auto") == "True";
            foreach (ToolStripMenuItem item in profiles.DropDownItems) item.Checked = item.Text == selectedProfile;
        }
        openConfig.Enabled = selectedProfile != null && paths.ContainsKey(selectedProfile) && File.Exists(paths[selectedProfile]);
        foreach (var item in actions) item.Enabled = !busy && !closing;
        profiles.Enabled = contexts.Enabled = idles.Enabled = update.Enabled = !busy && !closing;
        update.Text = "Updates: " + Text(snapshot, "version") + " (" + Text(snapshot, "update") + ")";
        update.DropDownItems[1].Enabled = Text(snapshot, "update") == "available";
        if (closing && !busy && error.Visible) { closing = false; exit.Enabled = true; }
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
        timer.Stop(); timer.Dispose();
        child.StandardInput.Close();
        job.Dispose(); child.Dispose();
        icon.Visible = false; icon.Dispose(); artwork.Dispose(); menu.Dispose();
        base.ExitThreadCore();
    }
}
