using System;
using System.Collections.Generic;
using System.Globalization;
using System.Windows.Forms;

// The production menu and snapshot presentation, without processes, timers or IO.
// Actions are bound by ManagedTray; the example renderer leaves them unbound.
internal sealed class ManagedMenu : IDisposable
{
    internal readonly ContextMenuStrip Menu = new ContextMenuStrip { Renderer = new StatusRenderer() };
    internal readonly StatusItem Api = new StatusItem("API: Checking"), Model = new StatusItem("Model: Unknown"),
        Lifecycle = new StatusItem("Lifecycle: Starting"), Web = new StatusItem("Web UI: Not Checked"), Error = new StatusItem("");
    internal readonly ToolStripMenuItem Profiles = new ToolStripMenuItem("Model Profile"),
        Contexts = new ToolStripMenuItem("Context"), Idles = new ToolStripMenuItem("Unload After Idle"),
        Update = new ToolStripMenuItem("Engine: Checking"), Automatic = new ToolStripMenuItem("Automatically Upgrade When Idle"),
        AppUpdate = new ToolStripMenuItem("App: Not Configured"),
        OpenConfig = new ToolStripMenuItem("Open Configuration"), Exit = new ToolStripMenuItem("Exit Tray And Stop Server"),
        OpenApp = new ToolStripMenuItem("Open Strata App"), OpenWeb = new ToolStripMenuItem("Open Web UI"),
        EngineCheck = new ToolStripMenuItem("Check Engine Updates"), EngineInstall = new ToolStripMenuItem("Install Available Engine Update"),
        AppCheck = new ToolStripMenuItem("Check App Updates"), AppInstall = new ToolStripMenuItem("Open App To Update"),
        AppNotes = new ToolStripMenuItem("Official App Release Notes"), TrayConfig = new ToolStripMenuItem("Open Tray Configuration"),
        CopyDiagnostics = new ToolStripMenuItem("Copy Diagnostics"), OpenLogs = new ToolStripMenuItem("Open Logs");
    internal readonly List<ToolStripMenuItem> Actions = new List<ToolStripMenuItem>();

    internal ManagedMenu(IEnumerable<string> names, bool hasDesktopApp)
    {
        foreach (string name in names) Profiles.DropDownItems.Add(new ToolStripMenuItem(name));
        foreach (int tokens in ManagedTray.ContextChoices)
            Contexts.DropDownItems.Add(new ToolStripMenuItem(ManagedTray.ContextLabel(tokens)) { Tag = tokens });
        foreach (int seconds in new[] { 0, 300, 600, 1800 })
            Idles.DropDownItems.Add(new ToolStripMenuItem(seconds == 0 ? "Never" : (seconds / 60) + " Minutes") { Tag = seconds });
        Menu.Items.AddRange(new ToolStripItem[] { Api, Model, Lifecycle, Web, Error, new ToolStripSeparator() });
        if (hasDesktopApp) Menu.Items.Add(OpenApp);
        Menu.Items.Add(OpenWeb);
        Menu.Items.AddRange(new ToolStripItem[] { Profiles, Contexts, Idles });
        AddAction("Start Server", "start");
        AddAction("Apply Configuration / Restart", "restart");
        AddAction("Stop Server", "stop");
        AddAction("Preload Model", "load");
        AddAction("Unload Model Now", "unload");
        Update.DropDownItems.AddRange(new ToolStripItem[] { EngineCheck, EngineInstall, Automatic });
        Menu.Items.Add(Update);
        AppUpdate.Enabled = hasDesktopApp;
        if (hasDesktopApp) AppUpdate.Text = "App: Not Checked";
        AppUpdate.DropDownItems.AddRange(new ToolStripItem[] { AppCheck, AppInstall, AppNotes });
        Menu.Items.Add(AppUpdate);
        OpenConfig.ToolTipText = "Opens The Selected Original Profile. Save, Then Use Apply Configuration / Restart.";
        OpenConfig.Enabled = false;
        Menu.Items.AddRange(new ToolStripItem[] { OpenConfig, TrayConfig, CopyDiagnostics, OpenLogs, new ToolStripSeparator(), Exit });
    }

    private void AddAction(string label, string action)
    {
        var item = new ToolStripMenuItem(label) { Tag = action };
        Menu.Items.Add(item); Actions.Add(item);
    }

    private static string DisplayState(Dictionary<string, object> state, string key)
    {
        return CultureInfo.InvariantCulture.TextInfo.ToTitleCase(ManagedTray.Text(state, key));
    }

    private static string IdleLabel(Dictionary<string, object> selection)
    {
        int seconds;
        return Int32.TryParse(ManagedTray.Text(selection, "idle"), out seconds) && seconds >= 0
            ? seconds == 0 ? "Never" : (seconds / 60) + " Minutes" : "Unknown";
    }

    internal void ApplyError(Dictionary<string, object> state)
    {
        Error.Text = ManagedTray.Text(state, "error"); Error.Visible = Error.Text != "" && Error.Text != "unknown";
    }

    internal void Apply(Dictionary<string, object> state, bool closing)
    {
        Api.Text = "API: " + DisplayState(state, "api");
        Web.Text = "Web UI: " + DisplayState(state, "web") + " (Anonymous Probe)";
        Model.Text = "Model State: " + DisplayState(state, "model");
        Lifecycle.Text = "Lifecycle: " + DisplayState(state, "phase") + "; Admitted Requests " + ManagedTray.Text(state, "active");
        object value;
        bool busy = state.TryGetValue("busy", out value) && value is bool && (bool)value;
        var selected = state.ContainsKey("selected") ? state["selected"] as Dictionary<string, object> : null;
        var effective = state.ContainsKey("effective") ? state["effective"] as Dictionary<string, object> : null;
        if (selected != null)
        {
            string profile = ManagedTray.Text(selected, "profile");
            Profiles.Text = "Profile: " + profile + " (Effective " + ManagedTray.Text(effective, "profile") + ")";
            int chosen, applied;
            Contexts.Text = "Context: " + (Int32.TryParse(ManagedTray.Text(selected, "context"), out chosen) ? ManagedTray.ContextLabel(chosen) : "Unknown") +
                " (Effective " + (Int32.TryParse(ManagedTray.Text(effective, "context"), out applied) ? ManagedTray.ContextLabel(applied) : "Unknown") + ")";
            Idles.Text = "Unload After Idle: " + IdleLabel(selected) + " (Effective " + IdleLabel(effective) + ")";
            Automatic.Checked = ManagedTray.Text(selected, "auto") == "True";
            foreach (ToolStripMenuItem item in Profiles.DropDownItems) item.Checked = item.Text == profile;
            foreach (ToolStripMenuItem item in Contexts.DropDownItems) item.Checked = ManagedTray.Text(selected, "context") == Convert.ToString(item.Tag);
            foreach (ToolStripMenuItem item in Idles.DropDownItems) item.Checked = ManagedTray.Text(selected, "idle") == Convert.ToString(item.Tag);
        }
        SetControls(!busy && !closing);
        Update.Text = "Engine: " + ManagedTray.Text(state, "version") + " (" + DisplayState(state, "update") + ")";
        EngineInstall.Enabled = ManagedTray.Text(state, "update") == "available";
    }

    internal void SetControls(bool enabled)
    {
        foreach (var item in Actions) item.Enabled = enabled;
        Profiles.Enabled = Contexts.Enabled = Idles.Enabled = Update.Enabled = enabled;
    }

    public void Dispose() { Menu.Dispose(); if (OpenApp.Owner == null) OpenApp.Dispose(); }
}
