using System.Diagnostics;
using System.Net;
using System.Text.Json;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace Strata.Desktop;

internal sealed class DesktopForm : Form
{
    private readonly string root;
    private readonly string? diagnosticDirectory;
    private readonly string? suppliedKeyFile;
    private readonly string stateDirectory;
    private readonly EventWaitHandle showSignal;
    private readonly WebView2 web = new() { Dock = DockStyle.Fill, DefaultBackgroundColor = Color.FromArgb(15, 18, 20) };
    private readonly ToolStrip toolbar = new() { GripStyle = ToolStripGripStyle.Hidden, BackColor = Color.FromArgb(24, 29, 31), ForeColor = Color.White };
    private readonly ToolStripLabel status = new("Starting Strata…");
    private readonly ToolStripButton reconnect = new("Reconnect") { Enabled = false };
    private readonly ToolStripButton toTray = new("Hide to tray");
    private readonly NotifyIcon tray;
    private readonly System.Windows.Forms.Timer signals = new() { Interval = 500 };
    private readonly System.Windows.Forms.Timer diagnosticsTimeout = new() { Interval = 180000 };
    private readonly HttpClient http = new(new HttpClientHandler { UseProxy = false, AllowAutoRedirect = false }) { Timeout = TimeSpan.FromSeconds(4) };
    private readonly Dictionary<string, int> apiResponses = new();
    private readonly Stopwatch startup = Stopwatch.StartNew();
    private readonly CancellationTokenSource lifetime = new();
    private string apiKey = "";
    private bool overrideServerKey;
    private bool starting;
    private bool desktopReady;
    private bool diagnosticsRunning;
    private int injectedRequests;
    private int blockedRequests;
    private int unauthorizedResponses;
    private string startupAction = "checking";
    private string? rememberedNative;
    internal int Result { get; private set; }

    internal DesktopForm(string root, EventWaitHandle showSignal, EventWaitHandle exitSignal, string? diagnostics, string? apiKeyFile)
    {
        this.root = root;
        this.showSignal = showSignal;
        diagnosticDirectory = diagnostics;
        suppliedKeyFile = apiKeyFile;
        stateDirectory = diagnostics ?? Path.Combine(root, ".local", "desktop");
        Directory.CreateDirectory(stateDirectory);
        try
        {
            using var preferences = JsonDocument.Parse(File.ReadAllText(PreferencesPath));
            var value = preferences.RootElement.GetProperty("nativeModel").GetString();
            if (AppPolicy.NativeConfig(value) is not null) rememberedNative = value;
        }
        catch (Exception) { /* Missing preferences use the most recently used installed config. */ }
        Text = "Strata";
        StartPosition = FormStartPosition.CenterScreen;
        AutoScaleMode = AutoScaleMode.Dpi;
        MinimumSize = new Size(950, 700);
        ClientSize = new Size(Math.Min(1360, Screen.PrimaryScreen!.WorkingArea.Width - 100),
                              Math.Min(860, Screen.PrimaryScreen.WorkingArea.Height - 120));
        BackColor = Color.FromArgb(15, 18, 20);
        var iconPath = Path.Combine(AppContext.BaseDirectory, "strata.ico");
        Icon = File.Exists(iconPath) ? new Icon(iconPath) : System.Drawing.Icon.ExtractAssociatedIcon(Application.ExecutablePath);
        toolbar.Items.AddRange([status, new ToolStripSeparator(), reconnect, toTray]);
        toolbar.Padding = new Padding(10, 4, 10, 4);
        Controls.Add(web);
        Controls.Add(toolbar);
        var menu = new ContextMenuStrip();
        menu.Items.Add("Open Strata", null, (_, _) => RestoreWindow());
        menu.Items.Add("Exit app (keep models running)", null, (_, _) => Close());
        tray = new NotifyIcon { Icon = Icon, Text = "Strata — local AI", Visible = diagnostics is null, ContextMenuStrip = menu };
        tray.DoubleClick += (_, _) => RestoreWindow();
        toTray.Click += (_, _) => { Hide(); ShowInTaskbar = false; };
        reconnect.Click += async (_, _) => await ConnectAsync();
        signals.Tick += (_, _) =>
        { if (exitSignal.WaitOne(0)) Close(); else if (showSignal.WaitOne(0)) RestoreWindow(); };
        signals.Start();
        if (diagnostics is not null)
        {
            diagnosticsTimeout.Tick += (_, _) => { Result = 1; WriteState("failed", "DiagnosticsTimeout"); Close(); };
            diagnosticsTimeout.Start();
        }
        Shown += async (_, _) => await ConnectAsync();
        FormClosing += (_, _) => lifetime.Cancel();
        FormClosed += (_, _) =>
        {
            signals.Stop(); diagnosticsTimeout.Stop(); tray.Visible = false; tray.Dispose(); http.Dispose();
            WriteState("closed");
        };
    }

    private void RestoreWindow()
    {
        ShowInTaskbar = true;
        Show();
        if (WindowState == FormWindowState.Minimized) WindowState = FormWindowState.Normal;
        Activate();
    }

    private string PreferencesPath => Path.Combine(root, ".local", "desktop", "preferences.json");
    private void RememberNativeChoice(JsonElement metadata)
    {
        if (metadata.TryGetProperty("busy", out var busy) && busy.ValueKind == JsonValueKind.True) return;
        if (!metadata.TryGetProperty("current", out var model)) return;
        var value = model.GetString();
        if (AppPolicy.NativeConfig(value) is null || rememberedNative == value) return;
        rememberedNative = value;
        Directory.CreateDirectory(Path.GetDirectoryName(PreferencesPath)!);
        File.WriteAllText(PreferencesPath, JsonSerializer.Serialize(new { nativeModel = value }));
    }
    private void ReadKey()
    {
        var credential = AppPolicy.ReadCredential(root, suppliedKeyFile, rememberedNative, Environment.GetEnvironmentVariable("STRATA_API_KEY"));
        apiKey = credential.Key;
        overrideServerKey = credential.OverrideServerKey;
    }

    private async Task<bool> CheckServerAsync()
    {
        try
        {
            ReadKey();
            using var request = new HttpRequestMessage(HttpMethod.Get, AppPolicy.Origin + "/health");
            if (apiKey.Length > 0) request.Headers.Authorization = new("Bearer", apiKey);
            using var response = await http.SendAsync(request, lifetime.Token);
            if (response.StatusCode == HttpStatusCode.Unauthorized)
                throw new InvalidOperationException("The API key does not match Strata. Check your connection settings.");
            if (!response.IsSuccessStatusCode) return false;
            using var value = JsonDocument.Parse(await response.Content.ReadAsStringAsync(lifetime.Token));
            if (!value.RootElement.TryGetProperty("service", out var name) || name.GetString() != "strata")
                throw new InvalidOperationException("Another application is using Strata's local address.");
            return true;
        }
        catch (HttpRequestException) { return false; }
        catch (TaskCanceledException) when (!lifetime.IsCancellationRequested) { return false; }
        catch (JsonException) { throw new InvalidOperationException("Could not verify the local Strata server."); }
    }

    private async Task EnsureServerAsync()
    {
        if (await CheckServerAsync()) { startupAction = "reused-existing-server"; return; }
        startupAction = "started-local-launcher";
        status.Text = "Starting the local model server…";
        var info = new ProcessStartInfo(Path.Combine(root, ".venv", "Scripts", "python.exe"))
        { WorkingDirectory = root, UseShellExecute = false, CreateNoWindow = true };
        if (overrideServerKey) info.Environment["STRATA_API_KEY"] = apiKey;
        info.ArgumentList.Add(Path.Combine(root, "tools", "start_models.py"));
        info.ArgumentList.Add("--no-browser");
        if (AppPolicy.NativeConfig(rememberedNative) is { } config)
        { info.ArgumentList.Add("--model"); info.ArgumentList.Add(config); }
        using var process = Process.Start(info) ?? throw new InvalidOperationException("Could not start Strata.");
        var deadline = DateTime.UtcNow.AddSeconds(180);
        while (DateTime.UtcNow < deadline)
        {
            lifetime.Token.ThrowIfCancellationRequested();
            if (await CheckServerAsync()) return;
            if (process.HasExited)
            {
                if (await CheckServerAsync()) return;
                throw new InvalidOperationException("Could not start Strata. Check STATUS-Strata.bat in the project folder.");
            }
            await Task.Delay(800, lifetime.Token);
        }
        throw new InvalidOperationException("Strata is still starting. Wait a moment, then reconnect.");
    }

    private async Task ConnectAsync()
    {
        if (starting) return;
        starting = true;
        reconnect.Enabled = false;
        WriteState("starting");
        try
        {
            ReadKey();
            await EnsureServerAsync();
            using (var request = new HttpRequestMessage(HttpMethod.Get, AppPolicy.Origin + "/v1/status"))
            {
                if (apiKey.Length > 0) request.Headers.Authorization = new("Bearer", apiKey);
                using var response = await http.SendAsync(request, lifetime.Token);
                if (response.StatusCode == HttpStatusCode.Unauthorized)
                    throw new InvalidOperationException("The API key does not match Strata. Check your connection settings.");
                response.EnsureSuccessStatusCode();
            }
            using (var request = new HttpRequestMessage(HttpMethod.Get, AppPolicy.Origin + "/api/local-models"))
            {
                if (apiKey.Length > 0) request.Headers.Authorization = new("Bearer", apiKey);
                using var response = await http.SendAsync(request, lifetime.Token);
                if (response.IsSuccessStatusCode)
                {
                    using var metadata = JsonDocument.Parse(await response.Content.ReadAsStringAsync(lifetime.Token));
                    RememberNativeChoice(metadata.RootElement);
                }
            }
            if (web.CoreWebView2 is null)
            {
                status.Text = "Preparing the desktop view…";
                var profile = Path.Combine(stateDirectory, "profile");
                var environment = await CoreWebView2Environment.CreateAsync(null, profile);
                await web.EnsureCoreWebView2Async(environment);
                var core = web.CoreWebView2 ?? throw new InvalidOperationException("Could not initialize the desktop view.");
                core.Settings.IsPasswordAutosaveEnabled = false;
                core.Settings.IsGeneralAutofillEnabled = false;
                core.Settings.AreHostObjectsAllowed = false;
                core.Settings.AreDevToolsEnabled = diagnosticDirectory is not null;
                core.Settings.IsWebMessageEnabled = diagnosticDirectory is not null;
                core.AddWebResourceRequestedFilter("*", CoreWebView2WebResourceContext.All, CoreWebView2WebResourceRequestSourceKinds.All);
                core.WebResourceRequested += OnResourceRequested;
                core.WebResourceResponseReceived += OnResourceResponse;
                core.NavigationStarting += (_, e) =>
                {
                    if (!AppPolicy.IsLocal(e.Uri)) { e.Cancel = true; status.Text = "External pages cannot open inside the app."; }
                };
                core.FrameNavigationStarting += (_, e) => { if (!AppPolicy.IsLocal(e.Uri)) e.Cancel = true; };
                core.NewWindowRequested += (_, e) =>
                {
                    e.Handled = true;
                    if (AppPolicy.IsLocal(e.Uri)) core.Navigate(e.Uri);
                    else if (e.IsUserInitiated && Uri.TryCreate(e.Uri, UriKind.Absolute, out var target) &&
                             target.Scheme == "https" && string.IsNullOrEmpty(target.UserInfo))
                        Process.Start(new ProcessStartInfo(target.AbsoluteUri) { UseShellExecute = true });
                };
                core.PermissionRequested += (_, e) => e.State = CoreWebView2PermissionState.Deny;
                core.NavigationCompleted += OnNavigationCompleted;
                if (diagnosticDirectory is not null) core.WebMessageReceived += OnDiagnosticMessage;
                await core.AddScriptToExecuteOnDocumentCreatedAsync(
                    "if(location.origin==='" + AppPolicy.Origin + "') Object.defineProperty(window,'__STRATA_DESKTOP__',{value:true});");
                core.Navigate(AppPolicy.Origin + (diagnosticDirectory is null ? "/" : "/#about"));
            }
            else
            {
                desktopReady = false;
                apiResponses.Clear();
                web.CoreWebView2.Reload();
            }
            status.Text = "Connected locally — loading models and history…";
            Result = 0;
        }
        catch (OperationCanceledException) when (lifetime.IsCancellationRequested) { }
        catch (Exception error)
        {
            Result = 1;
            status.Text = error is InvalidOperationException ? error.Message :
                "Could not connect. Reconnect, or install WebView2 from Microsoft if it is missing.";
            WriteState("failed", error.GetType().Name);
            if (diagnosticDirectory is not null) Close();
        }
        finally { starting = false; if (!IsDisposed) reconnect.Enabled = true; }
    }

    private void OnResourceRequested(object? sender, CoreWebView2WebResourceRequestedEventArgs e)
    {
        if (AppPolicy.IsInternalImage(e.Request.Uri) &&
            (e.ResourceContext == CoreWebView2WebResourceContext.Image || e.Request.Uri.StartsWith("blob:")))
        {
            // Inline attachment previews and locally generated backup downloads need no credentials.
            e.Request.Headers.RemoveHeader("Authorization");
            return;
        }
        if (!AppPolicy.IsLocal(e.Request.Uri))
        {
            blockedRequests++;
            e.Request.Headers.RemoveHeader("Authorization");
            e.Response = web.CoreWebView2.Environment.CreateWebResourceResponse(new MemoryStream(), 403, "Blocked", "Content-Type: text/plain\r\n");
            return;
        }
        try
        {
            if (AppPolicy.NeedsAuthentication(e.Request.Uri, e.Request.Method))
            {
                ReadKey();
                if (apiKey.Length > 0) e.Request.Headers.SetHeader("Authorization", "Bearer " + apiKey);
                else e.Request.Headers.RemoveHeader("Authorization");
                e.Request.Headers.SetHeader("Cache-Control", "no-store");
                if (apiKey.Length > 0) injectedRequests++;
            }
            else e.Request.Headers.RemoveHeader("Authorization");
        }
        catch (Exception)
        {
            e.Response = web.CoreWebView2.Environment.CreateWebResourceResponse(new MemoryStream(), 401, "Unauthorized", "Content-Type: application/json\r\n");
            status.Text = "Could not read the API key. Check its location and reconnect.";
        }
    }

    private async void OnResourceResponse(object? sender, CoreWebView2WebResourceResponseReceivedEventArgs e)
    {
        if (!AppPolicy.IsLocal(e.Request.Uri)) return;
        var path = new Uri(e.Request.Uri).AbsolutePath;
        if (e.Response.StatusCode == 401) unauthorizedResponses++;
        if (path is "/metrics" or "/api/providers" or "/api/local-models" or "/api/history" or "/v1/chat/completions")
            apiResponses[path] = e.Response.StatusCode;
        if (apiResponses.GetValueOrDefault("/metrics") == 200 &&
            apiResponses.GetValueOrDefault("/api/providers") == 200 &&
            apiResponses.GetValueOrDefault("/api/history") == 200)
        {
            desktopReady = true;
            status.Text = "Connected locally   ·   Closing the app keeps models running";
            WriteState("ready");
        }
        if (path == "/api/local-models" && e.Response.StatusCode == 200)
        {
            try
            {
                using var stream = await e.Response.GetContentAsync();
                if (stream is not null)
                {
                    using var metadata = await JsonDocument.ParseAsync(stream);
                    RememberNativeChoice(metadata.RootElement);
                }
            }
            catch (Exception) { /* Preferences are optional; chats and backend selection stay intact. */ }
        }
    }

    private async void OnNavigationCompleted(object? sender, CoreWebView2NavigationCompletedEventArgs e)
    {
        if (!e.IsSuccess)
        {
            status.Text = "Could not load the page. Reconnect to try again.";
            WriteState("failed", "NavigationFailed");
            return;
        }
        if (diagnosticDirectory is null || diagnosticsRunning) return;
        diagnosticsRunning = true;
        await RunDiagnosticsAsync();
    }

    private async Task RunDiagnosticsAsync()
    {
        // This is an app-owned integration test, never a browser/OS input automation path.
        // It emits booleans and counts only; actual transcripts and credentials are excluded.
        await web.CoreWebView2.ExecuteScriptAsync("""
        (async()=>{
          const until=Date.now()+60000;
          while(Date.now()<until && (document.getElementById('model-family-select').disabled || document.getElementById('history-new').disabled))
            await new Promise(r=>setTimeout(r,100));
          const providers=await fetch('api/providers',{cache:'no-store'});
          const history=await fetch('api/history?limit=1',{cache:'no-store'});
          const catalog=history.ok?await history.json():{};
          const answer=await fetch('v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({messages:[{role:'user',content:'Reply with only this exact string: DESKTOP-READY-9037'}],
              stream:false,max_tokens:32,temperature:0,reasoning_effort:'none',chat_template_kwargs:{enable_thinking:false}})});
          const reply=answer.ok?await answer.json():{};
          const correct=reply.choices?.[0]?.message?.content?.trim()==='DESKTOP-READY-9037';
          const gif='R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
          const imageLoads=url=>new Promise(resolve=>{const image=new Image();image.onload=()=>resolve(true);image.onerror=()=>resolve(false);image.src=url;setTimeout(()=>resolve(false),2000);});
          const inlineImage=await imageLoads('data:image/gif;base64,'+gif);
          const blobUrl=URL.createObjectURL(new Blob([Uint8Array.from(atob(gif),c=>c.charCodeAt(0))],{type:'image/gif'}));
          const blobImage=await imageLoads(blobUrl); URL.revokeObjectURL(blobUrl);
          // A reserved, nonresolving origin: assert it is intercepted before any network transmission.
          await fetch('https://example.invalid/strata-desktop-test').catch(()=>{});
          const family=document.getElementById('model-family-select');
          window.chrome.webview.postMessage({kind:'strata-desktop-diagnostics',desktop:window.__STRATA_DESKTOP__===true,
            controlsReady:!family.disabled,historyReady:!document.getElementById('history-new').disabled,
            providersStatus:providers.status,historyStatus:history.status,historyCount:catalog.total??0,
            completionStatus:answer.status,completionCorrect:correct,
            selectedFamily:family.value,selectedModel:document.getElementById('model-select').value,
            nativeOptions:document.getElementById('model-select').options.length,
            additionalOptions:document.getElementById('provider-select').options.length,
            keyStored:localStorage.getItem('strata.apikey')!==null,keyFieldHidden:document.getElementById('api-key-field').hidden,
            inlineImage,blobImage});
        })().catch(()=>window.chrome.webview.postMessage({kind:'strata-desktop-diagnostics',failed:true}));
        """);
    }

    private async void OnDiagnosticMessage(object? sender, CoreWebView2WebMessageReceivedEventArgs e)
    {
        if (diagnosticDirectory is null || !AppPolicy.IsLocal(e.Source)) return;
        try
        {
            using var document = JsonDocument.Parse(e.WebMessageAsJson);
            var value = document.RootElement;
            if (value.GetProperty("kind").GetString() != "strata-desktop-diagnostics") return;
            bool Flag(string key) => value.TryGetProperty(key, out var flag) && flag.ValueKind == JsonValueKind.True;
            int Number(string key) => value.TryGetProperty(key, out var number) && number.TryGetInt32(out var result) ? result : 0;
            string TextValue(string key) => value.TryGetProperty(key, out var text) && text.ValueKind == JsonValueKind.String ? text.GetString()![..Math.Min(80, text.GetString()!.Length)] : "";
            var success = desktopReady && Flag("desktop") && Flag("controlsReady") && Flag("historyReady") &&
                Flag("completionCorrect") && Flag("keyFieldHidden") && !Flag("keyStored") && Flag("inlineImage") && Flag("blobImage") &&
                Number("providersStatus") == 200 && Number("historyStatus") == 200 && Number("completionStatus") == 200 && unauthorizedResponses == 0 && blockedRequests > 0;
            Result = success ? 0 : 1;
            using (var screenshot = File.Create(Path.Combine(diagnosticDirectory, "app-preview.png")))
                await web.CoreWebView2.CapturePreviewAsync(CoreWebView2CapturePreviewImageFormat.Png, screenshot);
            var result = new { success, timestamp = DateTimeOffset.UtcNow, processId = Environment.ProcessId,
                browserRuntime = web.CoreWebView2.Environment.BrowserVersionString, startupAction,
                desktop = Flag("desktop"), controlsReady = Flag("controlsReady"), historyReady = Flag("historyReady"),
                providersStatus = Number("providersStatus"), historyStatus = Number("historyStatus"), historyCount = Number("historyCount"),
                completionStatus = Number("completionStatus"), completionCorrect = Flag("completionCorrect"),
                selectedFamily = TextValue("selectedFamily"), selectedModel = TextValue("selectedModel"),
                nativeOptions = Number("nativeOptions"), additionalOptions = Number("additionalOptions"),
                keyStored = Flag("keyStored"), keyFieldHidden = Flag("keyFieldHidden"), injectedRequests, blockedRequests, unauthorizedResponses,
                inlineImage = Flag("inlineImage"), blobImage = Flag("blobImage"),
                windowVisible = Visible, windowWidth = Width, windowHeight = Height,
                apiResponses = new Dictionary<string, int>(apiResponses) };
            File.WriteAllText(Path.Combine(diagnosticDirectory, "diagnostics.json"), JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch (Exception) { Result = 1; WriteState("failed", "DiagnosticsFailed"); }
        finally { Close(); }
    }

    private void WriteState(string state, string? errorType = null)
    {
        try
        {
            var value = new { state, processId = Environment.ProcessId, timestamp = DateTimeOffset.UtcNow, root,
                startupAction, startupMilliseconds = startup.ElapsedMilliseconds, desktopReady,
                injectedRequests, blockedRequests, unauthorizedResponses, errorType,
                browserRuntime = web.CoreWebView2?.Environment.BrowserVersionString,
                apiResponses = new Dictionary<string, int>(apiResponses) };
            var path = Path.Combine(stateDirectory, "status.json");
            File.WriteAllText(path + ".tmp", JsonSerializer.Serialize(value, new JsonSerializerOptions { WriteIndented = true }));
            File.Move(path + ".tmp", path, true);
        }
        catch (Exception) { /* Optional diagnostic output must never interrupt a chat. */ }
    }
}
