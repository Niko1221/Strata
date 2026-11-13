using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

internal sealed class AppUpdateState
{
    public string Current, Latest, Phase;
    public string CheckedAt;
}

// Read the same official metadata as the desktop updater. Installation stays in
// the app, which verifies the installer's signature and owns its restart.
internal static class AppUpdatePolicy
{
    internal const string Feed = "https://github.com/Niko1221/Strata/releases/latest/download/latest.json";
    private const int MaximumBytes = 262144;

    private static int[] VersionParts(string version)
    {
        var match = Regex.Match(version ?? "", @"^(\d+)\.(\d+)\.(\d+)(?:-(beta)(?:\.(\d+))?)?$");
        if (!match.Success) throw new FormatException("The App Version Is Unavailable.");
        var parts = new int[5];
        for (int index = 0; index < 3; index++)
            if (!Int32.TryParse(match.Groups[index + 1].Value, out parts[index]))
                throw new FormatException("The App Version Is Unavailable.");
        parts[3] = match.Groups[4].Success ? 0 : 1;
        if (match.Groups[5].Success && !Int32.TryParse(match.Groups[5].Value, out parts[4]))
            throw new FormatException("The App Version Is Unavailable.");
        return parts;
    }

    internal static string Installed(string executable)
    {
        if (String.IsNullOrEmpty(executable) || !File.Exists(executable)) return null;
        string version = FileVersionInfo.GetVersionInfo(executable).ProductVersion;
        VersionParts(version);
        return version;
    }

    internal static AppUpdateState Parse(string current, string body)
    {
        var installed = VersionParts(current);
        if (body == null || Encoding.UTF8.GetByteCount(body) > MaximumBytes)
            throw new FormatException("The App Update Feed Is Too Large.");
        Dictionary<string, object> data;
        try { data = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(body); }
        catch (ArgumentException) { throw new FormatException("Invalid App Update Feed."); }
        catch (InvalidOperationException) { throw new FormatException("Invalid App Update Feed."); }
        object rawVersion, rawPlatforms, rawWindows, rawUrl, rawSignature;
        if (data == null || !data.TryGetValue("version", out rawVersion) || !(rawVersion is string) ||
            !data.TryGetValue("platforms", out rawPlatforms)) throw new FormatException("Invalid App Update Feed.");
        string latest = (string)rawVersion;
        var offered = VersionParts(latest);
        var platforms = rawPlatforms as Dictionary<string, object>;
        if (platforms == null || !platforms.TryGetValue("windows-x86_64", out rawWindows))
            throw new FormatException("No Windows App Update Is Listed.");
        var windows = rawWindows as Dictionary<string, object>;
        if (windows == null || !windows.TryGetValue("url", out rawUrl) || !(rawUrl is string) ||
            !windows.TryGetValue("signature", out rawSignature) || !(rawSignature is string) ||
            String.IsNullOrWhiteSpace((string)rawSignature)) throw new FormatException("Invalid Windows App Update.");
        string prefix = "https://github.com/Niko1221/Strata/releases/";
        string asset = "/Strata-" + latest + "-windows-update.exe";
        if ((string)rawUrl != prefix + "latest/download" + asset &&
            (string)rawUrl != prefix + "download/v" + latest + asset)
            throw new FormatException("The App Update Is Outside The Official Release.");
        int comparison = 0;
        for (int index = 0; index < installed.Length && comparison == 0; index++)
            comparison = offered[index].CompareTo(installed[index]);
        return new AppUpdateState { Current = current, Latest = latest,
            Phase = comparison > 0 ? "available" : "current", CheckedAt = DateTime.UtcNow.ToString("o") };
    }

    internal static async Task<AppUpdateState> Check(string current, HttpClient client, CancellationToken shutdown)
    {
        VersionParts(current);
        using (var timeout = CancellationTokenSource.CreateLinkedTokenSource(shutdown)) {
            timeout.CancelAfter(TimeSpan.FromSeconds(15));
            using (var request = new HttpRequestMessage(HttpMethod.Get, Feed)) {
                request.Headers.UserAgent.ParseAdd("Strata-Tray/1.0");
                using (var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token)) {
                    response.EnsureSuccessStatusCode();
                    if (response.Content.Headers.ContentLength > MaximumBytes)
                        throw new FormatException("The App Update Feed Is Too Large.");
                    using (var source = await response.Content.ReadAsStreamAsync())
                    using (var buffer = new MemoryStream()) {
                        var chunk = new byte[8192];
                        int count;
                        while ((count = await source.ReadAsync(chunk, 0, chunk.Length, timeout.Token)) > 0) {
                            if (buffer.Length + count > MaximumBytes) throw new FormatException("The App Update Feed Is Too Large.");
                            buffer.Write(chunk, 0, count);
                        }
                        return Parse(current, Encoding.UTF8.GetString(buffer.ToArray()));
                    }
                }
            }
        }
    }

    internal static string Describe(AppUpdateState state)
    {
        if (state == null || String.IsNullOrEmpty(state.Current)) return "App: Version Unavailable";
        string prefix = "App: v" + state.Current;
        switch (state.Phase) {
            case "current": return prefix + " (Up To Date)";
            case "available": return prefix + " (v" + state.Latest + " Available)";
            case "checking": return prefix + " (Checking...)";
            case "failed": return prefix + " (Check Failed)";
            default: return prefix + " (Not Checked)";
        }
    }
}
