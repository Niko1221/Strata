using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Strata.Desktop;

internal static class AppPolicy
{
    internal sealed record Credential(string Key, bool OverrideServerKey);
    internal const string Origin = "http://127.0.0.1:8080";
    internal static bool IsLocal(string? address) =>
        Uri.TryCreate(address, UriKind.Absolute, out var uri) &&
        uri.Scheme == "http" && uri.Host == "127.0.0.1" && uri.Port == 8080 &&
        string.IsNullOrEmpty(uri.UserInfo);

    internal static string InstanceId(string root) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(Path.TrimEndingDirectorySeparator(Path.GetFullPath(root)).ToUpperInvariant())))[..20];

    internal static bool IsInternalImage(string address) =>
        address.StartsWith("data:image/", StringComparison.OrdinalIgnoreCase) ||
        (address.StartsWith("blob:", StringComparison.OrdinalIgnoreCase) && IsLocal(address[5..]));

    internal static bool NeedsAuthentication(string address, string method)
    {
        if (!IsLocal(address)) return false;
        var path = new Uri(address).AbsolutePath;
        return method != "GET" || (path != "/" && !path.StartsWith("/web/") && !path.StartsWith("/fonts/"));
    }

    internal static string? NativeConfig(string? id) => id switch
    {
        "original" => "iq2_xs", "coder" => "coder-iq1_m", "quality" => "iq3_s",
        "swift" => "swift-iq2_xs", "uncensored" => "uncensored-iq2_xs", _ => null
    };

    internal static string FindRoot(string? supplied)
    {
        if (supplied is not null)
        {
            var root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(supplied));
            if (IsProject(root)) return root;
            throw new InvalidOperationException("The selected folder is not an installed Strata project.");
        }
        for (DirectoryInfo? directory = new(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
            if (IsProject(directory.FullName)) return directory.FullName;
        throw new InvalidOperationException("Start the app with START-App.bat in your installed Strata folder.");
    }

    private static bool IsProject(string root) =>
        File.Exists(Path.Combine(root, "tools", "start_models.py")) &&
        File.Exists(Path.Combine(root, "serve", "web", "index.html")) &&
        File.Exists(Path.Combine(root, ".venv", "Scripts", "python.exe"));

    internal static Credential ReadCredential(string root, string? suppliedFile, string? nativeId, string? environmentKey)
    {
        var keyFile = suppliedFile is null ? Path.Combine(root, ".secrets", "strata-api-key.txt") :
            Path.GetFullPath(suppliedFile, root);
        if (File.Exists(keyFile)) return new(ValidateKey(File.ReadAllText(keyFile).Trim(), required: true), true);
        if (suppliedFile is not null) throw new InvalidOperationException("The selected API key file does not exist.");
        if (environmentKey is not null) return new(ValidateKey(environmentKey.Trim(), required: true), true);

        string? config = null;
        try
        {
            using var state = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, ".strata-mcp", "server.json")));
            var name = state.RootElement.TryGetProperty("config", out var value) ? value.GetString() : null;
            if (name is not null && Path.GetFileName(name) == name && name.StartsWith("strata-") && name.EndsWith(".json") &&
                !name.EndsWith(".shared-settings.json", StringComparison.OrdinalIgnoreCase) && File.Exists(Path.Combine(root, name)))
                config = Path.Combine(root, name);
        }
        catch (Exception error) when (error is IOException or JsonException or InvalidOperationException) { }
        if (config is null && NativeConfig(nativeId) is { } tag) config = Path.Combine(root, "strata-" + tag + ".json");
        config ??= Directory.EnumerateFiles(root, "strata-*.json")
            .Where(path => !path.EndsWith(".shared-settings.json", StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(File.GetLastWriteTimeUtc).FirstOrDefault();
        if (config is null || !File.Exists(config)) return new("", false);
        try
        {
            using var settings = JsonDocument.Parse(File.ReadAllText(config));
            var value = settings.RootElement;
            var key = value.TryGetProperty("api_key", out var apiKey) ? apiKey.GetString() ?? "" : "";
            return new(ValidateKey(key, required: false), false);
        }
        catch (Exception error) when (error is IOException or JsonException or InvalidOperationException or FormatException)
        { throw new InvalidOperationException("Could not read the installed model's API key setting."); }
    }

    private static string ValidateKey(string value, bool required)
    {
        if ((required && value.Length == 0) || value.Length > 4096 || value.Any(char.IsControl))
            throw new InvalidOperationException("The API key must be nonempty text without control characters.");
        return value;
    }

    internal static void SelfTest()
    {
        string[] good = [Origin, Origin + "/api/history?limit=40", Origin + "/#about"];
        string[] bad = ["https://127.0.0.1:8080", "http://localhost:8080", "http://127.0.0.1:8081",
            "http://127.0.0.1:8080.evil.example/", "http://127.0.0.1.evil.example:8080",
            "http://key@127.0.0.1:8080/", "file:///C:/example/private-key.txt",
            "https://strata.example.com", "data:text/html,test", "javascript:alert(1)", "", "relative"];
        if (good.Any(url => !IsLocal(url)) || bad.Any(IsLocal))
            throw new InvalidOperationException("Local origin policy failed");
        if (NativeConfig("uncensored") != "uncensored-iq2_xs" || NativeConfig("../../secrets") is not null)
            throw new InvalidOperationException("Native configuration policy failed");
        var exampleRoot = Path.Combine(Path.GetTempPath(), "strata-example");
        if (InstanceId(exampleRoot) != InstanceId(Path.Combine(exampleRoot, ".")) ||
            !IsInternalImage("blob:http://127.0.0.1:8080/synthetic") || IsInternalImage("blob:https://example.com/synthetic"))
            throw new InvalidOperationException("Desktop path or attachment policy failed");
        if (NeedsAuthentication(Origin + "/web/app.js", "GET") ||
            !NeedsAuthentication(Origin + "/health", "GET") || !NeedsAuthentication(Origin + "/api/history", "POST"))
            throw new InvalidOperationException("Authentication request policy failed");
        CredentialSelfTest();
    }

    private static void CredentialSelfTest()
    {
        var temporary = Directory.CreateTempSubdirectory("strata-desktop-policy-");
        try
        {
            var root = temporary.FullName;
            if (ReadCredential(root, null, null, null).Key != "") throw new InvalidOperationException("Optional authentication policy failed");
            File.WriteAllText(Path.Combine(root, "strata-test.json"), "{\"api_key\":\"synthetic-config-key\",\"port\":8080}");
            var sharedSettings = Path.Combine(root, "strata-test.shared-settings.json");
            File.WriteAllText(sharedSettings, "{\"temperature\":0.5}");
            File.SetLastWriteTimeUtc(sharedSettings, DateTime.UtcNow.AddSeconds(2));
            if (ReadCredential(root, null, null, null).Key != "synthetic-config-key") throw new InvalidOperationException("Config authentication policy failed");
            if (ReadCredential(root, null, null, "synthetic-env-key").Key != "synthetic-env-key") throw new InvalidOperationException("Environment authentication policy failed");
            var keyFile = Path.Combine(root, "test-key.txt");
            File.WriteAllText(keyFile, "test-key");
            var key = ReadCredential(root, keyFile, null, "synthetic-env-key");
            if (key.Key != "test-key" || !key.OverrideServerKey) throw new InvalidOperationException("File authentication policy failed");
            Directory.CreateDirectory(Path.Combine(root, ".strata-mcp"));
            File.WriteAllText(Path.Combine(root, ".strata-mcp", "server.json"), "{\"config\":\"strata-test.shared-settings.json\"}");
            if (ReadCredential(root, null, null, null).Key != "synthetic-config-key") throw new InvalidOperationException("Shared settings config policy failed");
            File.WriteAllText(Path.Combine(root, ".strata-mcp", "server.json"), "{\"config\":\"../../outside.json\"}");
            if (ReadCredential(root, null, null, null).Key != "synthetic-config-key") throw new InvalidOperationException("Managed config path policy failed");
            foreach (var value in new[] { "", "synthetic\nkey" })
            {
                var refused = false;
                try { ReadCredential(root, null, null, value); } catch (InvalidOperationException) { refused = true; }
                if (!refused) throw new InvalidOperationException("Invalid key policy failed");
            }
            var missingRefused = false;
            try { ReadCredential(root, "missing-key.txt", null, null); } catch (InvalidOperationException) { missingRefused = true; }
            if (!missingRefused) throw new InvalidOperationException("Missing explicit key file policy failed");
        }
        finally { temporary.Delete(recursive: true); }
    }
}
