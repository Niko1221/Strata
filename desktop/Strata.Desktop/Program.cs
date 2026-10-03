using System.Runtime.InteropServices;

namespace Strata.Desktop;

internal static class Program
{
    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    private static extern int SetCurrentProcessExplicitAppUserModelID(string appId);

    [STAThread]
    private static int Main(string[] args)
    {
        Application.SetHighDpiMode(HighDpiMode.PerMonitorV2);
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.SetDefaultFont(new Font("Segoe UI", 10));
        try
        {
            if (args.Contains("--policy-test"))
            {
                AppPolicy.SelfTest();
                return 0;
            }
            string? Value(string flag)
            {
                int index = Array.IndexOf(args, flag);
                if (index < 0) return null;
                if (index + 1 >= args.Length) throw new ArgumentException("A launch option is missing its value.");
                return args[index + 1];
            }
            var root = AppPolicy.FindRoot(Value("--root"));
            SetCurrentProcessExplicitAppUserModelID("Strata.Desktop." + AppPolicy.InstanceId(root));
            var diagnostics = Value("--diagnostics");
            if (diagnostics is not null)
            {
                diagnostics = Path.GetFullPath(diagnostics);
                var local = Path.GetFullPath(Path.Combine(root, ".local")) + Path.DirectorySeparatorChar;
                if (!diagnostics.StartsWith(local, StringComparison.OrdinalIgnoreCase))
                    throw new ArgumentException("Diagnostic output must be inside the project's .local folder.");
            }
            var name = "Local\\StrataDesktop-" + AppPolicy.InstanceId(root);
            if (diagnostics is not null) name += "-Diagnostics-" + Environment.ProcessId;
            using var signal = new EventWaitHandle(false, EventResetMode.AutoReset, name + "-Show");
            using var exitSignal = new EventWaitHandle(false, EventResetMode.AutoReset, name + "-Exit");
            using var mutex = new Mutex(true, name, out var first);
            if (args.Contains("--quit-app")) { if (!first) exitSignal.Set(); return 0; }
            if (!first) { signal.Set(); return 0; }
            using var form = new DesktopForm(root, signal, exitSignal, diagnostics, Value("--api-key-file"));
            Application.Run(form);
            return form.Result;
        }
        catch (Exception error)
        {
            // Never include HTTP headers, request bodies, or credential values in errors.
            MessageBox.Show(error is ArgumentException or InvalidOperationException ? error.Message :
                "Could not start Strata. Check START-App.bat and the installed project folder.",
                "Strata", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
