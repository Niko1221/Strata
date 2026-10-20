using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Windows.Forms;

// Offscreen native rendering only. No ManagedTray, NotifyIcon, application loop,
// config files, processes, network, clipboard or menu-click handlers are created.
internal static class MenuExamples
{
    internal static Dictionary<string, object> State()
    {
        return new Dictionary<string, object> {
            { "api", "available" }, { "model", "unloaded" }, { "phase", "ready" }, { "active", 0 },
            { "web", "reachable" }, { "error", "" }, { "busy", false },
            { "version", "0.1.45.3" }, { "update", "Use Official Component Updater" },
            { "selected", Selection(131072, 1800) }, { "effective", Selection(131072, 1800) }
        };
    }

    private static Dictionary<string, object> Selection(int context, int idle)
    {
        return new Dictionary<string, object> { { "profile", "Example Profile" }, { "context", context }, { "idle", idle }, { "auto", false } };
    }

    private static Bitmap Render(ToolStrip strip)
    {
        strip.AutoSize = false;
        strip.Size = strip.GetPreferredSize(Size.Empty);
        strip.PerformLayout();
        // Force a native control handle without ever calling Show or running a UI loop.
        IntPtr handle = strip.Handle;
        if (strip.Width < 40 || strip.Height < 20) throw new InvalidOperationException("Menu Layout Is Empty");
        var bitmap = new Bitmap(strip.Width, strip.Height, PixelFormat.Format32bppArgb);
        strip.DrawToBitmap(bitmap, new Rectangle(Point.Empty, bitmap.Size));
        int contrast = 0;
        Color corner = bitmap.GetPixel(0, 0);
        for (int y = 2; y < bitmap.Height - 2; y++)
            for (int x = 2; x < bitmap.Width - 2; x++)
                if (bitmap.GetPixel(x, y).ToArgb() != corner.ToArgb()) contrast++;
        if (contrast < 300) { bitmap.Dispose(); throw new InvalidOperationException("Menu Bitmap Is Blank"); }
        return bitmap;
    }

    private static void Save(string directory, string name, string title, ManagedMenu view, ToolStripMenuItem expanded)
    {
        using (var main = Render(view.Menu))
        using (var child = expanded == null ? null : Render(expanded.DropDown))
        using (var output = new Bitmap(main.Width + (child == null ? 0 : child.Width + 16) + 32, Math.Max(main.Height, child == null ? 0 : child.Height) + 110))
        using (var graphics = Graphics.FromImage(output))
        using (var heading = new Font(SystemFonts.MenuFont.FontFamily, 11, FontStyle.Bold))
        using (var stream = typeof(MenuExamples).Assembly.GetManifestResourceStream("Strata.ico"))
        using (var icon = new Icon(stream, new Size(32, 32)))
        {
            graphics.Clear(SystemColors.Window);
            graphics.DrawIcon(icon, new Rectangle(16, 16, 32, 32));
            graphics.DrawString("Strata Tray — Simulated Menu State", heading, SystemBrushes.WindowText, 60, 13);
            graphics.DrawString("Fixture Data Only · No Server Or Inference", SystemFonts.MenuFont, SystemBrushes.WindowText, 60, 37);
            graphics.DrawString(title, SystemFonts.MenuFont, SystemBrushes.WindowText, 16, 65);
            graphics.DrawImageUnscaled(main, 16, 90);
            if (child != null) graphics.DrawImageUnscaled(child, main.Width + 32, 90);
            output.Save(Path.Combine(directory, name), ImageFormat.Png);
            Console.WriteLine(name + " " + output.Width + "x" + output.Height);
        }
    }

    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Length != 2 || args[0] != "--out" || !Path.IsPathRooted(args[1])) throw new ArgumentException("Use --out With An Absolute Output Directory");
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Directory.CreateDirectory(args[1]);
        using (var view = new ManagedMenu(new[] { "Example Profile" }, true))
        {
            var state = State(); view.ApplyError(state); view.Apply(state, false);
            view.AppUpdate.Text = AppUpdatePolicy.Describe(new AppUpdateState { Current = "0.2.0-beta", Latest = "0.2.0-beta", Phase = "current" });
            Save(args[1], "managed-menu.png", "Managed Menu · Example Profile · Unloaded", view, null);
            Save(args[1], "context-menu.png", "Context Submenu · Selected And Effective: 128K", view, view.Contexts);
            Save(args[1], "idle-menu.png", "Idle Submenu · Selected And Effective: 30 Minutes", view, view.Idles);
            Save(args[1], "engine-update-menu.png", "Engine Submenu · Official Component Updater Required", view, view.Update);
            view.AppUpdate.Text = AppUpdatePolicy.Describe(new AppUpdateState { Current = "0.1.0", Latest = "0.2.0-beta", Phase = "available" });
            Save(args[1], "app-update-menu.png", "App Submenu · Simulated Available Update", view, view.AppUpdate);
        }
        return 0;
    }
}
