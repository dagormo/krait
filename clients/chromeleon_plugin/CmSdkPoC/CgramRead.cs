using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Windows.Forms;

using Thermo.Chromeleon.Sdk;                          // CmSdk, CmSdkScope
using Thermo.Chromeleon.Sdk.Common;
using Thermo.Chromeleon.Sdk.Interfaces;               // IDataItem, IItemFactory
using Thermo.Chromeleon.Sdk.Interfaces.Data;          // IInjection
using Thermo.Chromeleon.Sdk.Interfaces.Evaluation;    // IEvaluationContext
using Thermo.Chromeleon.Sdk.Interfaces.UserInterface; // IObjectPickerDialog

public sealed class PeakInfo
{
    public int Index { get; set; }
    public string Name { get; set; }
    public double RetentionTimeMinutes { get; set; }
}

/* ----------------------------- Column Picker UI ----------------------------- */
public sealed class ColumnPickerForm : Form
{
    private ComboBox combo;
    private Button okBtn, cancelBtn;

    public string SelectedColumn { get; private set; }

    public ColumnPickerForm(IEnumerable<string> columns)
    {
        Text = "Select Column";
        StartPosition = FormStartPosition.CenterScreen;
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MaximizeBox = MinimizeBox = false;
        Width = 420; Height = 150;

        var label = new Label { Left = 12, Top = 15, Width = 100, Text = "Column:" };
        combo = new ComboBox
        {
            Left = 12,
            Top = 35,
            Width = 380,
            DropDownStyle = ComboBoxStyle.DropDownList
        };
        foreach (var c in columns) combo.Items.Add(c);
        if (combo.Items.Count > 0) combo.SelectedIndex = 0;

        okBtn = new Button { Text = "OK", Left = 220, Top = 70, Width = 80, DialogResult = DialogResult.OK };
        cancelBtn = new Button { Text = "Cancel", Left = 312, Top = 70, Width = 80, DialogResult = DialogResult.Cancel };

        AcceptButton = okBtn;
        CancelButton = cancelBtn;

        Controls.Add(label);
        Controls.Add(combo);
        Controls.Add(okBtn);
        Controls.Add(cancelBtn);
    }

    protected override void OnShown(EventArgs e)
    {
        base.OnShown(e);
        combo.Focus();
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        base.OnFormClosing(e);
        if (DialogResult == DialogResult.OK)
            SelectedColumn = combo.SelectedItem as string ?? "";
    }
}

/* ------------------------------- Peak Reader ------------------------------- */
public static class PeakReader
{
    public static List<PeakInfo> GetPeaks(IInjection injection, string signalName = null, int maxPeaks = 2000)
        => GetPeaks(injection, signalName, maxPeaks, null);

    public static List<PeakInfo> GetPeaks(IInjection injection, string signalName, int maxPeaks, string debugLogPath)
    {
        if (injection == null) throw new ArgumentNullException(nameof(injection));

        if (string.IsNullOrEmpty(signalName))
            signalName = FirstSignalName(injection);  // use the first signal

        var peaks = new List<PeakInfo>();
        var fac = CmSdk.GetItemFactory();

        TextWriter dbg = null;
        try
        {
            if (!string.IsNullOrEmpty(debugLogPath))
                dbg = File.AppendText(debugLogPath);

            string[] rtTokens = { "peak.retention_time", "peak.retention time", "peak.retentiontime", "peak.time" };
            const string nameFormula = "peak.name";

            for (int i = 0; i < maxPeaks; i++)
            {
                var ctxIndex = fac.CreateEvaluationContext(signalName, i);

                var nameRv = injection.Evaluate(nameFormula, ctxIndex);
                string name = ResolveName(nameRv);

                double rtMin = ResolveRtMinutes(injection, ctxIndex, rtTokens, dbg, i, "idx");

                if (double.IsNaN(rtMin) && !string.IsNullOrEmpty(name))
                {
                    var ctxByName = fac.CreateEvaluationContext(signalName, name);
                    rtMin = ResolveRtMinutes(injection, ctxByName, rtTokens, dbg, i, "name");
                }

                if ((nameRv == null || nameRv.IsError) && double.IsNaN(rtMin))
                    break;

                if (string.IsNullOrEmpty(name)) name = "(unnamed)";
                peaks.Add(new PeakInfo { Index = i, Name = name, RetentionTimeMinutes = rtMin });
            }
        }
        finally { if (dbg != null) dbg.Dispose(); }

        return peaks;
    }

    private static string ResolveName(IReportValue nameRv)
    {
        if (nameRv == null || nameRv.IsError) return null;
        if (nameRv.IsString) return nameRv.StringValue;
        return nameRv.Value != null ? Convert.ToString(nameRv.Value, CultureInfo.InvariantCulture) : null;
    }

    private static double ResolveRtMinutes(
        IInjection inj, IEvaluationContext ctx, string[] rtTokens,
        TextWriter dbg, int index, string mode)
    {
        foreach (var token in rtTokens)
        {
            var rv = inj.Evaluate(token, ctx);

            if (dbg != null)
            {
                string kind = rv == null ? "null" :
                              rv.IsError ? "error" :
                              rv.IsNumeric ? "numeric" :
                              rv.IsTime ? "time" :
                              rv.IsString ? "string" : "other";
                string val = rv == null ? "" :
                             rv.IsNumeric ? rv.NumericValue.ToString("G", CultureInfo.InvariantCulture) :
                             rv.IsTime ? rv.TimeValue.ToString("HH:mm:ss.fff") :
                             rv.IsString ? rv.StringValue :
                             Convert.ToString(rv.Value, CultureInfo.InvariantCulture);
                dbg.WriteLine($"[{index}/{mode}] {token}: {kind} => {val}");
            }

            if (rv == null || rv.IsError) continue;
            if (rv.IsNumeric) return rv.NumericValue;
            if (rv.IsTime) return rv.TimeValue.TimeOfDay.TotalMinutes;
            if (rv.IsString)
            {
                var s = rv.StringValue ?? "";
                s = s.Replace("minutes", "").Replace("minute", "").Replace("min", "").Trim();
                if (s.IndexOf(',') >= 0 && s.IndexOf('.') < 0) s = s.Replace(',', '.');
                if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var d))
                    return d;
            }
            var o = rv.Value;
            if (o is DateTime dt) return dt.TimeOfDay.TotalMinutes;
            if (o is TimeSpan ts) return ts.TotalMinutes;
            if (o is double dd) return dd;
            if (o is float ff) return ff;
        }
        return double.NaN;
    }

    private static string FirstSignalName(IInjection injection)
    {
        foreach (var s in injection.Signals)
        {
            try { if (!string.IsNullOrEmpty(s.Name)) return s.Name; } catch { }
            break;
        }
        return null;
    }
}

/* ------------------------------ CSV utilities ------------------------------ */
public static class CsvUtil
{
    public static void WritePeaksCsv(string path, IEnumerable<PeakInfo> peaks, string columnName)
    {
        using (var sw = new StreamWriter(path, false))
        {
            sw.WriteLine("Peak Name,Retention Time [min],Column");
            foreach (var p in peaks)
                sw.WriteLine($"{Escape(p.Name)},{p.RetentionTimeMinutes.ToString("F3", CultureInfo.InvariantCulture)},{Escape(columnName)}");
        }
    }

    private static string Escape(string s)
    {
        if (string.IsNullOrEmpty(s)) return "";
        if (s.Contains(",") || s.Contains("\""))
            return "\"" + s.Replace("\"", "\"\"") + "\"";
        return s;
    }
}

/* ------------------------------- Picker utils ------------------------------ */
public static class PickerHelpers
{
    public static List<Uri> GetUrisFromPicker(IObjectPickerDialog picker)
    {
        var uris = new List<Uri>();

        var prop = picker.GetType().GetProperty("SelectedUrls") ??
                   picker.GetType().GetProperty("SelectedUris");
        if (prop?.GetValue(picker, null) is IEnumerable e1)
            foreach (var u in e1) if (u is Uri uu) uris.Add(uu);

        if (uris.Count == 0)
        {
            var prop2 = picker.GetType().GetProperty("SelectedObjects") ??
                        picker.GetType().GetProperty("SelectedItems");
            if (prop2?.GetValue(picker, null) is IEnumerable e2)
            {
                foreach (var o in e2)
                {
                    var urlProp = o.GetType().GetProperty("Url") ?? o.GetType().GetProperty("URI");
                    if (urlProp?.GetValue(o, null) is Uri u) uris.Add(u);
                }
            }
        }
        return uris;
    }

    // De-duplicate while preserving original order
    public static List<string> DistinctPreserveOrder(IEnumerable<string> items)
    {
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var list = new List<string>();
        foreach (var x in items)
        {
            if (string.IsNullOrWhiteSpace(x)) continue;
            if (set.Add(x)) list.Add(x.Trim());
        }
        return list;
    }
}

/* --------------------------------- Program --------------------------------- */
class Program
{
    [STAThread]
    static void Main()
    {
        Application.EnableVisualStyles();

        var outDir = @"C:\CM-Exports\peaks";
        Directory.CreateDirectory(outDir);
        var logPath = Path.Combine(outDir, $"peak_export_{DateTime.Now:yyyyMMdd_HHmmss}.log");

        try
        {
            using (var scope = new CmSdkScope())
            {
                CmSdk.Logon.DoLogon();

                // 1) Column picker first (same choice used for all selected injections)
                var columnListRaw = new[]
                {
                    "AS10","AS11","AS11-HC","AS12A","AS14","AS14","AS14A","AS14A","AS15",
                    "AS16","AS17","AS18","AS19","AS19","AS20","AS22","AS22-Fast","AS22-Fast-4um",
                    "AS23","AS24","AS27","AS4A-SC","AS9-HC","CS12A","CS16","CS17","CS18","CS19","PA20"
                };
                var columnList = PickerHelpers.DistinctPreserveOrder(columnListRaw);

                using (var colDlg = new ColumnPickerForm(columnList))
                {
                    if (colDlg.ShowDialog() != DialogResult.OK) return;
                    var selectedColumn = colDlg.SelectedColumn ?? "";

                    // 2) Pick injections
                    var ui = CmSdk.GetUserInterfaceFactory();
                    var fac = CmSdk.GetItemFactory();

                    var picker = ui.CreateObjectPickerDialog();
                    picker.Title = "Select one or more Injections";
                    picker.MultiSelect = true;
                    picker.AddFilter(typeof(IInjection));

                    if (picker.ShowDialog() != DialogResult.OK) return;

                    var uris = PickerHelpers.GetUrisFromPicker(picker);
                    if (uris.Count == 0)
                    {
                        MessageBox.Show("No injections selected.");
                        return;
                    }

                    // 3) Export peaks for each injection, including the chosen column
                    var written = new List<string>();
                    foreach (var uri in uris)
                    {
                        if (!fac.TryGetItem(uri, out IDataItem item)) continue;
                        var inj = item as IInjection;
                        if (inj == null) continue;

                        var peaks = PeakReader.GetPeaks(inj, signalName: null);
                        if (peaks.Count == 0) continue;

                        var safe = Sanitize(inj.Name);
                        var outPath = Path.Combine(outDir, safe + "_peaks.csv");
                        CsvUtil.WritePeaksCsv(outPath, peaks, selectedColumn);
                        written.Add(outPath);
                    }

                    MessageBox.Show(
                        written.Count > 0
                            ? "Export complete:\r\n" + string.Join("\r\n", written)
                            : "No peak results found or resolved for the selected injections.",
                        "Export Peaks");
                }
            }
        }
        catch (Exception ex)
        {
            File.AppendAllText(logPath, ex + Environment.NewLine);
            MessageBox.Show("Error. See log: " + logPath, "Error");
        }
    }

    private static string Sanitize(string name)
    {
        if (string.IsNullOrEmpty(name)) return "Injection";
        foreach (var c in Path.GetInvalidFileNameChars()) name = name.Replace(c, '_');
        return name;
    }
}
