using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Transactions;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

using Thermo.Chromeleon.Sdk;
using Thermo.Chromeleon.Sdk.Common;
using Thermo.Chromeleon.Sdk.Interfaces;
using Thermo.Chromeleon.Sdk.Interfaces.Data;
using Thermo.Chromeleon.Sdk.Interfaces.Data.InstrumentMethodScript;
using Thermo.Chromeleon.Sdk.Interfaces.Instruments.Symbols;
using Thermo.Chromeleon.Sdk.Interfaces.UserInterface;

namespace CmSdkPoC
{
    internal sealed class GradPt
    {
        public double T;   // minutes
        public double C;   // mM
        public int Ord;    // insertion order
    }

    internal static class EGCReadWrite
    {
        private const string TimeHeader = "Time [min]";
        private const string ConcHeader = "EGC Concentration [mM]";

        [STAThread]
        public static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Directory.CreateDirectory(@"C:\CM-Exports\gradients");

            bool? doExport = null;
            string csvPathArg = null;

            // Simple args: --export | --import [--csv <path>]
            for (int i = 0; i < args.Length; i++)
            {
                var a = args[i];
                if (string.Equals(a, "--export", StringComparison.OrdinalIgnoreCase)) doExport = true;
                else if (string.Equals(a, "--import", StringComparison.OrdinalIgnoreCase)) doExport = false;
                else if (string.Equals(a, "--csv", StringComparison.OrdinalIgnoreCase))
                {
                    if (i + 1 >= args.Length) throw new ArgumentException("--csv requires a file path");
                    csvPathArg = args[++i];
                }
            }

            using (var scope = new CmSdkScope())
            {
                CmSdk.Logon.DoLogon();

                var ui = CmSdk.GetUserInterfaceFactory();
                var fac = CmSdk.GetItemFactory();

                if (!doExport.HasValue)
                {
                    var r = MessageBox.Show(
                        "Would you like to EXPORT current EGC gradient(s) to CSV?\n\nYes = Export\nNo = Import from CSV",
                        "Krait EGC Reader/Writer",
                        MessageBoxButtons.YesNoCancel,
                        MessageBoxIcon.Question);

                    if (r == DialogResult.Cancel) return;
                    doExport = (r == DialogResult.Yes);
                }

                // Pick instrument methods
                var picker = ui.CreateObjectPickerDialog();
                picker.Title = "Select Instrument Method(s)";
                picker.AddFilter(typeof(IInstrumentMethod));
                if (picker.ShowDialog() != DialogResult.OK) return;

                var uris = GetUrisFromPicker(picker);
                if (uris.Count == 0)
                {
                    MessageBox.Show("No instrument method selected.", "Krait", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }

                if (doExport.Value)
                {
                    for (int i = 0; i < uris.Count; i++)
                    {
                        IDataItem item;
                        if (!fac.TryGetItem(uris[i], out item)) continue;
                        var method = item as IInstrumentMethod;
                        if (method == null) continue;

                        ExportMethod(method);
                    }
                }
                else
                {
                    string csvPath = csvPathArg;
                    if (string.IsNullOrWhiteSpace(csvPath))
                    {
                        using (var ofd = new OpenFileDialog())
                        {
                            ofd.Title = "Select CSV (Time [min], EGC Concentration [mM])";
                            ofd.Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*";
                            ofd.CheckFileExists = true;
                            if (ofd.ShowDialog() != DialogResult.OK) return;
                            csvPath = ofd.FileName;
                        }
                    }

                    if (!File.Exists(csvPath))
                    {
                        MessageBox.Show("CSV file not found:\n" + csvPath, "Krait", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                    }

                    var csvPoints = NormalizeAndValidate(LoadCsv(csvPath), true);

                    for (int i = 0; i < uris.Count; i++)
                    {
                        IDataItem item;
                        if (!fac.TryGetItem(uris[i], out item)) continue;
                        var method = item as IInstrumentMethod;
                        if (method == null) continue;

                        try
                        {
                            ImportIntoMethod(csvPoints, method);
                            TrySaveMethod(method); // reflection-based fallback save
                            MessageBox.Show("Updated EGC gradient for:\n" + method.Name, "Krait",
                                MessageBoxButtons.OK, MessageBoxIcon.Information);
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show("Failed to update '" + method.Name + "':\n\n" + ex.Message,
                                "Krait", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                    }
                }
            }
        }

        // ------------------ EXPORT ------------------
        private static void ExportMethod(IInstrumentMethod method)
        {
            var script = method.Script;

            var rows = new List<string> { TimeHeader + "," + ConcHeader };
            var raw = new List<GradPt>();
            int ord = 0;

            foreach (var stage in script.Stages)
            {
                foreach (var ts in stage.TimeSteps)
                {
                    foreach (var step in ts.Steps)
                    {
                        var symbolName = step.Symbol != null ? step.Symbol.ToString() : string.Empty;
                        if (!IsEluentGeneratorConcentration(symbolName)) continue;

                        double concVal = ReadStepNumericValue(step);
                        double minutes = ts.Time.Minutes;
                        if (minutes < 0) continue;

                        rows.Add(
                            minutes.ToString("F2", CultureInfo.InvariantCulture) + "," +
                            concVal.ToString("F2", CultureInfo.InvariantCulture));

                        var gp = new GradPt(); gp.T = minutes; gp.C = concVal; gp.Ord = ord++;
                        raw.Add(gp);
                    }
                }
            }

            // Sort + collapse duplicates at same time (last wins)
            raw.Sort(delegate (GradPt a, GradPt b)
            {
                int byT = a.T.CompareTo(b.T);
                return byT != 0 ? byT : a.Ord.CompareTo(b.Ord);
            });

            var points = new List<Tuple<double, double>>();
            for (int i = 0; i < raw.Count;)
            {
                int j = i + 1;
                while (j < raw.Count && Math.Abs(raw[j].T - raw[i].T) < 1e-9) j++;
                points.Add(Tuple.Create(raw[j - 1].T, raw[j - 1].C));
                i = j;
            }

            // Backfill t=0 if missing
            if (points.Count > 0 && points[0].Item1 > 1e-9)
            {
                double firstConc = points[0].Item2;
                points.Insert(0, Tuple.Create(0.0, firstConc));

                bool hasZeroRow = rows.Count > 1 && rows[1].StartsWith("0.00,", StringComparison.Ordinal);
                if (!hasZeroRow)
                    rows.Insert(1, "0.00," + firstConc.ToString("F2", CultureInfo.InvariantCulture));
            }

            var outPath = @"C:\CM-Exports\gradients\" + Sanitize(method.Name) + "_gradient.csv";
            File.WriteAllLines(outPath, rows);
            MessageBox.Show("Wrote CSV:\n" + outPath, "Krait", MessageBoxButtons.OK, MessageBoxIcon.Information);

            if (points.Count > 0) ShowChart(points, method.Name);
        }

        // ------------------ IMPORT ------------------
        private static void ImportIntoMethod(List<Tuple<double, double>> points, IInstrumentMethod method)
        {
            var script = method.Script;

            // Discover existing EGC symbol(s) and stages containing them; gather steps to remove
            var egcSymbols = new List<ISymbol>();
            var egcStages = new List<IStage>();
            var removeSteps = new List<IStep>();

            foreach (var stage in script.Stages)
            {
                foreach (var ts in stage.TimeSteps)
                {
                    foreach (var step in ts.Steps)
                    {
                        var symbolName = step.Symbol != null ? step.Symbol.ToString() : string.Empty;
                        if (!IsEluentGeneratorConcentration(symbolName)) continue;

                        if (step.Symbol != null && !egcSymbols.Contains(step.Symbol)) egcSymbols.Add(step.Symbol);
                        if (!egcStages.Contains(stage)) egcStages.Add(stage);
                        removeSteps.Add(step);
                    }
                }
            }

            if (egcSymbols.Count == 0)
                throw new InvalidOperationException("No existing EGC concentration step found to infer the symbol. Add one manual EGC step first.");

            var egcSymbol = egcSymbols[0];

            // Only write into stages that can host time-programming
            var writableStages = new List<IStage>();
            for (int i = 0; i < egcStages.Count; i++)
                if (egcStages[i].IsTimeTableSection) writableStages.Add(egcStages[i]);
            if (writableStages.Count == 0)
                throw new InvalidOperationException("EGC steps were found only in non time‑table stages; cannot insert time-programmed steps.");

            using (var tx = new TransactionScope(TransactionScopeOption.RequiresNew, TimeSpan.MaxValue))
            {
                // Remove old EGC steps
                for (int i = 0; i < removeSteps.Count; i++)
                    removeSteps[i].Remove();

                // Add new steps to each writable stage
                for (int s = 0; s < writableStages.Count; s++)
                {
                    var stage = writableStages[s];

                    for (int p = 0; p < points.Count; p++)
                    {
                        double t = points[p].Item1;
                        double c = points[p].Item2;

                        var mt = MethodTime.FromMinutes(t);
                        var ts = stage.TimeSteps.GetOrInsertTimeStep(mt);

                        var newStep = script.ElementFactory.CreateStep(egcSymbol);

                        if (!TrySetStepNumericValue(newStep, c))
                        {
                            // Fallback: some SDK builds expose step.ValueAsDouble (nullable) — set via reflection
                            var prop = newStep.GetType().GetProperty("ValueAsDouble");
                            if (prop != null && prop.CanWrite) prop.SetValue(newStep, c, null);
                            else throw new InvalidOperationException("EGC step does not expose a numeric value to assign.");
                        }

                        ts.Steps.Add(newStep);
                    }
                }

                tx.Complete();
            }
        }

        // ---- Step value helpers ----
        private static double ReadStepNumericValue(IStep step)
        {
            // Prefer first parameter's Expression if present
            var p0 = step.Parameters.FirstOrDefault();
            if (p0 != null && p0.Expression != null)
            {
                if (p0.Expression.ValueAsDouble.HasValue)
                    return p0.Expression.ValueAsDouble.Value;

                if (!string.IsNullOrEmpty(p0.Expression.ValueAsString))
                {
                    double parsed;
                    if (double.TryParse(p0.Expression.ValueAsString, NumberStyles.Any, CultureInfo.InvariantCulture, out parsed))
                        return parsed;
                }
            }

            // Fallback: step.ValueAsDouble via reflection (nullable)
            try
            {
                var prop = step.GetType().GetProperty("ValueAsDouble");
                if (prop != null)
                {
                    object obj = prop.GetValue(step, null);
                    if (obj is double?) return ((double?)obj) ?? 0.0;
                    if (obj is double) return (double)obj;
                }
            }
            catch { }

            return 0.0;
        }

        private static bool TrySetStepNumericValue(IStep step, double value)
        {
            var p0 = step.Parameters.FirstOrDefault();
            if (p0 != null && p0.Expression != null)
            {
                p0.Expression.ValueAsDouble = value;
                return true;
            }
            return false;
        }

        // ---- CSV helpers ----
        private static List<Tuple<double, double>> LoadCsv(string path)
        {
            using (var sr = new StreamReader(path))
            {
                string header = sr.ReadLine();
                if (header == null) throw new InvalidDataException("CSV is empty.");

                var cols = SplitCsvLine(header);
                int iTime = IndexOfHeader(cols, TimeHeader);
                int iConc = IndexOfHeader(cols, ConcHeader);
                if (iTime < 0 || iConc < 0)
                    throw new InvalidDataException("CSV must have headers '" + TimeHeader + "' and '" + ConcHeader + "'.");

                var list = new List<Tuple<double, double>>();
                string line;
                int lineNo = 1;
                while ((line = sr.ReadLine()) != null)
                {
                    lineNo++;
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var c = SplitCsvLine(line);
                    if (c.Count <= Math.Max(iTime, iConc))
                        throw new InvalidDataException("Line " + lineNo + ": not enough columns.");

                    double t, v;
                    if (!double.TryParse(c[iTime], NumberStyles.Float, CultureInfo.InvariantCulture, out t))
                        throw new InvalidDataException("Line " + lineNo + ": invalid time '" + c[iTime] + "'.");
                    if (!double.TryParse(c[iConc], NumberStyles.Float, CultureInfo.InvariantCulture, out v))
                        throw new InvalidDataException("Line " + lineNo + ": invalid concentration '" + c[iConc] + "'.");

                    list.Add(Tuple.Create(t, v));
                }

                if (list.Count == 0) throw new InvalidDataException("CSV contains no data rows.");
                return list;
            }
        }

        private static List<Tuple<double, double>> NormalizeAndValidate(List<Tuple<double, double>> raw, bool insertZeroStart)
        {
            var ordered = raw
                .Select(p => Tuple.Create(Math.Max(0.0, p.Item1), p.Item2))
                .OrderBy(p => p.Item1)
                .ToList();

            // Collapse duplicates (last wins)
            var dedup = new List<Tuple<double, double>>();
            for (int i = 0; i < ordered.Count;)
            {
                int j = i + 1;
                while (j < ordered.Count && Math.Abs(ordered[j].Item1 - ordered[i].Item1) < 1e-9) j++;
                dedup.Add(ordered[j - 1]);
                i = j;
            }

            if (insertZeroStart && dedup.Count > 0 && dedup[0].Item1 > 1e-9)
                dedup.Insert(0, Tuple.Create(0.0, dedup[0].Item2));

            for (int k = 0; k < dedup.Count; k++)
                if (dedup[k].Item2 < 0) throw new InvalidDataException("Concentrations must be >= 0 mM.");

            return dedup;
        }

        private static List<string> SplitCsvLine(string line)
        {
            var result = new List<string>();
            bool inQuotes = false;
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < line.Length; i++)
            {
                char ch = line[i];
                if (ch == '"')
                {
                    if (inQuotes && i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i++; }
                    else inQuotes = !inQuotes;
                }
                else if (ch == ',' && !inQuotes)
                { result.Add(sb.ToString().Trim()); sb.Clear(); }
                else sb.Append(ch);
            }
            result.Add(sb.ToString().Trim());
            return result;
        }

        private static int IndexOfHeader(List<string> headers, string expected)
        {
            for (int i = 0; i < headers.Count; i++)
            {
                var h = headers[i];
                if (h != null && string.Equals(h.Trim(), expected, StringComparison.OrdinalIgnoreCase))
                    return i;
            }
            return -1;
        }

        // ---- Misc helpers ----
        private static bool IsEluentGeneratorConcentration(string symbol)
        {
            if (string.IsNullOrEmpty(symbol)) return false;
            var s = symbol.ToLowerInvariant();
            return s.Contains("eluentgenerator") && s.Contains("concentration");
        }

        private static string Sanitize(string name)
        {
            foreach (var c in Path.GetInvalidFileNameChars()) name = name.Replace(c, '_');
            return name;
        }

        private static List<Uri> GetUrisFromPicker(IObjectPickerDialog picker)
        {
            var uris = new List<Uri>();

            var prop = picker.GetType().GetProperty("SelectedUrls");
            if (prop == null) prop = picker.GetType().GetProperty("SelectedUris");
            var e1 = prop != null ? prop.GetValue(picker, null) as System.Collections.IEnumerable : null;
            if (e1 != null)
                foreach (var u in e1) { var uu = u as Uri; if (uu != null) uris.Add(uu); }

            if (uris.Count == 0)
            {
                var prop2 = picker.GetType().GetProperty("SelectedObjects");
                if (prop2 == null) prop2 = picker.GetType().GetProperty("SelectedItems");
                var e2 = prop2 != null ? prop2.GetValue(picker, null) as System.Collections.IEnumerable : null;
                if (e2 != null)
                {
                    foreach (var o in e2)
                    {
                        var urlProp = o.GetType().GetProperty("Url");
                        if (urlProp == null) urlProp = o.GetType().GetProperty("URI");
                        var u = urlProp != null ? urlProp.GetValue(o, null) as Uri : null;
                        if (u != null) uris.Add(u);
                    }
                }
            }
            return uris;
        }

        private static void ShowChart(List<Tuple<double, double>> points, string methodName)
        {
            var form = new Form { Text = "EGC Gradient: " + methodName, Width = 900, Height = 600 };
            var chart = new Chart { Dock = DockStyle.Fill };
            form.Controls.Add(chart);

            var area = new ChartArea();
            area.AxisX.Title = TimeHeader;
            area.AxisY.Title = ConcHeader;
            chart.ChartAreas.Add(area);

            var series = new Series
            {
                ChartType = SeriesChartType.Line,
                BorderWidth = 2,
                XValueType = ChartValueType.Double,
                YValueType = ChartValueType.Double
            };

            for (int i = 0; i < points.Count; i++)
                series.Points.AddXY(points[i].Item1, points[i].Item2);

            chart.Series.Add(series);
            Application.Run(form);
        }

        // Try to call method.Save() if the SDK exposes it (avoids hard dependency on a specific factory API)
        private static void TrySaveMethod(IInstrumentMethod method)
        {
            try
            {
                var mi = method.GetType().GetMethod("Save", Type.EmptyTypes);
                if (mi != null) mi.Invoke(method, null);
            }
            catch { /* ignore; transaction already completed */ }
        }
    }
}
