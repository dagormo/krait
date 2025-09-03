using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Transactions;
using System.Windows.Forms;

using Thermo.Chromeleon.Sdk.Interfaces.Data;
using Thermo.Chromeleon.Sdk.Interfaces.Data.InstrumentMethodScript;
using Thermo.Chromeleon.Sdk.Interfaces.Instruments.Symbols;

using Krait.UI;
using Krait.Utils;

namespace Krait.Chromeleon
{
    public static class EGCReadWrite
    {
        private const string TimeHeader = "Time [min]";
        private const string ConcHeader = "EGC Concentration [mM]";

        private sealed class GradPt
        {
            public double T;
            public double C;
            public int Ord;
        }

        // ---------------- EXPORT ----------------
        public static void ExportMethod(IInstrumentMethod method)
        {
            IScript script = method.Script;
            List<string> rows = new List<string> { TimeHeader + "," + ConcHeader };

            List<GradPt> raw = new List<GradPt>();
            int ord = 0;

            foreach (IStage stage in script.Stages)
            {
                foreach (ITimeStep ts in stage.TimeSteps)
                {
                    foreach (IStep step in ts.Steps)
                    {
                        string symbolName = (step.Symbol != null) ? step.Symbol.ToString() : string.Empty;
                        if (!IsEluentGeneratorConcentration(symbolName)) continue;

                        double concVal = ReadStepNumericValue(step);
                        double minutes = ts.Time.Minutes;
                        if (minutes < 0) continue;

                        rows.Add(string.Format(CultureInfo.InvariantCulture, "{0:F2},{1:F2}", minutes, concVal));
                        raw.Add(new GradPt { T = minutes, C = concVal, Ord = ord++ });
                    }
                }
            }

            // Collapse duplicates (last-at-time wins)
            raw.Sort(delegate (GradPt a, GradPt b)
            {
                if (a.T != b.T) return a.T.CompareTo(b.T);
                return a.Ord.CompareTo(b.Ord);
            });

            List<Tuple<double, double>> points = new List<Tuple<double, double>>();
            for (int i = 0; i < raw.Count;)
            {
                int j = i + 1;
                while (j < raw.Count && Math.Abs(raw[j].T - raw[i].T) < 1e-9) j++;
                points.Add(Tuple.Create(raw[j - 1].T, raw[j - 1].C));
                i = j;
            }

            // Ensure t=0
            if (points.Count > 0 && points[0].Item1 > 1e-9)
            {
                double firstConc = points[0].Item2;
                points.Insert(0, Tuple.Create(0.0, firstConc));
                rows.Insert(1, string.Format(CultureInfo.InvariantCulture, "0.00,{0:F2}", firstConc));
            }

            // Hold to overall stop
            double stopTime = 0.0;
            foreach (IStage s in script.Stages)
                foreach (ITimeStep ts in s.TimeSteps)
                    if (ts.Time.Minutes > stopTime) stopTime = ts.Time.Minutes;

            if (points.Count > 0 && points[points.Count - 1].Item1 < stopTime - 1e-6)
            {
                double lastConc = points[points.Count - 1].Item2;
                points.Add(Tuple.Create(stopTime, lastConc));
                rows.Add(string.Format(CultureInfo.InvariantCulture, "{0:F2},{1:F2}", stopTime, lastConc));
            }

            string outPath = Path.Combine(@"C:\CM-Exports\gradients", CommonUtil.Sanitize(method.Name) + "_gradient.csv");
            Directory.CreateDirectory(Path.GetDirectoryName(outPath));
            File.WriteAllLines(outPath, rows);
            MessageBox.Show("Wrote CSV:\n" + outPath, "Krait", MessageBoxButtons.OK, MessageBoxIcon.Information);

            if (points.Count > 0) ChartHelpers.ShowGradientChart(points, method.Name);
        }

        // ---------------- IMPORT ----------------

        // ---- helper ----
        private static List<Tuple<double, double>> QuantizeAndOrderSameTime(List<Tuple<double, double>> pts)
        {
            // Chromeleon timetable shows/rounds at 0.001 min; use that as spacing
            const double RES = 0.001;       // 0.001 min = 0.06 s
            const double TOL = 1e-9;

            // Sort by time, but keep original order within equal-time groups
            var indexed = pts.Select((p, i) => new { p, i }).ToList();
            indexed.Sort((a, b) => {
                int byT = a.p.Item1.CompareTo(b.p.Item1);
                return (byT != 0) ? byT : a.i.CompareTo(b.i);
            });

            var outPts = new List<Tuple<double, double>>();
            double prevT = -1.0;

            for (int i = 0; i < indexed.Count;)
            {
                double t0 = indexed[i].p.Item1;
                int j = i + 1;
                while (j < indexed.Count && Math.Abs(indexed[j].p.Item1 - t0) < TOL) j++;

                int k = j - i; // number of points at time t0
                if (k == 1)
                {
                    outPts.Add(indexed[i].p);
                    prevT = indexed[i].p.Item1;
                }
                else
                {
                    double firstPre = Math.Max(0.0, Math.Max(prevT, t0 - RES * (k - 1)));
                    for (int g = 0; g < k - 1; g++)
                    {
                        double tg = firstPre + RES * g;
                        if (tg > t0 - RES) tg = t0 - RES;
                        outPts.Add(Tuple.Create(tg, indexed[i + g].p.Item2));   // old values
                    }
                    outPts.Add(Tuple.Create(t0, indexed[i + (k - 1)].p.Item2)); // new value at t0
                    prevT = t0;
                }

                i = j;
            }

            return outPts;
        }

        public static void ImportIntoMethod(List<Tuple<double, double>> points, IInstrumentMethod method, bool saveAsConverted)
        {
            if (points == null) throw new ArgumentNullException("points");

            points = QuantizeAndOrderSameTime(points);

            IScript script = method.Script;

            List<ISymbol> egcSymbols = new List<ISymbol>();
            List<IStage> egcStages = new List<IStage>();
            List<IStep> removeSteps = new List<IStep>();

            foreach (IStage stage in script.Stages)
            {
                foreach (ITimeStep ts in stage.TimeSteps)
                {
                    foreach (IStep step in ts.Steps)
                    {
                        string symbolName = (step.Symbol != null) ? step.Symbol.ToString() : string.Empty;
                        if (!IsEluentGeneratorConcentration(symbolName)) continue;

                        if (step.Symbol != null && !egcSymbols.Contains(step.Symbol)) egcSymbols.Add(step.Symbol);
                        if (!egcStages.Contains(stage)) egcStages.Add(stage);
                        removeSteps.Add(step);
                    }
                }
            }
            if (egcSymbols.Count == 0)
                throw new InvalidOperationException("No existing EGC concentration step found.");

            ISymbol egcSymbol = egcSymbols[0];

            // Choose a host stage (prefer one that already had EGC; else any time-table stage, prefer 'Run')
            IStage host = ChooseEgcHostStage(script, egcStages);
            if (host == null || !host.IsTimeTableSection) host = ChooseFallbackTimeTableStage(script);
            if (host == null || !host.IsTimeTableSection)
                throw new InvalidOperationException("No suitable EGC host stage with a time table section.");

            // Old max time across the method
            double oldMaxTime = 0.0;
            foreach (IStage s in script.Stages)
                foreach (ITimeStep ts in s.TimeSteps)
                    if (ts.Time.Minutes > oldMaxTime) oldMaxTime = ts.Time.Minutes;

            // New max time from incoming points
            double newMaxTime = (points.Count > 0) ? points[points.Count - 1].Item1 : 0.0;

            using (TransactionScope tx = new TransactionScope(TransactionScopeOption.RequiresNew, TimeSpan.MaxValue))
            {
                // 1) Remove prior EGC steps everywhere
                foreach (IStep rs in removeSteps) rs.Remove();

                // 2) Ensure the host stage actually reaches newMaxTime
                GetStageWindow(script, host, out double stageStart, out double stageEnd);

                if (newMaxTime > stageEnd - 1e-9)
                {
                    double desiredDuration = Math.Max(0.0, newMaxTime - stageStart + 0.001);
                    bool extended = TrySetRunDuration(script, host, desiredDuration);

                    if (!extended)
                        extended = TryRetimeStopRun(script, host, newMaxTime); // move/create Stop Run, shift Post Run

                    GetStageWindow(script, host, out stageStart, out stageEnd);

                    if (!extended || newMaxTime > stageEnd - 1e-9)
                        newMaxTime = Math.Min(newMaxTime, stageEnd - 1e-6);
                }

                // 3) Stretch host last timestep if needed (never move the first)
                ITimeStep lastTs = GetLastTimeStep(host, out double hostLastTime);
                if (newMaxTime > hostLastTime + 1e-6)
                {
                    if (IsFirstTimeStep(host, lastTs))
                    {
                        ITimeStep terminal = host.TimeSteps.GetOrInsertTimeStep(MethodTime.FromMinutes(newMaxTime));
                        CloneStepSet(script, lastTs, terminal);
                    }
                    else
                    {
                        lastTs.Time = MethodTime.FromMinutes(newMaxTime);
                    }
                }

                // 4) Insert EGC points into the host (guard by window)
                for (int k = 0; k < points.Count; k++)
                {
                    double t = points[k].Item1;
                    double c = points[k].Item2;

                    GetStageWindow(script, host, out double sStart, out double sEnd);
                    if (t > sEnd + 1e-9) continue;

                    MethodTime mt = MethodTime.FromMinutes(t);
                    ITimeStep ts = host.TimeSteps.GetOrInsertTimeStep(mt);

                    IStep newStep = script.ElementFactory.CreateStep(egcSymbol);
                    if (!TrySetStepNumericValue(newStep, c))
                    {
                        var prop = newStep.GetType().GetProperty("ValueAsDouble");
                        if (prop != null && prop.CanWrite) prop.SetValue(newStep, c, null);
                        else throw new InvalidOperationException("EGC step has no numeric value property.");
                    }
                    ts.Steps.Add(newStep);
                }

                // 5) Shift any steps exactly at oldMaxTime to newMaxTime (never break ordering)
                if (newMaxTime > oldMaxTime + 1e-6)
                {
                    MethodTime mtNew = MethodTime.FromMinutes(newMaxTime);

                    foreach (IStage s in script.Stages)
                    {
                        foreach (ITimeStep ts in s.TimeSteps)
                        {
                            double tm = ts.Time.Minutes;
                            if (Math.Abs(tm - oldMaxTime) >= 1e-6) continue;

                            if (IsLastTimeStep(s, ts) && !IsFirstTimeStep(s, ts))
                            {
                                // safe to retime the last TS
                                ts.Time = mtNew;
                            }
                            else
                            {
                                // clone the whole step set into a new TS at the end
                                ITimeStep tsNew = s.TimeSteps.GetOrInsertTimeStep(mtNew);
                                CloneStepSet(script, ts, tsNew);
                                // keep the earlier TS in place so we don't violate monotonicity
                            }
                        }
                    }
                }


                // 6) Save as NEW (rename + Save)
                if (saveAsConverted)
                {
                    try
                    {
                        string baseName = method.Name ?? "Method";
                        string newName = CommonUtil.Sanitize(baseName) + "_converted_" + DateTime.Now.ToString("yyyyMMdd_HHmmss");

                        var nameProp = method.GetType().GetProperty("Name");
                        if (nameProp != null && nameProp.CanWrite)
                            nameProp.SetValue(method, newName, null);
                        else
                            throw new InvalidOperationException("Method name is not writable via SDK.");

                        TrySaveMethod(method);

                        MessageBox.Show("Saved new method as: " + newName,
                            "Krait", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("Could not save new method: " + ex.Message,
                            "Krait", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
                else
                {
                    TrySaveMethod(method);
                }

                tx.Complete();
            }
        }

        // ---------------- CSV LOAD + NORMALIZE ----------------
        public static List<Tuple<double, double>> LoadCsv(string filePath)
        {
            List<Tuple<double, double>> result = new List<Tuple<double, double>>();
            try
            {
                string[] lines = File.ReadAllLines(filePath);
                for (int i = 1; i < lines.Length; i++) // skip header
                {
                    string line = lines[i].Trim();
                    if (line.Length == 0) continue;

                    string[] parts = line.Split(',');
                    if (parts.Length < 2) continue;

                    if (double.TryParse(parts[0], NumberStyles.Any, CultureInfo.InvariantCulture, out double t) &&
                        double.TryParse(parts[1], NumberStyles.Any, CultureInfo.InvariantCulture, out double c))
                    {
                        if (t < 0) t = 0; // clamp
                        result.Add(Tuple.Create(t, c));
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error reading gradient CSV: " + ex.Message);
            }
            return result;
        }

        public static List<Tuple<double, double>> NormalizeAndValidate(List<Tuple<double, double>> raw, bool insertZero)
        {
            if (raw == null) return new List<Tuple<double, double>>();

            // Order by time, clamp to >= 0
            List<Tuple<double, double>> ordered = raw
                .Select(p => Tuple.Create(p.Item1 < 0 ? 0.0 : p.Item1, p.Item2))
                .OrderBy(p => p.Item1)
                .ToList();

            // Deduplicate: keep last at same time
            List<Tuple<double, double>> dedup = new List<Tuple<double, double>>();
            for (int i = 0; i < ordered.Count;)
            {
                int j = i + 1;
                while (j < ordered.Count && Math.Abs(ordered[j].Item1 - ordered[i].Item1) < 1e-9) j++;
                dedup.Add(ordered[j - 1]);
                i = j;
            }

            // Optional t=0 insert
            if (insertZero && dedup.Count > 0 && dedup[0].Item1 > 1e-9)
                dedup.Insert(0, Tuple.Create(0.0, dedup[0].Item2));

            // Sanity
            for (int k = 0; k < dedup.Count; k++)
                if (dedup[k].Item2 < 0) throw new InvalidDataException("Concentrations must be >= 0 mM.");

            return dedup;
        }

        // ---------------- HELPERS ----------------
        private static bool IsEluentGeneratorConcentration(string symbol)
        {
            if (string.IsNullOrEmpty(symbol)) return false;
            string s = symbol.ToLowerInvariant();
            return s.Contains("eluentgenerator") && s.Contains("concentration");
        }

        private static double ReadStepNumericValue(IStep step)
        {
            double v;
            return TryReadStepNumericValue(step, out v) ? v : 0.0;
        }

        private static bool TryReadStepNumericValue(IStep step, out double value)
        {
            value = 0.0;
            try
            {
                // 1) Step.ValueAsDouble (could be double or double?)
                var prop = step.GetType().GetProperty("ValueAsDouble");
                if (prop != null)
                {
                    object obj = prop.GetValue(step, null);
                    if (obj is double) { value = (double)obj; return true; }
                    if (obj is double?) { var n = (double?)obj; if (n.HasValue) { value = n.Value; return true; } }
                }

                // 2) Step.ValueAsString
                var sp = step.GetType().GetProperty("ValueAsString");
                if (sp != null)
                {
                    object s = sp.GetValue(step, null);
                    if (s != null)
                    {
                        double d;
                        if (double.TryParse(s.ToString(), NumberStyles.Any, CultureInfo.InvariantCulture, out d))
                        { value = d; return true; }
                    }
                }

                // 3) First parameter’s expression (ValueAsDouble?/ValueAsString)
                var p0 = step.Parameters.FirstOrDefault();
                if (p0 != null && p0.Expression != null)
                {
                    var vd = p0.Expression.GetType().GetProperty("ValueAsDouble");
                    if (vd != null)
                    {
                        object obj = vd.GetValue(p0.Expression, null);
                        if (obj is double) { value = (double)obj; return true; }
                        if (obj is double?) { var n = (double?)obj; if (n.HasValue) { value = n.Value; return true; } }
                    }

                    var vs = p0.Expression.GetType().GetProperty("ValueAsString");
                    if (vs != null)
                    {
                        object s = vs.GetValue(p0.Expression, null);
                        if (s != null)
                        {
                            double d;
                            if (double.TryParse(s.ToString(), NumberStyles.Any, CultureInfo.InvariantCulture, out d))
                            { value = d; return true; }
                        }
                    }
                }
            }
            catch { }
            return false;
        }


        private static bool TrySetStepNumericValue(IStep step, double value)
        {
            var p0 = step.Parameters.FirstOrDefault();
            if (p0 != null && p0.Expression != null)
            {
                p0.Expression.ValueAsDouble = value;
                return true;
            }

            try
            {
                var prop = step.GetType().GetProperty("ValueAsDouble");
                if (prop != null && prop.CanWrite)
                {
                    prop.SetValue(step, value, null);
                    return true;
                }
            }
            catch { }

            return false;
        }

        private static void TrySaveMethod(IInstrumentMethod method)
        {
            try
            {
                var mi = method.GetType().GetMethod("Save", Type.EmptyTypes);
                if (mi != null) mi.Invoke(method, null);
            }
            catch { }
        }

        private static ITimeStep GetLastTimeStep(IStage stage, out double lastTime)
        {
            ITimeStep last = null;
            double max = double.MinValue;

            foreach (ITimeStep ts in stage.TimeSteps)
            {
                double t = ts.Time.Minutes;
                if (last == null || t >= max)
                {
                    last = ts;
                    max = t;
                }
            }

            if (last == null)
            {
                foreach (ITimeStep ts in stage.TimeSteps) { last = ts; break; }
                if (last == null) throw new InvalidOperationException("Stage has no time steps.");
                max = last.Time.Minutes;
            }

            lastTime = max;
            return last;
        }

        private static IStage ChooseEgcHostStage(IScript script, List<IStage> egcStages)
        {
            IStage best = null;
            double bestLast = double.NegativeInfinity;

            foreach (IStage s in egcStages)
            {
                GetLastTimeStep(s, out double last);
                if (last > bestLast) { bestLast = last; best = s; }
            }
            return best;
        }

        private static IStage ChooseFallbackTimeTableStage(IScript script)
        {
            IStage fallback = null;
            foreach (IStage s in script.Stages)
            {
                if (!s.IsTimeTableSection) continue;

                string name = "";
                try
                {
                    var p = s.GetType().GetProperty("Name");
                    if (p != null)
                    {
                        object obj = p.GetValue(s, null);
                        if (obj is string) name = (string)obj;
                    }
                }
                catch { }
                if (string.IsNullOrEmpty(name)) name = s.ToString();

                if (!string.IsNullOrEmpty(name) &&
                    name.IndexOf("Run", StringComparison.OrdinalIgnoreCase) >= 0)
                    return s;

                if (fallback == null) fallback = s;
            }
            return fallback;
        }

        // Stage window [start, end); end = next stage start or +∞ if last
        private static void GetStageWindow(IScript script, IStage stage, out double start, out double end)
        {
            start = stage.Time.Minutes;
            end = double.PositiveInfinity;

            bool hit = false;
            foreach (IStage s in script.Stages)
            {
                if (hit) { end = s.Time.Minutes; break; }
                if (object.ReferenceEquals(s, stage)) hit = true;
            }
        }

        // Try to set a duration-like parameter on a Run step
        private static bool TrySetRunDuration(IScript script, IStage host, double newDurationMinutes)
        {
            foreach (ITimeStep ts in host.TimeSteps)
            {
                foreach (IStep st in ts.Steps)
                {
                    string sym = (st.Symbol != null) ? st.Symbol.ToString() : "";
                    if (sym.IndexOf("Run", StringComparison.OrdinalIgnoreCase) < 0)
                        continue;

                    // Any numeric expression param
                    foreach (var p in st.Parameters)
                    {
                        try
                        {
                            if (p != null && p.Expression != null)
                            {
                                p.Expression.ValueAsDouble = newDurationMinutes;
                                return true;
                            }
                        }
                        catch { }
                    }

                    // Reflection: "Duration" as double/MethodTime
                    try
                    {
                        var propDur = st.GetType().GetProperty("Duration");
                        if (propDur != null && propDur.CanWrite)
                        {
                            if (propDur.PropertyType == typeof(double))
                            {
                                propDur.SetValue(st, newDurationMinutes, null);
                                return true;
                            }
                            else if (propDur.PropertyType == typeof(MethodTime))
                            {
                                propDur.SetValue(st, MethodTime.FromMinutes(newDurationMinutes), null);
                                return true;
                            }
                        }
                    }
                    catch { }

                    // Generic writable double/MethodTime
                    try
                    {
                        var props = st.GetType().GetProperties();
                        foreach (var pr in props)
                        {
                            if (!pr.CanWrite) continue;

                            if (pr.PropertyType == typeof(double))
                            {
                                pr.SetValue(st, newDurationMinutes, null);
                                return true;
                            }
                            if (pr.PropertyType == typeof(double?))
                            {
                                pr.SetValue(st, (double?)newDurationMinutes, null);
                                return true;
                            }
                            if (pr.PropertyType == typeof(MethodTime))
                            {
                                pr.SetValue(st, MethodTime.FromMinutes(newDurationMinutes), null);
                                return true;
                            }
                        }
                    }
                    catch { }
                }
            }
            return false;
        }

        // ---- robust stop-run retime/create helpers ----

        private static bool TryRetimeStopRun(IScript script, IStage host, double newEndMinutes)
        {
            // 1) Try to move a stop-like step in host
            if (MoveFirstStopLikeStepInStage(script, host, newEndMinutes))
                return true;

            // 2) Search all timetable stages
            foreach (IStage s in script.Stages)
            {
                if (!s.IsTimeTableSection) continue;
                if (MoveFirstStopLikeStepInStage(script, s, newEndMinutes))
                    return true;
            }

            // 3) Last resort: create a stop step at the end in host
            ITimeStep tsEnd = host.TimeSteps.GetOrInsertTimeStep(MethodTime.FromMinutes(newEndMinutes));
            IStep stopStep = CreateStopLikeStep(script);
            if (stopStep != null)
            {
                tsEnd.Steps.Add(stopStep);
                return true;
            }

            // Even an empty final TS helps some templates extend
            return true;
        }

        private static bool IsLastTimeStep(IStage stage, ITimeStep ts)
        {
            ITimeStep last = null;
            foreach (var t in stage.TimeSteps) last = t;
            return object.ReferenceEquals(last, ts);
        }

        private static bool MoveFirstStopLikeStepInStage(IScript script, IStage stage, double newEndMinutes)
        {
            foreach (ITimeStep ts in stage.TimeSteps)
            {
                foreach (IStep st in ts.Steps)
                {
                    if (!IsStopLike(st)) continue;

                    bool isFirst = IsFirstTimeStep(stage, ts);
                    bool isLast = IsLastTimeStep(stage, ts);

                    if (isLast && !isFirst)
                    {
                        // last TS (and not the first) → safe to retime
                        ts.Time = MethodTime.FromMinutes(newEndMinutes);
                    }
                    else
                    {
                        // first or middle TS → clone to a new TS at the end
                        ITimeStep tsNew = stage.TimeSteps.GetOrInsertTimeStep(MethodTime.FromMinutes(newEndMinutes));
                        IStep clone = script.ElementFactory.CreateStep(st.Symbol);
                        CopyNumericParams(st, clone);
                        tsNew.Steps.Add(clone);

                        // optional: remove the early stop to avoid duplicates
                        try { st.Remove(); } catch { /* ignore if not supported */ }
                    }

                    AlignPostRun(script, newEndMinutes);
                    return true;
                }
            }
            return false;
        }



        private static void AlignPostRun(IScript script, double newEndMinutes)
        {
            foreach (IStage s in script.Stages)
            {
                string name = "";
                try
                {
                    var p = s.GetType().GetProperty("Name");
                    if (p != null)
                    {
                        object obj = p.GetValue(s, null);
                        if (obj is string) name = (string)obj;
                    }
                }
                catch { }

                if ((name ?? s.ToString()).IndexOf("Post Run", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    ITimeStep first = null;
                    foreach (var ts in s.TimeSteps) { first = ts; break; }
                    if (first != null) first.Time = MethodTime.FromMinutes(newEndMinutes);
                    break;
                }
            }
        }

        private static bool IsStopLike(IStep step)
        {
            string sym = (step.Symbol != null) ? step.Symbol.ToString() : "";
            if (sym.Length == 0) return false;

            string s = sym.ToLowerInvariant();
            return s.Contains("stoprun") ||
                   s.Contains("stop run") ||
                   s.EndsWith(".stop") ||
                   s.Contains("stopacq") ||
                   s.Contains("stop acquisition") ||
                   (s.Contains("stop") && s.Contains("acquisition")) ||
                   (s.Contains("stop") && s.Contains("run"));
        }

        private static IStep CreateStopLikeStep(IScript script)
        {
            string[] candidates =
            {
                "System.StopRun",
                "System.Stop",
                "Acquisition.Stop",
                "Acquisition.StopRun",
                "Instrument.Stop",
                "Chromatography.Stop"
            };

            foreach (string path in candidates)
            {
                try
                {
                    ISymbol sym = FindSymbolByPath(script, path);
                    if (sym != null) return script.ElementFactory.CreateStep(sym);
                }
                catch { }
            }
            return null;
        }

        private static ISymbol FindSymbolByPath(IScript script, string dottedPath)
        {
            if (script == null || string.IsNullOrEmpty(dottedPath)) return null;

            ISymbol node = script.RootSymbol;
            string[] parts = dottedPath.Split(new[] { '.' }, StringSplitOptions.RemoveEmptyEntries);

            for (int i = 0; i < parts.Length && node != null; i++)
            {
                string part = parts[i];
                ISymbol next = null;

                foreach (ISymbol child in node.Children)
                {
                    string name = child.ToString();
                    if (string.Equals(name, part, StringComparison.OrdinalIgnoreCase) ||
                        name.EndsWith("." + part, StringComparison.OrdinalIgnoreCase) ||
                        name.IndexOf(part, StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        next = child;
                        break;
                    }
                }

                if (next == null) return null;
                node = next;
            }
            return node;
        }

        private static bool IsFirstTimeStep(IStage stage, ITimeStep ts)
        {
            foreach (ITimeStep t in stage.TimeSteps)
                return Object.ReferenceEquals(t, ts);
            return false;
        }

        private static void CloneStepSet(IScript script, ITimeStep src, ITimeStep dst)
        {
            foreach (IStep s in src.Steps)
            {
                IStep clone = script.ElementFactory.CreateStep(s.Symbol);
                double val = ReadStepNumericValue(s);
                TrySetStepNumericValue(clone, val);
                dst.Steps.Add(clone);
            }
        }

        private static void CopyNumericParams(IStep src, IStep dst)
        {
            try
            {
                var pSrc = src.Parameters.FirstOrDefault();
                var pDst = dst.Parameters.FirstOrDefault();
                if (pSrc != null && pSrc.Expression != null && pSrc.Expression.ValueAsDouble.HasValue &&
                    pDst != null && pDst.Expression != null)
                {
                    pDst.Expression.ValueAsDouble = pSrc.Expression.ValueAsDouble.Value;
                }
            }
            catch { }
        }
    }
}
