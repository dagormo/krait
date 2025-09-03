using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Thermo.Chromeleon.Sdk.Interfaces;
using Thermo.Chromeleon.Sdk.Interfaces.Data;
using Thermo.Chromeleon.Sdk.Interfaces.Data.InstrumentMethodScript;
using System.Windows.Forms;

namespace Krait.Chromeleon
{
    public sealed class RtSpec
    {
        public string Analyte;
        public double Rt;
        public double? Window;
    }

    public static class KraitIO
    {
        // ---------- PUBLIC ENTRY POINTS ----------

        /// <summary>
        /// Export current gradient + components to a single JSON file.
        /// </summary>
        public static void ExportCurrentToJson(IInjection inj, IInstrumentMethod method, string jsonPath, string selectedColumn)

        {
            if (inj == null) throw new ArgumentNullException("inj");
            if (method == null) throw new ArgumentNullException("method");
            if (string.IsNullOrWhiteSpace(jsonPath)) throw new ArgumentNullException("jsonPath");

            // Resolve Processing Method from the injection
            IProcessingMethod pm = FindProcessingMethodForInjection(inj);
            if (pm == null) throw new InvalidOperationException("Processing method not found from injection.");

            // Read gradient + components
            var gradient = ReadGradientPoints(method);     // List<Tuple<double t, double c>>
            var components = EnumerateComponents(pm);      // List<RtSpec>

            // Build JSON envelope
            var root = new JObject
            {
                ["schema_version"] = "1.0",
                ["context"] = new JObject
                {
                    ["processing_method_name"] = (inj.ProcessingMethodName != null) ? inj.ProcessingMethodName.Value : null,
                    ["injection_name"] = inj.Name,
                    ["column"] = selectedColumn,                                   // <-- add this
                    ["units"] = new JObject { ["rt"] = "min", ["window"] = "min", ["conc"] = "mM", ["time"] = "min" }
                },
                ["gradient"] = new JArray(gradient.Select(p => new JObject { ["t"] = p.Item1, ["c"] = p.Item2 })),
                ["components"] = new JArray(components.Select(r => new JObject
                {
                    ["name"] = r.Analyte,
                    ["rt"] = r.Rt,
                    ["window"] = r.Window.HasValue ? new JValue(r.Window.Value) : JValue.CreateNull()
                }))
            };


            // Write atomically
            Directory.CreateDirectory(Path.GetDirectoryName(jsonPath));
            var tmp = jsonPath + ".tmp";
            File.WriteAllText(tmp, root.ToString(Formatting.Indented));
            if (File.Exists(jsonPath)) File.Delete(jsonPath);
            File.Move(tmp, jsonPath);
        }

        /// <summary>
        /// Read gradient + components from a JSON file and apply:
        /// - gradient → import into instrument method (saveAsConverted: true)
        /// - components → update in-place by index (rename + RT/window)
        /// </summary>
        public static void ImportFromJson(IInjection inj, IInstrumentMethod method, string jsonPath)
        {
            if (inj == null) throw new ArgumentNullException("inj");
            if (method == null) throw new ArgumentNullException("method");
            if (!File.Exists(jsonPath)) throw new FileNotFoundException("JSON not found", jsonPath);

            var token = JToken.Parse(File.ReadAllText(jsonPath));

            var ctx = token["context"] as Newtonsoft.Json.Linq.JObject;
            string selectedColumn = ctx?["column"]?.ToString();
            // (optional) use it: display/log/propagate
            if (!string.IsNullOrEmpty(selectedColumn))
                System.Windows.Forms.MessageBox.Show("Column: " + selectedColumn, "Krait");

            // gradient: accept either [{t,c}] or [[t,c]]
            var pts = new List<Tuple<double, double>>();
            var gTok = token["gradient"];
            if (gTok != null)
            {
                if (gTok.Type == JTokenType.Array)
                {
                    foreach (var row in (JArray)gTok)
                    {
                        double t, c;
                        if (row.Type == JTokenType.Array && ((JArray)row).Count >= 2)
                        {
                            t = row[0].Value<double>();
                            c = row[1].Value<double>();
                            pts.Add(Tuple.Create(t, c));
                        }
                        else if (row.Type == JTokenType.Object)
                        {
                            var o = (JObject)row;
                            if (TryGetDouble(o, out t, "t", "time", "Time") &&
                                TryGetDouble(o, out c, "c", "conc", "Concentration"))
                            {
                                pts.Add(Tuple.Create(t, c));
                            }
                        }
                    }
                }
            }

            // components: array of {name, rt, window?}
            var specs = new List<RtSpec>();
            var cTok = token["components"] as JArray;
            if (cTok != null)
            {
                foreach (var row in cTok)
                {
                    var o = row as JObject;
                    if (o == null) continue;

                    string name = TryGetString(o, "name", "analyte", "Component", "Name");
                    if (string.IsNullOrWhiteSpace(name) || !TryGetDouble(o, out double rtVal, "rt", "Rt", "retentionTime", "RetentionTime"))
                        continue;

                    double? win = TryGetDouble(o, out double winVal, "window", "Window", "timeWindow", "TimeWindow", "tolerance", "Tolerance") ? (double?)winVal : null;

                    specs.Add(new RtSpec { Analyte = name.Trim(), Rt = rtVal, Window = win });
                }
            }

            // Apply
            if (pts.Count > 0)
            {
                var ptsOrdered = pts.Select((p, i) => new { p, i })
                                    .OrderBy(x => x.p.Item1)
                                    .ThenBy(x => x.i)
                                    .Select(x => x.p)
                                    .ToList();

                if (ptsOrdered.Count > 0 && ptsOrdered[0].Item1 > 1e-9)
                    ptsOrdered.Insert(0, Tuple.Create(0.0, ptsOrdered[0].Item2));

                // import
                EGCReadWrite.ImportIntoMethod(ptsOrdered, method, saveAsConverted: true);
            }

            if (specs.Count > 0)
            {
                ProcMethodUpdater.UpdateFromRtSpecsInPlace(inj, specs);
            }
        }

        // ---------- HELPERS ----------

        private static bool TryGetDouble(JObject o, out double value, params string[] keys)
        {
            for (int i = 0; i < keys.Length; i++)
            {
                var v = o[keys[i]];
                if (v == null || v.Type == JTokenType.Null) continue;
                if (v.Type == JTokenType.Integer || v.Type == JTokenType.Float)
                {
                    value = v.Value<double>();
                    return true;
                }
                if (double.TryParse(v.ToString(), NumberStyles.Any, CultureInfo.InvariantCulture, out double d))
                {
                    value = d; return true;
                }
            }
            value = 0;
            return false;
        }

        private static string TryGetString(JObject o, params string[] keys)
        {
            for (int i = 0; i < keys.Length; i++)
            {
                var v = o[keys[i]];
                if (v != null && v.Type != JTokenType.Null) return v.ToString();
            }
            return null;
        }

        private static IProcessingMethod FindProcessingMethodForInjection(IInjection inj)
        {
            var seq = inj.Parent as ISequence;
            if (seq == null) return null;

            string pmName = (inj.ProcessingMethodName != null) ? inj.ProcessingMethodName.Value : null;
            if (string.IsNullOrWhiteSpace(pmName)) return null;

            foreach (IDataItem child in seq.Children)
            {
                var pm = child as IProcessingMethod;
                if (pm == null) continue;
                string n = (pm.Name != null) ? pm.Name : null;
                if (!string.IsNullOrEmpty(n) && string.Equals(n, pmName, StringComparison.OrdinalIgnoreCase))
                    return pm;
            }
            return null;
        }

        // Read current components → RtSpec list
        private static List<RtSpec> EnumerateComponents(IProcessingMethod pm)
        {
            var list = new List<RtSpec>();
            foreach (IComponent c in pm.Components)
            {
                string name = (c.Name != null) ? c.Name.Value : null;
                double rt = 0.0;
                double? win = null;

                if (c.RetentionTimeSettings != null && c.RetentionTimeSettings.Time != null)
                    rt = c.RetentionTimeSettings.Time.Value ?? 0.0;

                if (c.RetentionTimeSettings != null)
                {
                    object rts = c.RetentionTimeSettings;
                    var t = rts.GetType();
                    var prop =
                        t.GetProperty("Window") ??
                        t.GetProperty("TimeWindow") ??
                        t.GetProperty("Tolerance") ??
                        t.GetProperty("TimeTolerance");
                    if (prop != null)
                    {
                        try
                        {
                            object v = prop.GetValue(rts, null);
                            if (v is double) win = (double)v;
                            else if (v is float) win = Convert.ToDouble((float)v, CultureInfo.InvariantCulture);
                            else if (v != null)
                            {
                                if (double.TryParse(v.ToString(), NumberStyles.Any, CultureInfo.InvariantCulture, out double parsed))
                                    win = parsed;
                            }
                        }
                        catch { }
                    }
                }

                if (!string.IsNullOrWhiteSpace(name))
                    list.Add(new RtSpec { Analyte = name, Rt = rt, Window = win });
            }
            return list;
        }

        // Read gradient from instrument method directly
        // Replace your current ReadGradientPoints with this version
        private static List<Tuple<double, double>> ReadGradientPoints(IInstrumentMethod method)
        {
            var pts = new List<Tuple<double, double>>();
            var script = method.Script;
            int ord = 0;
            var raw = new List<_GradPt>();

            // 1) Collect any concentration assignments found in time steps
            foreach (var stage in script.Stages)
            {
                foreach (var ts in stage.TimeSteps)
                {
                    foreach (var step in ts.Steps)
                    {
                        if (!IsConcentrationStep(step)) continue;

                        // time in minutes (stage/time step time is MethodTime)
                        double minutes = ts.Time.Minutes;
                        if (minutes < 0) continue;

                        // read numeric value from the step
                        if (!TryReadStepDouble(step, out double concVal)) continue;

                        raw.Add(new _GradPt { T = minutes, C = concVal, Ord = ord++ });
                    }
                }
            }

            // 2) If we saw timetable points, normalize them
            if (raw.Count > 0)
            {
                // sort by time, then by insertion order; keep the last assignment at each time
                raw.Sort((a, b) => a.T != b.T ? a.T.CompareTo(b.T) : a.Ord.CompareTo(b.Ord));
                for (int i = 0; i < raw.Count;)
                {
                    int j = i + 1;
                    while (j < raw.Count && Math.Abs(raw[j].T - raw[i].T) < 1e-9) j++;
                    pts.Add(Tuple.Create(raw[j - 1].T, raw[j - 1].C));
                    i = j;
                }
            }

            // 3) Isocratic fallback: if no timetable was found, try to read a single concentration
            if (pts.Count == 0)
            {
                if (TryFindAnyConcentration(script, out double conc0))
                {
                    double endMin = ComputeMethodEndMinutes(script);
                    if (endMin <= 0) endMin = 10.0; // conservative default if method has no duration
                    pts.Add(Tuple.Create(0.0, conc0));
                    pts.Add(Tuple.Create(endMin, conc0));
                }
            }

            // 4) Backfill t=0 if timetable started later
            if (pts.Count > 0 && pts[0].Item1 > 1e-9)
                pts.Insert(0, Tuple.Create(0.0, pts[0].Item2));

            return pts;
        }

        // Heuristic: accept any step where the symbol or first parameter implies "concentration"
        private static bool IsConcentrationStep(IStep step)
        {
            string sym = step.Symbol != null ? step.Symbol.ToString() : "";
            var s = sym.ToLowerInvariant();
            if (s.Contains("concentration")) return true;                    // generic
            if (s.Contains("eluent") && s.Contains("conc")) return true;     // eluent-related
            if (s.Contains("eluentgenerator") && s.Contains("concentration")) return true; // old filter

            // also check parameter name
            foreach (var p in step.Parameters)
            {
                var pname = (p.Parameter != null) ? p.Parameter.Name : null;
                if (!string.IsNullOrEmpty(pname) && pname.ToLowerInvariant().Contains("concentration"))
                    return true;
            }
            return false;
        }

        // Read a numeric value from a step (either via step.Value* or its first parameter's expression)
        // Add: using System.Globalization; using System.Linq;

        private static bool TryReadStepDouble(IStep step, out double value)
        {
            value = 0.0;

            try
            {
                // 1) step.ValueAsDouble (nullable on some builds)
                try
                {
                    // Use reflection so it works whether the type is double or double?
                    var prop = step.GetType().GetProperty("ValueAsDouble");
                    if (prop != null)
                    {
                        object obj = prop.GetValue(step, null);
                        if (obj is double) { value = (double)obj; return true; }
                        if (obj is double?) { var n = (double?)obj; if (n.HasValue) { value = n.Value; return true; } }
                    }
                }
                catch { /* ignore */ }

                // 2) step.ValueAsString
                try
                {
                    var sp = step.GetType().GetProperty("ValueAsString");
                    if (sp != null)
                    {
                        object s = sp.GetValue(step, null);
                        if (s != null)
                        {
                            if (double.TryParse(s.ToString(), NumberStyles.Any, CultureInfo.InvariantCulture, out double d))
                            { value = d; return true; }
                        }
                    }
                }
                catch { /* ignore */ }

                // 3) First parameter's expression (ValueAsDouble? / ValueAsString)
                var p0 = step.Parameters.FirstOrDefault();
                if (p0 != null && p0.Expression != null)
                {
                    try
                    {
                        var vd = p0.Expression.GetType().GetProperty("ValueAsDouble");
                        if (vd != null)
                        {
                            object obj = vd.GetValue(p0.Expression, null);
                            if (obj is double) { value = (double)obj; return true; }
                            if (obj is double?) { var n = (double?)obj; if (n.HasValue) { value = n.Value; return true; } }
                        }
                    }
                    catch { /* ignore */ }

                    try
                    {
                        var vs = p0.Expression.GetType().GetProperty("ValueAsString");
                        if (vs != null)
                        {
                            object s = vs.GetValue(p0.Expression, null);
                            if (s != null)
                            {
                                if (double.TryParse(s.ToString(), NumberStyles.Any, CultureInfo.InvariantCulture, out double d))
                                { value = d; return true; }
                            }
                        }
                    }
                    catch { /* ignore */ }
                }
            }
            catch { /* ignore */ }

            return false;
        }


        // When timetable is empty, scan all time steps for any concentration and build a constant profile
        private static bool TryFindAnyConcentration(IScript script, out double conc)
        {
            conc = 0;
            bool found = false;
            foreach (var stage in script.Stages)
            {
                foreach (var ts in stage.TimeSteps)
                {
                    foreach (var step in ts.Steps)
                    {
                        if (!IsConcentrationStep(step)) continue;
                        double v;
                        if (TryReadStepDouble(step, out v)) { conc = v; found = true; }
                    }
                }
            }
            return found;
        }

        // Use stage times/durations to estimate the method end time
        private static double ComputeMethodEndMinutes(IScript script)
        {
            double endMin = 0.0;
            foreach (var stage in script.Stages)
            {
                double start = stage.Time.Minutes;           // begin time of the stage
                double dur = stage.Duration.Minutes;       // duration of the stage
                double end = start + dur;
                if (end > endMin) endMin = end;
            }
            return endMin;
        }


        private class _GradPt { public double T; public double C; public int Ord; }

        private static bool IsEluentGeneratorConcentration(string symbol)
        {
            if (string.IsNullOrEmpty(symbol)) return false;
            var s = symbol.ToLowerInvariant();
            return s.Contains("eluentgenerator") && s.Contains("concentration");
        }
    }
}
