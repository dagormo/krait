using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Transactions;
using System.Windows.Forms;
using Thermo.Chromeleon.Sdk.Interfaces;
using Thermo.Chromeleon.Sdk.Interfaces.Data;

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;


public sealed class RtSpec
{
    public string Analyte;
    public double Rt;         // minutes
    public double? Window;    // optional minutes
}

public static class ProcMethodUpdater
{
    public static void UpdateFromCsvInPlace(IInjection injection, string csvPath)
    {
        if (injection == null) throw new ArgumentNullException("injection");
        if (string.IsNullOrWhiteSpace(csvPath) || !File.Exists(csvPath))
            throw new FileNotFoundException("CSV not found", csvPath);

        // Parse CSV to ordered list
        List<RtSpec> rows = LoadRtSpecsFromFile(csvPath);
        if (rows.Count == 0)
            throw new InvalidOperationException("No valid rows found. Expected: Analyte,ExpectedRT[,Window].");

        // Resolve parent sequence and processing method
        ISequence seq = injection.Parent as ISequence;
        if (seq == null) throw new InvalidOperationException("Injection has no parent sequence.");

        string pmName = (injection.ProcessingMethodName != null) ? injection.ProcessingMethodName.Value : null;
        if (string.IsNullOrWhiteSpace(pmName))
            throw new InvalidOperationException("Injection has no ProcessingMethodName assigned.");

        IProcessingMethod pm = FindProcessingMethodByName(seq, pmName);
        if (pm == null) throw new InvalidOperationException("Processing method '" + pmName + "' not found under the sequence.");

        // Work inside a transaction
        using (var tx = new TransactionScope(TransactionScopeOption.RequiresNew, TimeSpan.MaxValue))
        {
            // Snapshot components to allow index addressing regardless of SDK list implementation
            var comps = new List<IComponent>();
            foreach (IComponent c in pm.Components) comps.Add(c);

            int canUpdate = Math.Min(rows.Count, comps.Count);
            int updated = 0, windowsSet = 0;

            for (int i = 0; i < canUpdate; i++)
            {
                var spec = rows[i];
                var comp = comps[i];

                // Rename to analyte
                SetComponentName(comp, spec.Analyte);

                // Set absolute retention time
                if (comp.RetentionTimeSettings != null && comp.RetentionTimeSettings.Time != null)
                {
                    comp.RetentionTimeSettings.Time.Value = spec.Rt;
                }

                // Optional: set a window/tolerance if exposed by this SDK build
                if (spec.Window.HasValue && comp.RetentionTimeSettings != null)
                {
                    object rts = comp.RetentionTimeSettings;
                    var t = rts.GetType();
                    var prop =
                        t.GetProperty("Window") ??
                        t.GetProperty("TimeWindow") ??
                        t.GetProperty("Tolerance") ??
                        t.GetProperty("TimeTolerance");
                    if (prop != null && prop.CanWrite)
                    {
                        try { prop.SetValue(rts, spec.Window.Value, null); windowsSet++; }
                        catch { /* best effort */ }
                    }
                }

                updated++;
            }

            tx.Complete();

            // Report any size mismatch with the template
            int csvExcess = rows.Count - comps.Count;
            int placeholdersLeft = comps.Count - rows.Count;

            string msg =
                "Processing Method (in-place): " + pmName + Environment.NewLine +
                "Components updated: " + updated + Environment.NewLine +
                (windowsSet > 0 ? ("Windows set: " + windowsSet + Environment.NewLine) : "") +
                (csvExcess > 0 ? ("CSV had " + csvExcess + " more analyte(s) than template components." + Environment.NewLine) : "") +
                (placeholdersLeft > 0 ? (placeholdersLeft + " placeholder component(s) remain unchanged.") : "");
            MessageBox.Show(msg, "Processing Method Updated", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
    }

    // ---------- helpers ----------

    public static void UpdateFromRtSpecsInPlace(IInjection injection, IList<Krait.Chromeleon.RtSpec> rows)
    {
        if (injection == null) throw new ArgumentNullException("injection");
        if (rows == null || rows.Count == 0) return;

        ISequence seq = injection.Parent as ISequence;
        if (seq == null) throw new InvalidOperationException("Injection has no parent sequence.");

        string pmName = (injection.ProcessingMethodName != null) ? injection.ProcessingMethodName.Value : null;
        if (string.IsNullOrWhiteSpace(pmName))
            throw new InvalidOperationException("Injection has no ProcessingMethodName assigned.");

        IProcessingMethod pm = FindProcessingMethodByName(seq, pmName);
        if (pm == null) throw new InvalidOperationException("Processing method '" + pmName + "' not found under the sequence.");

        using (var tx = new System.Transactions.TransactionScope(System.Transactions.TransactionScopeOption.RequiresNew, TimeSpan.MaxValue))
        {
            // snapshot
            var comps = new List<IComponent>();
            foreach (IComponent c in pm.Components) comps.Add(c);

            int canUpdate = Math.Min(rows.Count, comps.Count);
            int windowsSet = 0;

            for (int i = 0; i < canUpdate; i++)
            {
                var spec = rows[i];
                var comp = comps[i];

                // rename
                SetComponentName(comp, spec.Analyte);

                // RT
                if (comp.RetentionTimeSettings != null && comp.RetentionTimeSettings.Time != null)
                    comp.RetentionTimeSettings.Time.Value = spec.Rt;

                // window
                if (spec.Window.HasValue && comp.RetentionTimeSettings != null)
                {
                    object rts = comp.RetentionTimeSettings;
                    var t = rts.GetType();
                    var prop = t.GetProperty("Window") ?? t.GetProperty("TimeWindow") ?? t.GetProperty("Tolerance") ?? t.GetProperty("TimeTolerance");
                    if (prop != null && prop.CanWrite)
                    {
                        try { prop.SetValue(rts, spec.Window.Value, null); windowsSet++; }
                        catch { }
                    }
                }
            }

            tx.Complete();
        }
    }
    
    // Detect JSON vs CSV and dispatch accordingly
    private static List<RtSpec> LoadRtSpecsFromFile(string path)
    {
        string text = File.ReadAllText(path);
        string trimmed = text.TrimStart();
        bool isJson = path.EndsWith(".json", StringComparison.OrdinalIgnoreCase) ||
                      (trimmed.Length > 0 && (trimmed[0] == '{' || trimmed[0] == '['));
        return isJson ? LoadJsonRows(text) : LoadCsvRows(text);
    }

    private static List<RtSpec> LoadJsonRows(string json)
    {
        var list = new List<RtSpec>();

        // Parse with double precision for numbers
        using (var sr = new StringReader(json))
        using (var jr = new JsonTextReader(sr) { FloatParseHandling = FloatParseHandling.Double })
        {
            var token = JToken.ReadFrom(jr);

            // Case A: { "components": [ ... ] }
            if (token.Type == JTokenType.Object && token["components"] is JArray compsA)
            {
                AddFromArray(compsA, list);
                return list;
            }

            // Case B: bare array: [ ... ]
            if (token is JArray compsB)
            {
                AddFromArray(compsB, list);
                return list;
            }

            // Case C: dict keyed by analyte: { "Fluoride": { "rt": 6.84, ... }, ... }
            if (token.Type == JTokenType.Object)
            {
                foreach (var prop in ((JObject)token).Properties())
                {
                    var o = prop.Value as JObject;
                    if (o == null) continue;
                    double? rt = TryGetDouble(o, "rt", "Rt", "retentionTime", "RetentionTime");
                    double? win = TryGetDouble(o, "window", "Window", "timeWindow", "TimeWindow", "tolerance", "Tolerance");
                    if (rt.HasValue)
                        list.Add(new RtSpec { Analyte = prop.Name, Rt = rt.Value, Window = win });
                }
            }
        }
        return list;
    }

    private static void AddFromArray(JArray arr, List<RtSpec> list)
    {
        foreach (var t in arr)
        {
            var o = t as JObject;
            if (o == null) continue;
            string name = TryGetString(o, "name", "Name", "analyte", "Analyte", "component", "Component");
            double? rt = TryGetDouble(o, "rt", "Rt", "retentionTime", "RetentionTime");
            double? win = TryGetDouble(o, "window", "Window", "timeWindow", "TimeWindow", "tolerance", "Tolerance");
            if (!string.IsNullOrWhiteSpace(name) && rt.HasValue)
                list.Add(new RtSpec { Analyte = name.Trim(), Rt = rt.Value, Window = win });
        }
    }

    private static string TryGetString(JObject o, params string[] keys)
    {
        for (int i = 0; i < keys.Length; i++)
        {
            JToken v = o[keys[i]];
            if (v != null && v.Type != JTokenType.Null) return v.ToString();
        }
        return null;
    }

    private static double? TryGetDouble(JObject o, params string[] keys)
    {
        for (int i = 0; i < keys.Length; i++)
        {
            JToken v = o[keys[i]];
            if (v == null || v.Type == JTokenType.Null) continue;
            if (v.Type == JTokenType.Integer || v.Type == JTokenType.Float) return v.Value<double>();
            double d;
            if (double.TryParse(v.ToString(), NumberStyles.Any, CultureInfo.InvariantCulture, out d)) return d;
        }
        return null;
    }

    private static void WriteRtSpecsJson(string jsonPath, IList<RtSpec> rows, string pmName)
    {
        var payload = new JObject
        {
            ["schema_version"] = "1.0",
            ["context"] = new JObject
            {
                ["processing_method_name"] = pmName,
                ["units"] = new JObject { ["rt"] = "min", ["window"] = "min" }
            },
            ["components"] = new JArray(rows.Select(r => new JObject
            {
                ["name"] = r.Analyte,
                ["rt"] = r.Rt,
                ["window"] = r.Window.HasValue ? new JValue(r.Window.Value) : JValue.CreateNull()
            }))
        };

        Directory.CreateDirectory(Path.GetDirectoryName(jsonPath));
        var tmp = jsonPath + ".tmp";
        File.WriteAllText(tmp, payload.ToString(Formatting.None));
        if (File.Exists(jsonPath)) File.Delete(jsonPath);
        File.Move(tmp, jsonPath);
    }

    private static List<RtSpec> LoadCsvRows(string csvPath)
    {
        var list = new List<RtSpec>();
        string[] lines = File.ReadAllLines(csvPath);

        for (int i = 0; i < lines.Length; i++)
        {
            string raw = lines[i];
            if (string.IsNullOrWhiteSpace(raw)) continue;

            string[] parts = raw.Split(new[] { ',', ';', '\t' }, StringSplitOptions.None);
            if (parts.Length < 2) continue;

            // Skip header if 2nd token isn't numeric
            if (i == 0 && !double.TryParse(parts[1].Trim(), NumberStyles.Any, CultureInfo.InvariantCulture, out _))
                continue;

            string analyte = parts[0].Trim();
            if (string.IsNullOrWhiteSpace(analyte)) continue;

            if (!double.TryParse(parts[1].Trim(), NumberStyles.Any, CultureInfo.InvariantCulture, out double rt))
                continue;

            double? win = null;
            if (parts.Length >= 3)
            {
                if (double.TryParse(parts[2].Trim(), NumberStyles.Any, CultureInfo.InvariantCulture, out double w))
                    win = w;
            }

            list.Add(new RtSpec { Analyte = analyte, Rt = rt, Window = win });
        }
        return list;
    }

    private static IProcessingMethod FindProcessingMethodByName(ISequence seq, string name)
    {
        foreach (IDataItem child in seq.Children)
        {
            IProcessingMethod pm = child as IProcessingMethod;
            if (pm == null) continue;
            string n = (pm.Name != null) ? pm.Name : null; // PM.Name is typically IStringValue
            if (!string.IsNullOrEmpty(n) && string.Equals(n, name, StringComparison.OrdinalIgnoreCase))
                return pm;
        }
        return null;
    }

    private static void SetComponentName(IComponent comp, string name)
    {
        if (comp == null) return;

        var nameProp = comp.GetType().GetProperty("Name");
        if (nameProp == null) return;

        object nameObj = nameProp.GetValue(comp, null);
        if (nameObj != null)
        {
            var svType = nameObj.GetType();               // IStringValue case
            var valueProp = svType.GetProperty("Value");
            if (valueProp != null) { valueProp.SetValue(nameObj, name, null); return; }
        }

        // Fallback: writable string property (if any build exposes it)
        if (nameProp.CanWrite) nameProp.SetValue(comp, name, null);
    }
}
