using System;
using System.Collections.Generic;
using System.Globalization;
using Thermo.Chromeleon.Sdk;
using Thermo.Chromeleon.Sdk.Common;
using Thermo.Chromeleon.Sdk.Interfaces;
using Thermo.Chromeleon.Sdk.Interfaces.Data;
using Thermo.Chromeleon.Sdk.Interfaces.Evaluation;

namespace Krait.Chromeleon
{
    public sealed class PeakInfo
    {
        public int Index { get; set; }
        public string Name { get; set; }
        public double RetentionTimeMinutes { get; set; }
    }

    public static class PeakReader
    {
        public static List<PeakInfo> GetPeaks(IInjection injection, string signalName = null, int maxPeaks = 2000)
        {
            if (injection == null) throw new ArgumentNullException(nameof(injection));
            if (string.IsNullOrEmpty(signalName))
                signalName = FirstSignalName(injection);

            var peaks = new List<PeakInfo>();
            var fac = CmSdk.GetItemFactory();

            string[] rtTokens = { "peak.retention_time", "peak.retention time", "peak.retentiontime", "peak.time" };
            const string nameFormula = "peak.name";

            for (int i = 0; i < maxPeaks; i++)
            {
                var ctxIndex = fac.CreateEvaluationContext(signalName, i);

                var nameRv = injection.Evaluate(nameFormula, ctxIndex);
                string name = ResolveName(nameRv);

                double rtMin = ResolveRtMinutes(injection, ctxIndex, rtTokens);

                if (double.IsNaN(rtMin) && !string.IsNullOrEmpty(name))
                {
                    var ctxByName = fac.CreateEvaluationContext(signalName, name);
                    rtMin = ResolveRtMinutes(injection, ctxByName, rtTokens);
                }

                if ((nameRv == null || nameRv.IsError) && double.IsNaN(rtMin))
                    break;

                if (string.IsNullOrEmpty(name)) name = "(unnamed)";
                peaks.Add(new PeakInfo { Index = i, Name = name, RetentionTimeMinutes = rtMin });
            }

            return peaks;
        }

        private static string ResolveName(IReportValue nameRv)
        {
            if (nameRv == null || nameRv.IsError) return null;
            if (nameRv.IsString) return nameRv.StringValue;
            return nameRv.Value != null ? Convert.ToString(nameRv.Value, CultureInfo.InvariantCulture) : null;
        }

        private static double ResolveRtMinutes(IInjection inj, IEvaluationContext ctx, string[] rtTokens)
        {
            foreach (var token in rtTokens)
            {
                var rv = inj.Evaluate(token, ctx);
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
}
