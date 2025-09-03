using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Krait.Chromeleon;

namespace Krait.Utils
{
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
}
