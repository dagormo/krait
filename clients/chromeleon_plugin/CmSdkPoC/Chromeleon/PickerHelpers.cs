using System;
using System.Collections;
using System.Collections.Generic;
using Thermo.Chromeleon.Sdk.Interfaces.UserInterface;

namespace Krait.Chromeleon
{
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

        public static List<string> DefaultColumns()
        {
            return DistinctPreserveOrder(new[]
            {
                "AS10","AS11","AS11-HC","AS12A","AS14","AS14","AS14A","AS14A","AS15",
                "AS16","AS17","AS18","AS19","AS19","AS20","AS22","AS22-Fast","AS22-Fast-4um",
                "AS23","AS24","AS27","AS4A-SC","AS9-HC","CS12A","CS16","CS17","CS18","CS19","PA20"
            });
        }

        private static List<string> DistinctPreserveOrder(IEnumerable<string> items)
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
}
