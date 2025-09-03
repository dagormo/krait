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

using Krait.Chromeleon;
using Krait.UI;
using Krait.Utils;

namespace Krait
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();

            Directory.CreateDirectory(@"C:\CM-Exports");

            using (var scope = new CmSdkScope())
            {
                CmSdk.Logon.DoLogon();

                var ui = CmSdk.GetUserInterfaceFactory();
                var fac = CmSdk.GetItemFactory();

                // 1. Pick injections
                var injPicker = ui.CreateObjectPickerDialog();
                injPicker.Title = "Select Injection(s)";
                injPicker.MultiSelect = true;
                injPicker.AddFilter(typeof(IInjection));

                if (injPicker.ShowDialog() != DialogResult.OK) return;
                var injUris = PickerHelpers.GetUrisFromPicker(injPicker);
                if (injUris.Count == 0) return;

                // 1.1 Column picker
                var columns = PickerHelpers.DefaultColumns();
                string selectedColumn = "";
                using (var colDlg = new ColumnPickerForm(columns))
                {
                    if (colDlg.ShowDialog() != DialogResult.OK) return;
                    selectedColumn = colDlg.SelectedColumn ?? "";
                }

                foreach (var uri in injUris)
                {
                    if (!fac.TryGetItem(uri, out IDataItem item)) continue;
                    var inj = item as IInjection;
                    if (inj == null) continue;

                    // Resolve instrument method once
                    var method = InjectionHelpers.ResolveInstrumentMethod(inj, fac);
                    if (method == null)
                    {
                        MessageBox.Show($"Could not resolve instrument method for injection {inj.Name}. Skipping.",
                                        "Krait", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        continue;
                    }

                    // Step 2: export current gradient + components
                    using (var sfd = new SaveFileDialog
                    {
                        Title = "Save current gradient + components (JSON)",
                        Filter = "JSON files (*.json)|*.json|All files (*.*)|*.*",
                        FileName = CommonUtil.Sanitize(inj.Name) + "_current.json",
                        OverwritePrompt = true
                    })
                    {
                        if (sfd.ShowDialog() != DialogResult.OK) return;
                        Krait.Chromeleon.KraitIO.ExportCurrentToJson(inj, method, sfd.FileName, selectedColumn);
                    }

                    // Step 3: import new gradient + components
                    using (var ofd = new OpenFileDialog
                    {
                        Title = "Select NEW gradient + components (JSON)",
                        Filter = "JSON files (*.json)|*.json|All files (*.*)|*.*"
                    })
                    {
                        if (ofd.ShowDialog() != DialogResult.OK) return;
                        Krait.Chromeleon.KraitIO.ImportFromJson(inj, method, ofd.FileName);
                    }
                }
            }
        }

    }
}
