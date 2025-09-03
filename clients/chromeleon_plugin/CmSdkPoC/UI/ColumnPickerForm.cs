using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace Krait.UI
{
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

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
            if (DialogResult == DialogResult.OK)
                SelectedColumn = combo.SelectedItem as string ?? "";
        }
    }
}
