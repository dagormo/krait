using System;
using System.Collections.Generic;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace Krait.UI
{
    public static class ChartHelpers
    {
        public static void ShowGradientChart(List<Tuple<double, double>> points, string methodName)
        {
            var form = new Form
            {
                Text = "EGC Gradient: " + methodName,
                Width = 900,
                Height = 600
            };

            var chart = new Chart { Dock = DockStyle.Fill };
            form.Controls.Add(chart);

            var area = new ChartArea();
            area.AxisX.Title = "Time [min]";
            area.AxisY.Title = "EGC Concentration [mM]";
            chart.ChartAreas.Add(area);

            var series = new Series
            {
                ChartType = SeriesChartType.Line,
                BorderWidth = 2,
                XValueType = ChartValueType.Double,
                YValueType = ChartValueType.Double
            };

            foreach (var pt in points)
                series.Points.AddXY(pt.Item1, pt.Item2);

            chart.Series.Add(series);
            Application.Run(form);
        }
    }
}
