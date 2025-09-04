import pandas as pd


def default_analytes_table():
    rows = [
        ("Bromide", 6.249, 4.83, 3.699),
        ("Carbonate", 4.952, 3.205, 2.335),
        ("Chloride", 3.395, 2.883, 2.481),
        ("Fluoride", 2.12, 2.026, 1.943),
        ("Iodide", 28.057, 19.69, 13.023),
        ("Nitrate", 6.609, 5.059, 3.82),
        ("Nitrite", 3.93, 3.254, 2.707),
        ("Phosphate", 17.809, 7.067, 3.128),
        ("Sulfate", 6.296, 3.756, 2.521),
    ]
    return pd.DataFrame(rows, columns=["Analyte","RT@Conc1 (min)","RT@Conc2 (min)","RT@Conc3 (min)"])

def default_gradient_table():
    return pd.DataFrame({
        "time_min": [0.0, 10.0, 20.0],
        "conc_mM": [5, 50, 100],
        "curve":   [9, 1, 5]
    })
