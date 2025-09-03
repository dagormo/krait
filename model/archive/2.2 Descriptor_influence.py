import pandas as pd

df = pd.read_csv("pca_space_95pct.csv")
Name = "Acetate"  # example
analyte_row = df[df["Name"] == name]
descriptor_values = analyte_row[["logP", "TPSA", "MoRSE_mass", "WHIM_symmetry"]].iloc[0]
