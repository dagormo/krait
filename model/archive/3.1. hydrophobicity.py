import os
import pandas as pd
import numpy as np

# ----- CONFIG -----
eluents = ['hydroxide', 'carbonate', 'msa']
vc_file_template  = "merged_{}.csv"
all_output_path   = "column_hydrophobicity_all.csv"

# Map each eluent to the temperatures and analyte pair to use
temps_map = {
    'hydroxide': [30.0],
    'carbonate': [30.0],
    'msa':       [40.0],
}
pairs_map = {
    'hydroxide': ("Propanesulfonate", "Butanesulfonate"),
    'carbonate': ("Propanesulfonate", "Butanesulfonate"),
    'msa':       ("Ethylamine",        "Propylamine"),
}

def extract_hydrophobicity_index(vc_df: pd.DataFrame,
                                 pair: tuple[str, str],
                                 temps: list[float]) -> pd.DataFrame:
    a1, a2 = pair
    dfT = vc_df[vc_df['Temp_rnd'].isin(temps)]
    if dfT.empty:
        return pd.DataFrame(columns=['Chemistry', 'Hydrophobicity index'])
    pivot = (
        dfT.pivot_table(
            index=['Chemistry','Temp_rnd'],
            columns='Analyte',
            values="k'"
        )
    )
    if not set(pair).issubset(pivot.columns):
        return pd.DataFrame(columns=['Chemistry','Hydrophobicity index'])
    ratio = pivot[pair[1]] / pivot[pair[0]]
    out = (
        pd.DataFrame({
            'Chemistry':            pivot.index.get_level_values('Chemistry'),
            'Hydrophobicity index': ratio.values,
        })
        .groupby('Chemistry', as_index=False)
        .mean()
    )
    return out

def main():
    combined_results = []

    for eluent in eluents:
        path = vc_file_template.format(eluent)
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue

        vc_df = pd.read_csv(path)
        missing = {'Analyte','Chemistry','log(k)','Temperature'} - set(vc_df.columns)
        if missing:
            print(f"❌ Missing columns in {path}: {missing}")
            continue

        # compute helpers in-memory
        vc_df["k'"]      = 10 ** vc_df["log(k)"]
        vc_df['Temp_rnd'] = vc_df["Temperature"].round(1)

        # compute hydrophobicity
        pair    = pairs_map[eluent]
        temps   = temps_map[eluent]
        hydro   = extract_hydrophobicity_index(vc_df, pair, temps)
        if hydro.empty:
            print(f"⚠️  No hydrophobicity data for {eluent}")
            continue
        hydro['Eluent'] = eluent
        combined_results.append(hydro)

        # merge only the Hydrophobicity index back, then drop the helpers
        annotated = vc_df.merge(
            hydro[['Chemistry','Hydrophobicity index']],
            on='Chemistry', how='left'
        ).drop(columns=["k'", "Temp_rnd"])

        annotated.to_csv(path, index=False)
        print(f"✔️  Updated {path} (added only Hydrophobicity index)")

    if combined_results:
        pd.concat(combined_results, ignore_index=True).to_csv(all_output_path, index=False)
        print(f"✔️  Wrote combined results to {all_output_path}")

if __name__ == "__main__":
    main()
