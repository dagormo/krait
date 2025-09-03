import pandas as pd

# Load descriptor data
pca_df = pd.read_csv(r"../data/pca_space_95pct.csv")  # assumes 'Name' column matches 'Analyte'

# Load the .xlsm workbook
excel_file = r"../data/VirtualColumn.xlsm"

# List of eluent sheet names
eluents = ['Hydroxide']  # , 'Carbonate', 'MSA']

# Loop through each eluent and save a merged file
for eluent in eluents:
    # Load specific worksheet
    retention_df = pd.read_excel(excel_file, sheet_name=eluent)

    # Standardize column names for consistency
    retention_df.columns = retention_df.columns.str.strip()
    pca_df.columns = pca_df.columns.str.strip()

    # Merge on analyte name
    merged = pd.merge(pca_df, retention_df, left_on='Name', right_on='Analyte', how='inner')

    # Save merged result
    output_filename = f'merged_{eluent.lower()}.csv'
    merged.to_csv(output_filename, index=False)

    print(f"{eluent}: {len(merged)} rows merged. Saved to {output_filename}")
