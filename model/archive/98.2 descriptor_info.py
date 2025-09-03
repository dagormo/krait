from mordred import Calculator, descriptors
import pandas as pd

# === MORDRED DESCRIPTORS ===
calc = Calculator(descriptors, ignore_3D=True)
desc_info = [
    {
        "Descriptor Java Class": type(d).__module__.split(".")[-1].capitalize() + "Descriptor",
        "Descriptor": str(d),
        "Description": d.__doc__.splitlines()[0] if d.__doc__ else "No description available",
        "Class": "2D"
    }
    for d in calc.descriptors
]
df_mordred = pd.DataFrame(desc_info)

# === PADEL DESCRIPTORS ===
df_padel = pd.read_csv("../../data/PaDel-Descriptor/PaDEL_descriptors.csv")
df_mordred["Source"] = "Mordred"
df_padel["Source"] = "PaDEL"

# === COMBINE BOTH ===
combined = pd.concat([df_padel, df_mordred], ignore_index=True)
combined = combined.rename(columns={"Descriptor Java Class": "Category"})

# === WRAP STRINGS IN QUOTES ===
combined["Description"] = combined["Description"].apply(lambda x: f'"{x}"' if pd.notna(x) else x)

# === REMOVE DUPLICATES: Keep PaDEL version if present ===
combined = combined.sort_values("Source", ascending=False)  # PaDEL before Mordred
combined = combined.drop_duplicates(subset="Descriptor", keep="first")

# === EXPORT ===
combined.to_csv("descriptor_metadata_combined.csv", index=False)
print(f"✅ Saved {len(combined)} unique descriptors to descriptor_metadata_combined.csv")
