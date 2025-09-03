import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === FILES ===
LOADINGS_FILE = "../../data/pca_loadings.csv"
IMPORTANCE_FILE = "../../data/pc_feature_importances.csv"

# === LOAD DATA ===
loadings = pd.read_csv(LOADINGS_FILE, index_col=0)  # descriptors x PCs
importances = pd.read_csv(IMPORTANCE_FILE)

# Align and filter PCs (important!)
pc_names = importances["Feature"].tolist()
pc_names = [x for x in pc_names if x.startswith('PC')]
importance_values = importances.set_index("Feature").loc[pc_names, "Importance"].values
loadings = loadings[pc_names]

# === COMPUTE DESCRIPTOR INFLUENCE ===
# Multiply loadings by PC importance to get per-descriptor influence
descriptor_influence = loadings.values @ importance_values
descriptor_df = pd.DataFrame({
    "Descriptor": loadings.index,
    "Influence": descriptor_influence
}).sort_values("Influence", key=abs, ascending=False)

# === OPTIONAL: Keep top N for visualization ===
top_n = 50
top_df = descriptor_df.head(top_n)

# === PLOT ===
plt.figure()
colors = ["green" if x > 0 else "red" for x in top_df["Influence"]]
bars = plt.barh(top_df["Descriptor"], top_df["Influence"], color=colors)
plt.xlabel("Influence on Retention")
plt.title("Top Descriptor Influences (via PCA and Model)")
plt.axvline(0, color='gray', linewidth=0.8)
plt.gca().invert_yaxis()

# Annotate bars
for bar in bars:
    plt.text(bar.get_width() + 0.01 if bar.get_width() > 0 else bar.get_width() - 0.05,
             bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.3f}", va="center", fontsize=8)

plt.show()

descriptor_df.to_csv("descriptor_influence.csv", index=False)