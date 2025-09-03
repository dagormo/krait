import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# Step 1: Load your PCA-transformed dataset and extract just the PC columns (adjust this if your headers differ)
df = pd.read_csv("pca_space_95pct.csv")
pc_columns = [col for col in df.columns if col.startswith("PC")]
X = df[pc_columns]

# K-means clustering
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X)

score = silhouette_score(X, labels)
print(f"Silhouette score (k={n_clusters}): {score:.3f}")

# Pairwise Euclidean distances in PCA space
dist_matrix = pairwise_distances(X, metric='euclidean')
dist_df = pd.DataFrame(dist_matrix, index=df["Name"], columns=df["Name"])
plt.figure(figsize=(14,12))
sns.heatmap(dist_df, cmap='viridis', square=True)

plt.title("Pairwise Distance Heatmap – PCA Space")
plt.tight_layout()
plt.savefig("pca_distance_heatmap.png", dpi=300)
plt.show()