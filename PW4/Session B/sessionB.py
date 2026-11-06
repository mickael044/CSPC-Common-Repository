import pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

print("--- B1. Load and hide the labels ---")
#load the wine dataset as a pandas
wine = load_wine(as_frame=True)

#seperte x and cultivar
X = wine.data
cultivar = wine.target #set aside until B6[cite: 22]
print("\n--- Data Description ---")
print(wine.DESCR[:1500])

#print statistical summary of the features
print("\n--- X.describe() ---")
print(X.describe())

print("\n--- B2. PCA on the raw data ---")
#initialize PCA keeping all components
pca_raw = PCA()
pca_raw.fit(X)

#get the explained variance ratio of the first principal component (PC1)
pc1_variance = pca_raw.explained_variance_ratio_[0]
print(f"Explained variance ratio of PC1: {pc1_variance:.4f} ({pc1_variance*100:.2f}%)")

#get the loadings of PC1 and sort them
pc1_loadings = pd.Series(pca_raw.components_[0], index=X.columns).sort_values(ascending=False)
print("\nPC1 loadings (sorted):")
print(pc1_loadings)