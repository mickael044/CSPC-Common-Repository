import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

os.makedirs("results", exist_ok=True)
CHOSEN_K = 3   # set after reading the elbow plot (justify in README)

# ===== B1 =====
wine = load_wine(as_frame=True)
X = wine.data
cultivar = wine.target
print(wine.DESCR)
print(X.describe().T)
print("\nStd (sorted):\n", X.std().sort_values(ascending=False))

# ===== B2. PCA on raw data =====
pca_raw = PCA().fit(X)
print("\nRaw PCA explained variance ratio:", np.round(pca_raw.explained_variance_ratio_, 4))
print("PC1 loadings (raw):\n",
      pd.Series(pca_raw.components_[0], index=X.columns).sort_values(key=abs, ascending=False))

# ===== B3. Scale then PCA =====
scaler = StandardScaler()
Z = scaler.fit_transform(X)
print("\nZ mean:", np.round(Z.mean(axis=0), 6)[:3], "... std:", np.round(Z.std(axis=0), 6)[:3])
pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
print("\nExplained variance ratio:", np.round(evr, 4))
print("Cumulative:", np.round(cum, 4))
print("Components for 90%:", int(np.argmax(cum >= 0.9)) + 1, "| PC1-PC2 keep:", round(cum[1], 4))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, 14), evr, label="per component")
ax.plot(range(1, 14), cum, "o-", color="C1", label="cumulative")
ax.axhline(0.9, color="k", ls="--", lw=0.8)
ax.set_xlabel("component"); ax.set_ylabel("explained variance ratio"); ax.legend()
plt.tight_layout(); plt.savefig("results/B3_scree.png", dpi=120); plt.close()

load = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
print("\nLoadings sorted by PC1:\n", load.sort_values("PC1").round(3))
print("\nLoadings sorted by PC2:\n", load.sort_values("PC2").round(3))

scores = pca.transform(Z)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(scores[:, 0], scores[:, 1], s=15)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.tight_layout(); plt.savefig("results/B3_pc_scatter.png", dpi=120); plt.close()

# ===== B4. K-Means =====
inert = [KMeans(n_clusters=3, init="k-means++", n_init=1, random_state=rs).fit(Z).inertia_ for rs in range(8)]
print("\nB4a inertia, n_init=1, seeds 0-7:", np.round(inert, 2))

ks = range(1, 11)
inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z).inertia_ for k in ks]
print("B4b inertia k=1..10:", np.round(inertias, 1))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(ks), inertias, "o-"); ax.set_xlabel("k"); ax.set_ylabel("inertia")
plt.tight_layout(); plt.savefig("results/B4_elbow.png", dpi=120); plt.close()

# ===== B5. Clusters =====
km = KMeans(n_clusters=CHOSEN_K, init="k-means++", n_init=10, max_iter=300, random_state=0).fit(Z)
print("\nCluster sizes:", np.bincount(km.labels_))
centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=X.columns)
print("Centres (real units):\n", centers.round(2).T)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(scores[:, 0], scores[:, 1], c=km.labels_, s=15, cmap="viridis")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.tight_layout(); plt.savefig("results/B5_clusters.png", dpi=120); plt.close()

# ===== B6. The reveal =====
def agreement(labels):
    table = pd.crosstab(cultivar, labels)
    return table, table.max(axis=0).sum() / len(cultivar)

km_raw = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
km_pc2 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(scores[:, :2])
res = []
for name, lab in [("scaled", KMeans(n_clusters=3, n_init=10, random_state=0).fit(Z).labels_),
                  ("raw", km_raw.labels_),
                  ("PC1-PC2 of scaled", km_pc2.labels_)]:
    table, ag = agreement(lab)
    print(f"\nCrosstab ({name}):\n", table)
    res.append({"version": name, "agreement": ag})
agree = pd.DataFrame(res)
print("\nAgreement:\n", agree.round(4).to_string(index=False))
agree.to_csv("results/B_agreement.csv", index=False)

# ===== Bonus: what do labels buy you? =====
X_tr, X_te, y_tr, y_te = train_test_split(X, cultivar, test_size=0.25, random_state=42, stratify=cultivar)
m1 = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(X_tr, y_tr)
m2 = make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(5)).fit(X_tr, y_tr)
print("\nBonus supervised test accuracy: all 13 features", round(m1.score(X_te, y_te), 3),
      "| PCA(2)", round(m2.score(X_te, y_te), 3))
