import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)

os.makedirs("results", exist_ok=True)

# ---- choices to set AFTER reading the sweeps below (and justify in README) ----
CHOSEN_K = 7
CHOSEN_DEPTH = 5
CHOSEN_THRESHOLD = 0.5
# -------------------------------------------------------------------------------

# ===== A2. Clean and look =====
df = pd.read_csv("data/concrete.csv")
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:\n", df.isna().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Rows after cleaning:", len(df))
print(df.describe().T)

X = df.drop(columns=["strength"])
y = df["strength"]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(df["cement"], y, s=8); ax[0].set_xlabel("cement (kg/m3)"); ax[0].set_ylabel("strength (MPa)")
ax[1].scatter(df["age"], y, s=8); ax[1].set_xlabel("age (days)"); ax[1].set_xscale("log")
plt.tight_layout(); plt.savefig("results/A2_scatter.png", dpi=120); plt.close()

# ===== A3. Split =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

def adj_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# ===== A4. Linear regression =====
lin = LinearRegression().fit(X_train, y_train)
ptr, pte = lin.predict(X_train), lin.predict(X_test)
r2_tr = r2_score(y_train, ptr)
a4 = pd.DataFrame([{
    "train_RMSE": np.sqrt(mean_squared_error(y_train, ptr)),
    "test_RMSE": np.sqrt(mean_squared_error(y_test, pte)),
    "test_MAE": mean_absolute_error(y_test, pte),
    "train_R2": r2_tr,
    "train_adjR2": adj_r2(r2_tr, len(X_train), X_train.shape[1]),
    "test_R2": r2_score(y_test, pte)}])
print("\nA4 linear regression:\n", a4.round(3).to_string(index=False))
print("intercept:", round(lin.intercept_, 4))
for name, c in zip(X.columns, lin.coef_):
    print(f"  {name}: {c:.5f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X_test["age"], y_test - pte, s=10)
ax.axhline(0, color="k", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("age (days)"); ax.set_ylabel("residual (MPa)")
plt.tight_layout(); plt.savefig("results/A4_residuals.png", dpi=120); plt.close()

# ===== A5. Polynomial regression =====
rows = []
for d in [1, 2, 3, 4]:
    poly = make_pipeline(PolynomialFeatures(degree=d, include_bias=False),
                         StandardScaler(), LinearRegression())
    poly.fit(X_train, y_train)
    p = poly[0].n_output_features_
    a, b = poly.predict(X_train), poly.predict(X_test)
    r2t = r2_score(y_train, a)
    rows.append({"degree": d, "p": p,
                 "train_RMSE": np.sqrt(mean_squared_error(y_train, a)),
                 "test_RMSE": np.sqrt(mean_squared_error(y_test, b)),
                 "train_R2": r2t, "train_adjR2": adj_r2(r2t, len(X_train), p),
                 "test_R2": r2_score(y_test, b)})
print("\nA5 polynomial regression:\n", pd.DataFrame(rows).round(3).to_string(index=False))

# ===== A6. Classification =====
y_class = (df["strength"] >= 40).astype(int)
print("\nClass proportions:\n", y_class.value_counts(normalize=True))
Xc_train, Xc_test, c_train, c_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class)

# KNN sweep
ks = [1, 3, 5, 7, 15, 25, 51, 101]
rows = []
for k in ks:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(Xc_train, c_train)
    rows.append({"K": k, "train_acc": m.score(Xc_train, c_train), "test_acc": m.score(Xc_test, c_test)})
knn_sweep = pd.DataFrame(rows)
print("\nKNN sweep:\n", knn_sweep.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(knn_sweep.K, knn_sweep.train_acc, "o-", label="train")
ax.plot(knn_sweep.K, knn_sweep.test_acc, "s-", label="test")
ax.set_xscale("log"); ax.set_xlabel("K"); ax.set_ylabel("accuracy"); ax.legend()
plt.tight_layout(); plt.savefig("results/A6_knn_sweep.png", dpi=120); plt.close()

knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=CHOSEN_K)).fit(Xc_train, c_train)
knn_noscale = KNeighborsClassifier(n_neighbors=CHOSEN_K).fit(Xc_train, c_train)
print(f"\nK={CHOSEN_K}: test acc with scaler {knn.score(Xc_test, c_test):.3f}, "
      f"without {knn_noscale.score(Xc_test, c_test):.3f}")

accs = []
for rs in range(5):
    a_tr, a_te, b_tr, b_te = train_test_split(X, y_class, test_size=0.2, random_state=rs, stratify=y_class)
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=CHOSEN_K)).fit(a_tr, b_tr)
    accs.append(m.score(a_te, b_te))
print("5-split test accuracies:", np.round(accs, 3), "mean", round(np.mean(accs), 3), "std", round(np.std(accs), 3))

# Tree sweep
rows = []
for d in [1, 2, 3, 4, 5, 6, 8, None]:
    t = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=0).fit(Xc_train, c_train)
    rows.append({"max_depth": d, "leaves": t.get_n_leaves(),
                 "train_acc": t.score(Xc_train, c_train), "test_acc": t.score(Xc_test, c_test)})
tree_sweep = pd.DataFrame(rows)
print("\nTree sweep:\n", tree_sweep.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(6, 4))
xs = [str(d) for d in tree_sweep.max_depth]
ax.plot(xs, tree_sweep.train_acc, "o-", label="train")
ax.plot(xs, tree_sweep.test_acc, "s-", label="test")
ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy"); ax.legend()
plt.tight_layout(); plt.savefig("results/A6_tree_sweep.png", dpi=120); plt.close()

tree = DecisionTreeClassifier(criterion="entropy", max_depth=CHOSEN_DEPTH, random_state=0).fit(Xc_train, c_train)
print(f"\nTree depth={CHOSEN_DEPTH}, top two levels:")
print(export_text(tree, feature_names=list(X.columns), max_depth=2))

# ===== A7. Metrics =====
def metrics_row(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"model": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred)}

proba = knn.predict_proba(Xc_test)[:, 1]
final_pred = (proba >= CHOSEN_THRESHOLD).astype(int)
a_metrics = pd.DataFrame([
    metrics_row(f"KNN(K={CHOSEN_K}, t={CHOSEN_THRESHOLD})", c_test, final_pred),
    metrics_row(f"Tree(depth={CHOSEN_DEPTH})", c_test, tree.predict(Xc_test))])
print("\nA7 metrics:\n", a_metrics.round(3).to_string(index=False))
a_metrics.to_csv("results/A_metrics.csv", index=False)

rows = []
for t in [0.3, 0.5, 0.7, 0.9]:
    r = metrics_row(f"KNN t={t}", c_test, (proba >= t).astype(int))
    rows.append(r)
print("\nThreshold table (KNN):\n", pd.DataFrame(rows).round(3).to_string(index=False))

# ===== A8. Feature importance =====
pipe = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
pi_knn = permutation_importance(knn, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)
pi_tree = permutation_importance(tree, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)
imp = pd.DataFrame({"raw_coef": lin.coef_, "std_coef": pipe[-1].coef_,
                    "tree_importance": tree.feature_importances_,
                    "perm_knn": pi_knn.importances_mean,
                    "perm_tree": pi_tree.importances_mean}, index=X.columns)
print("\nA8 importances:\n", imp.round(4))
imp.to_csv("results/A_importance.csv")
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
imp["std_coef"].plot.bar(ax=ax[0], title="standardised coef (MPa/sd)")
imp["tree_importance"].plot.bar(ax=ax[1], title="tree importance")
imp["perm_knn"].plot.bar(ax=ax[2], title="permutation (KNN, F1 drop)")
plt.tight_layout(); plt.savefig("results/A8_importance.png", dpi=120); plt.close()

# ===== Bonus: physics-based features =====
def bonus(Xd):
    Z = Xd.copy()
    Z["age"] = np.log(Z["age"])
    return Z
def bonus2(Xd):
    Z = bonus(Xd)
    Z["wb"] = Z["water"] / (Z["cement"] + Z["slag"] + Z["ash"])
    return Z.drop(columns=["water"])
for name, f in [("log(age)", bonus), ("log(age) + w/b", bonus2)]:
    m = LinearRegression().fit(f(X_train), y_train)
    print(f"Bonus {name}: p={f(X_train).shape[1]}, test R2={r2_score(y_test, m.predict(f(X_test))):.3f}")