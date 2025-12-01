
# Cell 1 — Imports & fmt
from __future__ import annotations
import os, math
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
    accuracy_score, f1_score
)
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

def fmt(x, digits=3):
    """Pretty-format numbers; returns 'nan' for None/NaN."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "nan"
        return f"{x:.{digits}f}"
    except Exception:
        return str(x)

# ===============================================
# Cell 9.5 — Upload datasets to Colab working dir
# ===============================================
from google.colab import files

print("Please select your CSV files:")
uploaded = files.upload()

print("\nUploaded files:")
for fn in uploaded.keys():
    print(" -", fn)

# Cell 2 — Data & label utilities
def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] == 0:
        raise ValueError("No numeric columns found after filtering numeric types.")
    return num_df

def try_attach_builtin_labels(filename: str, n_rows: int) -> Optional[np.ndarray]:
    name = filename.lower()
    if "iris" in name and n_rows == 150:
        return load_iris().target
    if "wine" in name and n_rows == 178:
        return load_wine().target
    return None

def get_label_series_if_present(df: pd.DataFrame) -> Optional[pd.Series]:
    candidates = {"label", "labels", "target", "class", "species"}
    for c in df.columns:
        if c.lower() in candidates:
            return df[c]
    return None

# Cell 3 — Parameter selection (auto-eps)
def auto_eps(X: np.ndarray, min_samples: int = 3, k_extra: int = 2) -> float:
    k = max(1, min_samples + k_extra)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nn.kneighbors(X)
    kth = np.sort(dists[:, -1])
    eps = float(np.percentile(kth, 90))
    if eps <= 0:
        eps = float(np.mean(kth[kth > 0])) if np.any(kth > 0) else 0.5
    return eps

# Cell 4 — Surrogate tree + DNF export
FIXED_TREE_DEPTH = 4
FIXED_TREE_MIN_LEAF = 5

def train_surrogate_tree(X: np.ndarray, y_pred: np.ndarray) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=FIXED_TREE_DEPTH,
        min_samples_leaf=FIXED_TREE_MIN_LEAF,
        random_state=42
    )
    clf.fit(X, y_pred)
    return clf

def tree_to_dnf(clf: DecisionTreeClassifier, feature_names: List[str]) -> List[str]:
    tree_: _tree.Tree = clf.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    rules = []

    def recurse(node: int, preds: List[str]):
        if feature[node] == _tree.TREE_UNDEFINED:
            value = tree_.value[node][0]
            cls = int(np.argmax(value))
            rule = f"IF {' AND '.join(preds) if preds else 'TRUE'} THEN cluster={cls}"
            rules.append(rule)
            return
        fname = feature_names[feature[node]]
        thr = threshold[node]
        recurse(tree_.children_left[node],  preds + [f"{fname} <= {thr:.6g}"])
        recurse(tree_.children_right[node], preds + [f"{fname} > {thr:.6g}"])

    recurse(0, [])
    return rules

# Cell 5 — DBSCAN objective & diagnostics
def dbscan_density_objective(X: np.ndarray, labels: np.ndarray, eps: float, min_samples: int) -> float:
    """
    Density-surplus objective for DBSCAN:
      - Ignore noise (-1)
      - For each non-noise point, count neighbors within eps that share the same label
      - Surplus = max(0, same_label_neighbors - (min_samples - 1))  # exclude self
      - Objective = sum of surplus over all non-noise points
    """
    # Work only on non-noise points
    mask = labels != -1
    if mask.sum() == 0:
        return 0.0

    Xn = X[mask]
    Ln = labels[mask]

    nn = NearestNeighbors(radius=max(1e-8, eps), metric="euclidean").fit(Xn)
    neigh_ix = nn.radius_neighbors(Xn, return_distance=False)

    surplus_total = 0
    for i, nbrs in enumerate(neigh_ix):
        # same-label neighbors (exclude self)
        same = np.sum(Ln[nbrs] == Ln[i]) - 1
        surplus_total += max(0, same - (min_samples - 1))

    return float(surplus_total)


def structure_diagnostics(Xs: np.ndarray, eps: float, min_samples: int, y_pred: np.ndarray) -> Dict[str, float]:
    k = min_samples + 2
    nnk = NearestNeighbors(n_neighbors=k).fit(Xs)
    dists, _ = nnk.kneighbors(Xs)
    kth = dists[:, -1]
    global_kdist_med = float(np.median(kth))
    global_kdist_iqr = float(np.percentile(kth, 75) - np.percentile(kth, 25))

    nne = NearestNeighbors(radius=eps).fit(Xs)
    deg = np.array([len(ix) for ix in nne.radius_neighbors(Xs, return_distance=False)])

    is_noise = (y_pred == -1)
    is_core = (~is_noise) & (deg >= min_samples)
    is_border = (~is_noise) & (~is_core)

    core_frac = float(np.mean(is_core))
    border_frac = float(np.mean(is_border))
    noise_frac = float(np.mean(is_noise))

    core_kth = kth[is_core] if np.any(is_core) else np.array([])
    core_kdist_med = float(np.median(core_kth)) if core_kth.size else float('nan')
    density_contrast = float(core_kdist_med / global_kdist_med) if global_kdist_med > 0 and not math.isnan(core_kdist_med) else float('nan')

    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    idim = int(np.searchsorted(cum, 0.95) + 1)

    labels_nonoise = y_pred[y_pred != -1]
    if labels_nonoise.size:
        _, counts = np.unique(labels_nonoise, return_counts=True)
        sizes = np.sort(counts.astype(float))
        gini = (np.sum((2*np.arange(1, len(sizes)+1) - len(sizes) - 1) * sizes) /
                (len(sizes) * np.sum(sizes))) if np.sum(sizes) > 0 else float('nan')
    else:
        gini = float('nan')

    return dict(
        kdist_med=global_kdist_med,
        kdist_iqr=global_kdist_iqr,
        density_contrast=density_contrast,
        frac_core=core_frac,
        frac_border=border_frac,
        frac_noise=noise_frac,
        pcs_95var=idim,
        cluster_size_gini=gini
    )

# Cell 6 — Core evaluation for one CSV
def evaluate_dbscan_for_csv(csv_file: str,
                            out_rules_suffix: str = "_dbscan_DNF.txt",
                            min_samples: int = 3,
                            holdout_surrogate: bool = False,
                            holdout_test_size: float = 0.2) -> Dict[str, object]:

    # --- Load raw data ---
    df_raw = pd.read_csv(csv_file)
    n_rows = len(df_raw)

    # --- HANDLE CATEGORICAL VARIABLES (NEW BLOCK) ---
    cat_cols = df_raw.select_dtypes(include=['object', 'category']).columns
    num_cols = df_raw.select_dtypes(include=[np.number]).columns

    if len(cat_cols) > 0:
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_cat = enc.fit_transform(df_raw[cat_cols])
        X_num = df_raw[num_cols].to_numpy()

        df_raw = pd.DataFrame(
            np.hstack([X_num, X_cat]),
            columns=list(num_cols) + list(enc.get_feature_names_out(cat_cols))
        )

    # --- LABEL HANDLING (after encoding) ---
    y_true_series = get_label_series_if_present(df_raw)
    if y_true_series is not None:
        y_true = y_true_series.to_numpy()
        X_df = df_raw.drop(columns=[y_true_series.name])
    else:
        y_true = try_attach_builtin_labels(csv_file, n_rows)
        X_df = df_raw

    # --- Numeric data only ---
    X_df_num = safe_numeric_df(X_df)
    feature_names = list(X_df_num.columns)
    X = X_df_num.to_numpy(dtype=float)

    # --- Standardization ---
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # --- Auto eps ---
    eps = auto_eps(Xs, min_samples=min_samples, k_extra=2)

    # --- Run DBSCAN ---
    db = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = db.fit_predict(Xs)

    unique_labels = sorted([l for l in np.unique(y_pred) if l != -1])
    n_clusters = len(unique_labels)
    noise_rate = float(np.mean(y_pred == -1))

    def safe_metric(fn, default=np.nan):
        try:
            return float(fn(Xs, y_pred))
        except Exception:
            return default

    if n_clusters >= 2 and noise_rate < 1.0:
        silhouette = safe_metric(silhouette_score)
        ch = safe_metric(calinski_harabasz_score)
        dbi = safe_metric(davies_bouldin_score)
    else:
        silhouette = ch = dbi = np.nan

    # --- Surrogate Tree ---
    if holdout_surrogate and len(np.unique(y_pred)) > 1:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=holdout_test_size, random_state=42)
        train_idx, test_idx = next(sss.split(Xs, y_pred))

        clf = train_surrogate_tree(Xs[train_idx], y_pred[train_idx])
        y_sur = clf.predict(Xs[test_idx])

        sur_acc = float(accuracy_score(y_pred[test_idx], y_sur))
        sur_f1m = float(f1_score(y_pred[test_idx], y_sur, average="macro"))
        y_sur_full = clf.predict(Xs)
    else:
        clf = train_surrogate_tree(Xs, y_pred)
        y_sur_full = clf.predict(Xs)
        sur_acc = float(accuracy_score(y_pred, y_sur_full))
        sur_f1m = float(f1_score(y_pred, y_sur_full, average="macro"))

    # --- Export DNF rules ---
    dnf_rules = tree_to_dnf(clf, feature_names)
    dnf_file = os.path.splitext(os.path.basename(csv_file))[0] + out_rules_suffix

    with open(dnf_file, "w", encoding="utf-8") as f:
        f.write("\n".join(dnf_rules))

    # --- PoX ---
    orig_obj = dbscan_density_objective(Xs, y_pred, eps=eps, min_samples=min_samples)
    sur_obj  = dbscan_density_objective(Xs, y_sur_full, eps=eps, min_samples=min_samples)

    pox_abs = float(sur_obj - orig_obj)
    pox_rel = float(pox_abs / (abs(orig_obj) + 1e-9))

    # --- External metrics (ARI/NMI) ---
    if y_true is not None and len(y_true) == len(y_pred):
        try:
            ari = float(adjusted_rand_score(y_true, y_pred))
            nmi = float(normalized_mutual_info_score(y_true, y_pred))
        except Exception:
            ari = np.nan
            nmi = np.nan
    else:
        ari = np.nan
        nmi = np.nan

    diag = structure_diagnostics(Xs, eps=eps, min_samples=min_samples, y_pred=y_pred)

    # --- Final output dictionary ---
    out = {
        "dataset": os.path.splitext(os.path.basename(csv_file))[0],
        "method": "dbscan",
        "n_clusters": n_clusters,
        "noise_rate": noise_rate,
        "silhouette": silhouette,
        "calinski_harabasz": ch,
        "davies_bouldin": dbi,
        "surrogate_acc": sur_acc,
        "surrogate_f1_macro": sur_f1m,
        "surrogate_depth": clf.get_depth(),
        "surrogate_leaves": clf.get_n_leaves(),
        "objective_name": f"dbscan_density(eps={eps:.6f},min_samples={min_samples})",
        "orig_obj": orig_obj,
        "surrogate_obj": sur_obj,
        "price_explainability_abs": pox_abs,
        "price_explainability_rel": pox_rel,
        "ARI": ari,
        "NMI": nmi,
        "eps": eps,
        "min_samples": min_samples,
        "dnf_file": dnf_file,
        "kdist_med": diag["kdist_med"],
        "kdist_iqr": diag["kdist_iqr"],
        "density_contrast": diag["density_contrast"],
        "frac_core": diag["frac_core"],
        "frac_border": diag["frac_border"],
        "frac_noise": diag["frac_noise"],
        "pcs_95var": diag["pcs_95var"],
        "cluster_size_gini": diag["cluster_size_gini"],
    }

    return out

# Cell 7 — Batch runner
def run_all_dbscan(csv_list: List[str],
                   min_samples_map: Optional[Dict[str, int]] = None,
                   holdout_surrogate: bool = False) -> pd.DataFrame:
    rows = []
    for csv in csv_list:
        ms = (min_samples_map or {}).get(os.path.basename(csv), 3)
        res = evaluate_dbscan_for_csv(csv, min_samples=ms, holdout_surrogate=holdout_surrogate)
        rows.append(res)

        print(
            f"→ {res['dataset']}: k={res['n_clusters']}  noise={fmt(res['noise_rate'])}  "
            f"sil={fmt(res['silhouette'])}  ARI={fmt(res['ARI'])}  NMI={fmt(res['NMI'])}  "
            f"eps={fmt(res['eps'],6)}  PoXabs={fmt(res['price_explainability_abs'])}  "
            f"PoXrel={fmt(res['price_explainability_rel'])}"
        )
        print(f"   DNF → {res['dnf_file']} | depth={res['surrogate_depth']} leaves={res['surrogate_leaves']}")

    df = pd.DataFrame(rows)
    df.to_csv("dbscan_metrics.csv", index=False)
    print("\n Saved dbscan_metrics.csv")
    return df

# Cell 8 — Main (adjust paths/min_samples as needed)
if __name__ == "__main__":
    csvs = [
        "iris_clean.csv",
        "wine_clean.csv",
        "Mall_Customers_clean.csv",
        "Wholesale_customers_clean.csv",
    ]
    ms_map = {
        "iris_clean.csv": 3,
        "wine_clean.csv": 3,
        "Mall_Customers_clean.csv": 3,
        "Wholesale_customers_clean.csv": 5,
    }

    df_metrics = run_all_dbscan(csvs, min_samples_map=ms_map, holdout_surrogate=False)

    print("\n=== dbscan_metrics.csv preview ===")
    with pd.option_context('display.max_columns', None, 'display.width', 160):
        print(df_metrics)

# ===============================================
# Patch D — peek at headers to set label_cols correctly (optional)
# ===============================================
import pandas as pd

for fp in ["iris_clean.csv","wine_clean.csv","Mall_Customers_clean.csv","Wholesale_customers_clean.csv"]:
    if os.path.exists(fp):
        df = pd.read_csv(fp, nrows=3)
        print(f"\n{fp} → columns: {list(df.columns)}")

# Cell 9 — Visualize DBSCAN clustering results (PCA 2D) for all datasets
import matplotlib.pyplot as plt

def visualize_dbscan_clusters(csv_file: str, eps: float = None, min_samples: int = 3):
    """
    Visualize DBSCAN clusters (colored) and noise points (grey) using PCA 2D projection.
    """
    df = pd.read_csv(csv_file)
    y_true_series = get_label_series_if_present(df)
    if y_true_series is not None:
        X_df = df.drop(columns=[y_true_series.name])
    else:
        X_df = df
    X_df_num = safe_numeric_df(X_df)
    X = StandardScaler().fit_transform(X_df_num.to_numpy(dtype=float))

    # Auto eps if not provided
    if eps is None:
        eps = auto_eps(X, min_samples=min_samples)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = db.fit_predict(X)

    # PCA to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot
    plt.figure(figsize=(7,6))
    unique_labels = np.unique(y_pred)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for color, label in zip(colors, unique_labels):
        mask = y_pred == label
        if label == -1:
            plt.scatter(X_pca[mask,0], X_pca[mask,1],
                        c="lightgrey", s=40, label="Noise", edgecolors="none")
        else:
            plt.scatter(X_pca[mask,0], X_pca[mask,1],
                        c=[color], s=40, label=f"Cluster {label}", edgecolors="none")

    plt.title(f"DBSCAN Clusters for {os.path.basename(csv_file)}\n(eps={eps:.3f}, min_samples={min_samples})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()


# === Run visualization for all datasets ===
datasets = [
    ("iris_clean.csv", 3),
    ("wine_clean.csv", 3),
    ("Mall_Customers_clean.csv", 3),
    ("Wholesale_customers_clean.csv", 5)
]

for file, ms in datasets:
    print(f"\nVisualizing {file} ...")
    visualize_dbscan_clusters(file, min_samples=ms)
