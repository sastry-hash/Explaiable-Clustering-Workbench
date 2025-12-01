
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42 #it is noise to check the robustness of the algorithim

# Cell 1: Upload the csv

from google.colab import files
uploaded = files.upload()

def load_dataset(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], pd.DataFrame]:
    df = pd.read_csv(path)

    # 1) Named label columns (case-insensitive)
    lower_map = {c.lower(): c for c in df.columns}
    for wanted in ["target", "label", "class", "species", "y"]:
        if wanted in lower_map:
            cname = lower_map[wanted]
            y_raw = df[cname].values
            # Factorize to ensure 0..K-1 integer labels even if input is string/int with gaps
            y = pd.factorize(y_raw)[0]

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feat_names = [c for c in num_cols if c != cname]
            X = df[feat_names].to_numpy(dtype=float) if feat_names else np.empty((len(df), 0), dtype=float)
            return X, y, feat_names, df

    # 2) If the last column is categorical or low-cardinality integer, treat as labels
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    last = df.columns[-1]
    if last not in num_cols:
        y = pd.factorize(df[last].values)[0]
        X = df[num_cols].to_numpy(dtype=float)
        return X, y, num_cols, df
    else:
        series = df[last]
        if pd.api.types.is_integer_dtype(series) and series.nunique() <= max(10, len(df)//10):
            # Factorize anyway to normalize labels
            y = pd.factorize(series.values)[0]
            feat_names = [c for c in num_cols if c != last]
            X = df[feat_names].to_numpy(dtype=float) if feat_names else np.empty((len(df), 0), dtype=float)
            return X, y, feat_names, df

    # 3) Default: no labels; only numeric features
    feat_names = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[feat_names].to_numpy(dtype=float)
    return X, None, feat_names, df

#here standardizing error to mean 0 standard deviation is 1
def standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd_safe = np.where(sd == 0, 1.0, sd)

    return (X - mu) / sd_safe
#determines the cluster compactness
def sse_to_centroids(X, centers, labels):
    X = np.asarray(X, float); labels = np.asarray(labels, int)
    sse = 0.0
    for i in np.unique(labels):
        pts = X[labels == i]
        if pts.size:
            sse += float(((pts - centers[i])**2).sum())
    return sse

#determines how good of a split uses to find the nearest centroids
def _pairwise_sq_dists_to_centers(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x2 = (X**2).sum(axis=1, keepdims=True)
    c2 = (centers**2).sum(axis=1, keepdims=True).T
    xc = X @ centers.T
    return x2 + c2 - 2.0 * xc

#construction of a threshold tree

@dataclass
class TTNode:
    is_leaf: bool
    label: Optional[int] = None
    feature: Optional[int] = None
    thresh: Optional[float] = None
    left: Optional["TTNode"] = None
    right: Optional["TTNode"] = None
    n: int = 0
    depth: int = 0

def tree_depth(root: TTNode) -> int:
    if root is None: return 0
    if root.is_leaf: return 1
    return 1 + max(tree_depth(root.left), tree_depth(root.right))

def tree_leaves(root: TTNode) -> int:
    if root is None: return 0
    if root.is_leaf: return 1
    return tree_leaves(root.left) + tree_leaves(root.right)

def predict_tree(root: TTNode, X: np.ndarray) -> np.ndarray:
    def go(node: TTNode, x: np.ndarray) -> int:
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.thresh else node.right
        return node.label
    return np.array([go(root, X[i]) for i in range(X.shape[0])], dtype=int)

def build_threshold_tree(
    X: np.ndarray,
    centers: np.ndarray,
    max_depth: int = 5,
    max_leaves: Optional[int] = None,
    min_leaf: int = 5
) -> TTNode:
    X = np.asarray(X, float)
    n, d = X.shape
    k = centers.shape[0]
    d2_all = _pairwise_sq_dists_to_centers(X, centers)
    nearest_center = d2_all.argmin(axis=1)

    def leaf_sse(idxs: np.ndarray) -> float:
        if idxs.size == 0: return 0.0
        labs = nearest_center[idxs]
        maj = np.bincount(labs, minlength=k).argmax()
        return float(d2_all[idxs, maj].sum())

    leaves_built = 0
    def build(idxs: np.ndarray, depth: int) -> TTNode:
        nonlocal leaves_built
        n_here = idxs.size

        # stop conditions
        if depth >= max_depth or n_here <= min_leaf or (np.unique(nearest_center[idxs]).size == 1):
            labs = nearest_center[idxs] if n_here > 0 else np.array([], int)
            lab = int(np.bincount(labs, minlength=k).argmax()) if n_here > 0 else 0
            leaves_built += 1
            return TTNode(is_leaf=True, label=lab, n=n_here, depth=depth)

        if max_leaves is not None and leaves_built >= max_leaves - 1:
            labs = nearest_center[idxs]
            lab = int(np.bincount(labs, minlength=k).argmax())
            leaves_built += 1
            return TTNode(is_leaf=True, label=lab, n=n_here, depth=depth)

        base_sse = leaf_sse(idxs)
        best_gain = 0.0
        best_feat = None
        best_thr = None
        best_left = best_right = None

        max_thresholds_per_feat = 32
        for j in range(d):
            vals = X[idxs, j]
            uniq = np.unique(vals)
            if uniq.size <= 1: continue
            cuts = (uniq[:-1] + uniq[1:]) * 0.5
            if cuts.size > max_thresholds_per_feat:
                qs = np.linspace(0, 1, max_thresholds_per_feat + 2)[1:-1]
                cuts = np.quantile(uniq, qs)

            for thr in cuts:
                left_mask = vals <= thr
                right_mask = ~left_mask
                if left_mask.sum() < min_leaf or right_mask.sum() < min_leaf:
                    continue
                left_idx = idxs[left_mask]
                right_idx = idxs[right_mask]
                sse_split = leaf_sse(left_idx) + leaf_sse(right_idx)
                gain = base_sse - sse_split
                if gain > best_gain:
                    best_gain = gain
                    best_feat, best_thr = j, float(thr)
                    best_left, best_right = left_idx, right_idx

        if best_feat is None:
            labs = nearest_center[idxs]
            lab = int(np.bincount(labs, minlength=k).argmax())
            leaves_built += 1
            return TTNode(is_leaf=True, label=lab, n=n_here, depth=depth)

        left_node = build(best_left, depth + 1)
        right_node = build(best_right, depth + 1)
        return TTNode(is_leaf=False, feature=best_feat, thresh=best_thr,
                      left=left_node, right=right_node, n=n_here, depth=depth)

    return build(np.arange(n, dtype=int), depth=0)

# Replace the dataclass
@dataclass
class ExplainableKMeansResult:
    k: int
    centers: np.ndarray
    labels_kmeans: np.ndarray
    labels_tree: np.ndarray
    sse_kmeans: float
    sse_tree: float
    pox_abs: float
    pox_rel: float
    depth: int
    leaves: int
    ARI: Optional[float] = None
    NMI: Optional[float] = None
    # NEW:
    root: Optional[Any] = None
    Xs: Optional[np.ndarray] = None  # standardized features

# Replace the function
def explainable_kmeans(
    X: np.ndarray,
    k: int,
    y_true: Optional[np.ndarray] = None,
    max_depth: int = 5,
    max_leaves: Optional[int] = None,
    min_leaf: int = 5,
) -> ExplainableKMeansResult:
    Xs = standardize(X)

        # === Baseline K-Means ===
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_km = km.fit_predict(Xs)
    centers = km.cluster_centers_

    # Compute SSE using FIXED K-Means centers
    sse_km = sse_to_centroids(Xs, centers, labels_km)

    # === Threshold-tree explainable K-Means (Dasgupta-style) ===
    root = build_threshold_tree(
        Xs, centers,
        max_depth=max_depth,
        max_leaves=max_leaves,
        min_leaf=min_leaf
    )
    labels_tree = predict_tree(root, Xs)

  # Again, compute SSE with the SAME fixed centers
    sse_tr = sse_to_centroids(Xs, centers, labels_tree)

    pox_abs = sse_tr - sse_km
    pox_rel = pox_abs / max(sse_km, 1e-12)

    ARI = NMI = None
    if y_true is not None:
        try:
            ARI = float(adjusted_rand_score(y_true, labels_tree))
            NMI = float(normalized_mutual_info_score(y_true, labels_tree))
        except Exception:
            ARI = NMI = None

    return ExplainableKMeansResult(
        k=k,
        centers=centers,
        labels_kmeans=labels_km,
        labels_tree=labels_tree,
        sse_kmeans=round(sse_km, 6),
        sse_tree=round(sse_tr, 6),
        pox_abs=round(pox_abs, 6),
        pox_rel=round(pox_rel, 6),
        depth=tree_depth(root),
        leaves=tree_leaves(root),
        ARI=ARI,
        NMI=NMI,
        root=root,
        Xs=Xs
    )

# Cell 5: Optional — export human-readable rules from the threshold tree

def export_rules(root: TTNode, feature_names: List[str]) -> List[str]:
    """
    Produce simple 'if ... and ... then cluster=c' rules by traversing the tree.
    """
    rules = []

    def walk(node: TTNode, conds: List[str]):
        if node.is_leaf:
            rules.append("IF " + (" AND ".join(conds) if conds else "TRUE")
                         + f" THEN cluster = {node.label}")
            return
        feat = feature_names[node.feature] if feature_names and node.feature < len(feature_names) else f"x{node.feature}"
        walk(node.left,  conds + [f"{feat} <= {node.thresh:.6g}"])
        walk(node.right, conds + [f"{feat} > {node.thresh:.6g}"])

    walk(root, [])
    return rules

def _ground_truth_if_missing(fname: str, y_from_csv):
    lname = os.path.basename(fname).lower()
    try:
        if "iris" in lname:
            from sklearn.datasets import load_iris
            return load_iris().target.astype(int)   # force true labels
        if "wine" in lname:
            from sklearn.datasets import load_wine
            return load_wine().target.astype(int)   # force true labels
    except Exception:
        pass
    return y_from_csv

# 1) Show what Colab actually has
print("Working dir:", os.getcwd())
print("Files in /content:", sorted(os.listdir("."))[:50])

# 2) Helper: case-insensitive resolver (handles .csv/.CSV and minor case diffs)
def resolve_path(expected_name: str) -> str | None:
    exp = expected_name
    exp_base, exp_ext = os.path.splitext(exp)
    candidates = glob.glob("*")
    # prefer exact match first
    if os.path.exists(exp):
        return exp
    # try case-insensitive + .csv/.CSV swap
    for c in candidates:
        base, ext = os.path.splitext(c)
        if base.lower() == exp_base.lower() and ext.lower() in (".csv",):
            return c
    # last resort: strip spaces/underscores and compare
    def norm(s): return re.sub(r"[\s_]+","", s).lower()
    for c in candidates:
        base, ext = os.path.splitext(c)
        if ext.lower() == ".csv" and norm(base) == norm(exp_base):
            return c
    return None

# 3) Expected list (change here if your names differ)
expected = [
    "iris_clean.csv",
    "wine_clean.csv",
    "Mall_Customers_clean.csv",
    "Wholesale_customers_clean.csv",
]

# 4) Resolve all expected names to actual paths
resolved = {name: resolve_path(name) for name in expected}
print("\nResolved paths:")
for k,v in resolved.items():
    print(f"  {k} -> {v}")

missing = [k for k,v in resolved.items() if v is None]
if missing:
    print("\n Missing (not found in /content with any case):", missing)
    print("Tip: Re-run the upload cell and select ALL four CSVs at once, or drag them into the Files pane.")
else:
    print("\n All expected files found (some may differ in case/extension).")

# 5) Run explainable k-means on all found files
# 6) Run explainable k-means on all found files (tuned + fixed) and save combined summary

def choose_k(X, y):
    return len(np.unique(y)) if y is not None else min(8, max(2, int(np.sqrt(X.shape[0]/2))))

def _fit_once(X, y, k, max_depth, min_leaf, max_leaves_or_none):
    """Try one fit; pass max_leaves if supported; return (res, used_budget, used_fallback)"""
    try:
        if max_leaves_or_none is None:
            res = explainable_kmeans(X, k=k, y_true=y, max_depth=max_depth, min_leaf=min_leaf)
            return res, None, True  # fallback path (no max_leaves)
        else:
            res = explainable_kmeans(X, k=k, y_true=y, max_depth=max_depth,
                                     max_leaves=max_leaves_or_none, min_leaf=min_leaf)
            return res, int(max_leaves_or_none), False
    except TypeError:
        # Older signature: retry without max_leaves
        res = explainable_kmeans(X, k=k, y_true=y, max_depth=max_depth, min_leaf=min_leaf)
        return res, None, True

# Helper (local to this cell): supply GT labels for Iris/Wine when CSV lacks them
def _ground_truth_if_missing(fname: str, y_from_csv):
    y_true = y_from_csv
    lname = os.path.basename(fname).lower()
    if y_true is None:
        try:
            if "iris" in lname:
                from sklearn.datasets import load_iris
                y_true = load_iris().target.astype(int)
            elif "wine" in lname:
                from sklearn.datasets import load_wine
                y_true = load_wine().target.astype(int)
        except Exception:
            y_true = None
    return y_true

# Fixed-parameter tree settings (fairness comparison)
FIXED_MAX_DEPTH  = 5
FIXED_MIN_LEAF   = 5
FIXED_MAX_LEAVES = 20

summary_rows = []

for exp_name, path in resolved.items():
    if path is None:
        continue
    try:
        X, y, feat_names, df = load_dataset(path)
        if X.size == 0:
            print(f" {exp_name}: no numeric feature columns; skipping.")
            continue

        # Ensure ground truth available for ARI/NMI if CSV lacks labels
        y_true = _ground_truth_if_missing(path, y)

        k = choose_k(X, y_true if y_true is not None else y)

        # Per-dataset overrides to improve fidelity where needed
        name_l = os.path.basename(path).lower()
        max_depth = 5
        min_leaf = 5
        if "wholesale" in name_l:
            max_depth = 8
            min_leaf = 3

        #  Base leaf budget ~3×k (cap 40)
        base_budget = min(40, max(22, 3 * k))

        # Sweep budgets on harder datasets; single budget on others
        if "wholesale" in name_l:
            budgets = [max(24, base_budget), 30, 36, 40]
        elif "mall" in name_l:
            budgets = [max(27, base_budget), 30]
        else:
            budgets = [base_budget]

        print(f"\n Running {exp_name} with k={k}, max_depth={max_depth}, min_leaf={min_leaf}, budgets={budgets}")

        best = None
        best_key = (np.inf, np.inf)  # (pox_abs, sse_tree) to minimize
        used_fallback = False

        for b in budgets:
            res, used_budget, fell_back = _fit_once(X, y_true, k, max_depth, min_leaf, max_leaves_or_none=b)
            used_fallback = used_fallback or fell_back

            # Protect against missing attrs
            pox_abs = getattr(res, "pox_abs", np.inf)
            sse_tree = getattr(res, "sse_tree", np.inf)

            # Track best by lowest PoX_abs, then lowest sse_tree
            key = (pox_abs, sse_tree)
            if key < best_key:
                best = (res, used_budget)
                best_key = key

            # Log each attempt
            print(f"   • budget={used_budget if used_budget is not None else 'n/a'} "
                  f" PoX_abs={pox_abs:.6f}  SSE_tree={sse_tree:.6f}  leaves={getattr(res,'leaves',-1)}")

        # Use best result (TUNED)
        res_tuned, used_budget = best
        if used_fallback:
            print("    Your explainable_kmeans() ignored max_leaves; "
                  "leaf count is likely controlled by max_depth/min_leaf. Consider adding max_leaves in your impl.")

        print(f"    selected budget={used_budget if used_budget is not None else 'n/a'} "
              f" PoX_abs={res_tuned.pox_abs:.6f}, leaves={res_tuned.leaves}, depth={res_tuned.depth}")

        # Save tuned outputs
        stem = os.path.splitext(os.path.basename(path))[0]
        out_labels = f"{stem}_labels_kmeans_tree.csv"
        np.savetxt(out_labels, res_tuned.labels_tree, fmt="%d", delimiter=",")

        rules = export_rules(res_tuned.root, feat_names)
        rules_path = f"{stem}_rules.txt"
        with open(rules_path, "w") as f:
            f.write("\n".join(rules))

        summary_rows.append({
            "config": "tuned",
            "file": path,
            "k": int(k),
            "sse_original": res_tuned.sse_kmeans,
            "sse_tree": res_tuned.sse_tree,
            "pox_abs": res_tuned.pox_abs,
            "pox_rel": res_tuned.pox_rel,
            "depth": res_tuned.depth,
            "leaves": res_tuned.leaves,
            "ARI": res_tuned.ARI,
            "NMI": res_tuned.NMI,
            "labels_csv": out_labels,
            "rules_txt": rules_path
        })

        print(f" {exp_name}  {path} [Tuned]: k={k}  SSE(orig)={res_tuned.sse_kmeans:.6f}  "
              f"SSE(tree)={res_tuned.sse_tree:.6f}  PoX={res_tuned.pox_abs:.6f} ({res_tuned.pox_rel:.4f})  "
              f"depth={res_tuned.depth}  leaves={res_tuned.leaves}  ARI={res_tuned.ARI}  NMI={res_tuned.NMI}")

        # ---- Fixed-parameter comparison (FAIRNESS) ----
        print(f" Running [Fixed] {exp_name} with k={k}, max_depth={FIXED_MAX_DEPTH}, "
              f"min_leaf={FIXED_MIN_LEAF}, max_leaves={FIXED_MAX_LEAVES}")

        res_fixed, _, _ = _fit_once(X, y_true, k, FIXED_MAX_DEPTH, FIXED_MIN_LEAF, max_leaves_or_none=FIXED_MAX_LEAVES)

        # Save fixed outputs
        fixed_labels = f"{stem}_labels_kmeans_tree_FIXED.csv"
        np.savetxt(fixed_labels, res_fixed.labels_tree, fmt="%d", delimiter=",")
        fixed_rules = f"{stem}_rules_FIXED.txt"
        with open(fixed_rules, "w") as f:
            f.write("\n".join(export_rules(res_fixed.root, feat_names)))

        print(f"[Fixed] {exp_name}: SSE(orig)={res_fixed.sse_kmeans:.6f}  SSE(tree)={res_fixed.sse_tree:.6f}  "
              f"PoX={res_fixed.pox_abs:.6f} ({res_fixed.pox_rel:.4f})  depth={res_fixed.depth}  leaves={res_fixed.leaves}  "
              f"ARI={res_fixed.ARI}  NMI={res_fixed.NMI}")

        summary_rows.append({
            "config": "fixed",
            "file": path,
            "k": int(k),
            "sse_original": res_fixed.sse_kmeans,
            "sse_tree": res_fixed.sse_tree,
            "pox_abs": res_fixed.pox_abs,
            "pox_rel": res_fixed.pox_rel,
            "depth": res_fixed.depth,
            "leaves": res_fixed.leaves,
            "ARI": res_fixed.ARI,
            "NMI": res_fixed.NMI,
            "labels_csv": fixed_labels,
            "rules_txt": fixed_rules
        })

    except Exception as e:
        print(f" {exp_name} → {path}: {e}")

# Save combined summary
if summary_rows:
    summary = pd.DataFrame(summary_rows)
    cols = ["config","file","k","sse_original","sse_tree","pox_abs","pox_rel",
            "depth","leaves","ARI","NMI","labels_csv","rules_txt"]
    # Ensure all columns exist even if some values are missing
    for c in cols:
        if c not in summary.columns:
            summary[c] = None
    summary = summary[cols]
    summary.to_csv("kmeans_explainable_metrics_ALL.csv", index=False)
    print("\nSaved: kmeans_explainable_metrics_ALL.csv")
    try:
        display(summary)
    except Exception:
        print(summary.to_string(index=False))
else:
    print("\nNo datasets processed. Fix missing files above and re-run this cell.")

expected = [
    "iris_clean.csv",
    "wine_clean.csv",
    "Mall_Customers_clean.csv",
    "Wholesale_customers_clean.csv",
]

resolved = {name: resolve_path(name) for name in expected}

print("\nResolved paths:")
for k, v in resolved.items():
    print(f"  {k} -> {v}")

missing = [k for k, v in resolved.items() if v is None]
if missing:
    print("\nMissing:", missing)
    print("Tip: upload all four CSVs or fix names/case.")
else:
    print("\nAll expected files found.")

summary_rows = []

for exp_name, path in resolved.items():
    if path is None:
        print(f"Skipping {exp_name} — file missing.")
        continue

    try:
        # --------------------
        # Load + prepare data
        # --------------------
        X, y, feat_names, df = load_dataset(path)
        if X.size == 0:
            print(f"{exp_name}: no numeric columns — skipped.")
            continue

        y_true = _ground_truth_if_missing(path, y)
        if y_true is not None and len(y_true) != X.shape[0]:
            print(f"WARNING: ignoring y_true — size mismatch.")
            y_true = None

        k = choose_k(X, y_true if y_true is not None else y)
        stem = os.path.splitext(os.path.basename(path))[0]

        print(f"\n========== {exp_name} (k={k}) ==========")

        # ---------------------------
        # 1. Tuned sweep (best model)
        # ---------------------------
        base_budget = int(min(40, max(18, 3 * k)))
        sweep_caps = sorted({k, int(np.ceil(1.5*k)), 2*k, 3*k, 40})
        max_depth_tuned = 6
        min_leaf_tuned  = 5

        print(f" Tuned sweep: depth={max_depth_tuned}, min_leaf={min_leaf_tuned}, caps={sweep_caps}")

        best_result = None
        best_key = (np.inf, np.inf)

        for cap in sweep_caps:
            res, _, _ = _fit_once(X, y_true, k, max_depth_tuned, min_leaf_tuned, max_leaves_or_none=cap)
            key = (float(res.pox_abs), float(res.sse_tree))

            print(f" cap={cap} → PoX_rel={100*res.pox_rel:.2f}%, leaves={res.leaves}")

            if key < best_key:
                best_key = key
                best_result = (res, cap)

        res_tuned, tuned_cap = best_result
        print(f" Tuned pick: cap={tuned_cap}, PoX_rel={100*res_tuned.pox_rel:.2f}%, leaves={res_tuned.leaves}, depth={res_tuned.depth}")

        # Save tuned results
        tuned_labels = f"{stem}_labels_TUNED.csv"
        np.savetxt(tuned_labels, res_tuned.labels_tree.reshape(-1,1), fmt="%d", delimiter=",")
        tuned_rules = f"{stem}_rules_TUNED.txt"
        with open(tuned_rules, "w") as f:
            f.write("\n".join(export_rules(res_tuned.root, feat_names)))

        summary_rows.append({
            "config": "tuned",
            "file": path,
            "k": k,
            "sse_original": float(res_tuned.sse_kmeans),
            "sse_tree": float(res_tuned.sse_tree),
            "pox_abs": float(res_tuned.pox_abs),
            "pox_rel": float(res_tuned.pox_rel),
            "depth": res_tuned.depth,
            "leaves": res_tuned.leaves,
            "ARI": res_tuned.ARI,
            "NMI": res_tuned.NMI,
            "labels_csv": tuned_labels,
            "rules_txt": tuned_rules
        })

        # ---------------------------
        # 2. FIXED MODEL (required)
        # ---------------------------
        print(f"\n Running FIXED config: {FIXED_PARAMS}")
        res_fixed, _, _ = _fit_once(
            X, y_true, k,
            FIXED_PARAMS["max_depth"],
            FIXED_PARAMS["min_leaf"],
            FIXED_PARAMS["max_leaves"]
        )

        print(f"[FIXED] PoX_rel={100*res_fixed.pox_rel:.2f}%, SSE(orig)={res_fixed.sse_kmeans:.2f}, "
              f"SSE(tree)={res_fixed.sse_tree:.2f}, depth={res_fixed.depth}, leaves={res_fixed.leaves}")

        fixed_labels = f"{stem}_labels_FIXED.csv"
        np.savetxt(fixed_labels, res_fixed.labels_tree.reshape(-1,1), fmt="%d", delimiter=",")

        fixed_rules = f"{stem}_rules_FIXED.txt"
        with open(fixed_rules, "w") as f:
            f.write("\n".join(export_rules(res_fixed.root, feat_names)))

        summary_rows.append({
            "config": "fixed",
            "file": path,
            "k": k,
            "sse_original": float(res_fixed.sse_kmeans),
            "sse_tree": float(res_fixed.sse_tree),
            "pox_abs": float(res_fixed.pox_abs),
            "pox_rel": float(res_fixed.pox_rel),
            "depth": res_fixed.depth,
            "leaves": res_fixed.leaves,
            "ARI": res_fixed.ARI,
            "NMI": res_fixed.NMI,
            "labels_csv": fixed_labels,
            "rules_txt": fixed_rules
        })

    except Exception as e:
        print(f"ERROR in {exp_name}: {e}")