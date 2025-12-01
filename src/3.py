
#upload all the important modules

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph

from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt

RANDOM_STATE = 42

# Expected dataset filenames (place in working dir or upload to Colab)
DATASETS = {
    "Iris": "iris_clean.csv",
    "Wine": "wine_clean.csv",
    "Mall_Customers": "Mall_Customers_clean.csv",
    "Wholesale_Customers": "Wholesale_customers_clean.csv",
}

AUTO_DETECT_LABELS = True

#code for uploading files
def in_colab() -> bool:
    try:
        import google.colab  # noqa
        return True
    except Exception:
        return False

def prompt_upload_if_missing(filenames: List[str]) -> None:
    missing = [f for f in filenames if not os.path.exists(f)]
    if not missing:
        return
    if in_colab():
        from google.colab import files
        print(" Some CSVs are missing — upload now:")
        for f in missing: print(" -", f)
        files.upload()
    else:
        print(" Missing files:", missing)

def load_csv_with_fallback(fname: str) -> pd.DataFrame:
    if os.path.exists(fname):
        return pd.read_csv(fname)
    if in_colab():
        from google.colab import files
        print(f" '{fname}' not found — please upload it now.")
        files.upload()
        if os.path.exists(fname):
            return pd.read_csv(fname)
    raise FileNotFoundError(f"Could not find '{fname}'.")

#selecting the target variables in the datasets
def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    possible_labels = ["target", "label", "class", "species", "y"]
    for cname in possible_labels:
        if cname in df.columns:
            return cname
    last_col = df.columns[-1]
    series = df[last_col]
    if not pd.api.types.is_numeric_dtype(series) or series.nunique() <= max(10, len(df)//10):
        return last_col
    return None

def maybe_extract_labels(df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    label_col = detect_label_column(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if label_col:
        y = pd.factorize(df[label_col])[0]
        if label_col in numeric_cols:
            numeric_cols.remove(label_col)
    else:
        y = None

    X = df[numeric_cols].to_numpy(dtype=float)
    return X, y, numeric_cols

def standardize(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)

import numpy as np

#creation of the silhouette score
def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    uniq = np.unique(labels)
    if len(uniq) < 2 or any((labels == u).sum() < 2 for u in uniq):
        return None
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return None
#creation of the 2d visualizaion of he datapoints
def quick_viz_2d(X: np.ndarray, labels: np.ndarray, title: str = ""):
    X2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X) if X.shape[1] > 2 else X
    plt.figure(figsize=(6,5))
    for u in np.unique(labels):
        plt.scatter(X2[labels==u,0], X2[labels==u,1], s=22, label=f"cluster {u}")
    plt.title(title); plt.legend(loc="best"); plt.tight_layout(); plt.show()

def smart_eps_grid(Xs: np.ndarray, k: int = 4) -> list:
    nn = NearestNeighbors(n_neighbors=k).fit(Xs)
    dists, _ = nn.kneighbors(Xs)
    kth = np.sort(dists[:, -1])
    qs = np.quantile(kth, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    med = float(np.median(kth))
    widen = [0.9*med, 1.1*med]
    return sorted(set([round(float(x), 3) for x in np.concatenate([qs, widen])]))
#creation of the lables to determine the clustering quality
def evaluate_unsupervised(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    m = {"n_clusters": int(len(np.unique(labels)))}
    sil = safe_silhouette(X, labels)
    m["silhouette"] = None if sil is None else round(sil, 4)
    try: m["calinski_harabasz"] = round(float(calinski_harabasz_score(X, labels)), 3)
    except Exception: m["calinski_harabasz"] = None
    try: m["davies_bouldin"] = round(float(davies_bouldin_score(X, labels)), 3)
    except Exception: m["davies_bouldin"] = None
    return m

from scipy.sparse.csgraph import connected_components

def _sym(W: csr_matrix) -> csr_matrix:
    return W.maximum(W.T).tocsr()

def _heat_from_dist(Gdist: csr_matrix, sigma: float) -> csr_matrix:
    """Global heat kerel for undirected graph."""
    G = Gdist.copy().tocsr()
    G.data = np.exp(-(G.data**2)/(2.0*sigma*sigma + 1e-12))
    return _sym(G)

def _mutual_knn_graph(Xs: np.ndarray, k: int, mode="distance") -> csr_matrix:
    """Edges are formed if both are in each others neighbour lists."""
    G = kneighbors_graph(Xs, n_neighbors=k, mode=mode, include_self=False)
    return _sym(G.minimum(G.T))  # keep only mutual edges

def _local_scaling_affinity(Xs: np.ndarray, k: int) -> csr_matrix:
    """
    Zelnik–Perona local scaling affinity (self-tuning spectral clustering).
    Builds the kNN affinity matrix using local scale parameters and then
    symmetrises it. This version avoids shape mismatches between the data
    array and the nonzero index arrays.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(Xs)
    dists, neigh = nn.kneighbors(Xs)

    # sigma_i = distance to the k-th neighbour (local scale)
    sig = dists[:, -1] + 1e-12

    rows, cols, vals = [], [], []

    # build directed affinity matrix
    for i in range(Xs.shape[0]):
        for j_idx, dist_ij in zip(neigh[i], dists[i]):
            denom = sig[i] * sig[j_idx]
            w_ij = np.exp(-(dist_ij**2) / denom)
            rows.append(i)
            cols.append(j_idx)
            vals.append(w_ij)

    A = csr_matrix((vals, (rows, cols)), shape=(Xs.shape[0], Xs.shape[0]))
    return _sym(A)
  # undirected

def graph_connectivity_stats(W: csr_matrix) -> dict:
    """Connected components summary and giant-component fraction."""
    n_comp, labels = connected_components(W, directed=False)
    # giant component size:
    vals, counts = np.unique(labels, return_counts=True)
    gc = int(counts.max()) if counts.size else 0
    return {
        "n_components": int(n_comp),
        "giant_component_frac": float(gc / W.shape[0] if W.shape[0] else 0.0),
    }

def build_affinity(
    Xs: np.ndarray,
    mode: str,
    eps: float | None = None,
    min_samples: int | None = None,
    sigma: float | None = None,
    ls_k: int | None = None
) -> csr_matrix:

    if mode == "radius":
        if eps is None: raise ValueError("radius mode requires eps.")
        Gdist = radius_neighbors_graph(Xs, radius=eps, mode="distance", include_self=False)
        sig = sigma if sigma else max(1e-6, eps)
        return _heat_from_dist(Gdist, sig)

    if mode == "knn":
        if (min_samples is None) or (min_samples < 1): raise ValueError("knn mode requires min_samples>=1.")
        Gdist = kneighbors_graph(Xs, n_neighbors=min_samples, mode="distance", include_self=False)
        nz = Gdist.data
        med = np.median(nz) if nz.size>0 else 1.0
        sig = sigma if sigma else max(1e-6, med)
        return _heat_from_dist(Gdist, sig)

    if mode == "mutual_knn":
        if (min_samples is None) or (min_samples < 1): raise ValueError("mutual_knn requires min_samples>=1.")
        Gdist = _mutual_knn_graph(Xs, k=min_samples, mode="distance")
        nz = Gdist.data
        med = np.median(nz) if nz.size>0 else 1.0
        sig = sigma if sigma else max(1e-6, med)
        return _heat_from_dist(Gdist, sig)

    if mode == "local_scaling":
        kk = ls_k or min_samples
        if (kk is None) or (kk < 2): raise ValueError("local_scaling requires ls_k>=2 (or min_samples).")
        return _local_scaling_affinity(Xs, k=kk)

    raise ValueError("mode must be one of: 'radius','knn','mutual_knn','local_scaling'.")

def objective_ncut(W: csr_matrix, labels: np.ndarray) -> float:
    """Normalized Cut on affinity W for partition induced by labels. Lower is better."""
    if not issparse(W): W = csr_matrix(W)
    deg = np.array(W.sum(axis=1)).ravel()
    ncut = 0.0
    for c in np.unique(labels):
        idx = (labels == c)
        if idx.sum() == 0:
            continue
        S = np.flatnonzero(idx)
        notS = np.flatnonzero(~idx)
        cut_val = float(W[S][:, notS].sum())
        assoc_S = float(deg[idx].sum()) + 1e-12
        ncut += cut_val / assoc_S
    return ncut

#a robust eigen solver is created to safely find eigenvalues and eigen vectors without errors
import numpy as np
from scipy.sparse import csr_matrix, issparse, eye as speye
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

def _dense_eigh_wrapper(A: np.ndarray, k: int, smallest: bool = True) -> tuple:
    """Dense fallback using numpy.linalg.eigvalsh/eigh (symmetric)."""
    # NOTE: returns the first k eigenpairs in ascending/descending order
    w, V = np.linalg.eigh(A)
    if smallest:
        idx = np.argsort(w)[:k]
    else:
        idx = np.argsort(w)[-k:]
    return w[idx], V[:, idx]

def robust_eigsh_sym(A, k: int, which: str = "SM", maxiter: int = 20000, tol: float = 1e-6, jitter: float = 1e-8):

    # Cast to csr
    if issparse(A):
        A_csr = A.tocsr()
    else:
        A_csr = csr_matrix(A)

    n = A_csr.shape[0]
    k = max(1, min(k, n - 1))  # safety

    # 1) Try ARPACK
    try:
        vals, vecs = eigsh(A_csr, k=k, which=which, maxiter=maxiter, tol=tol)
        return vals, vecs
    except ArpackNoConvergence:
        pass
    except Exception:
        pass

    # 2) Try with diagonal jitter to improve conditioning
    try:
        A_jit = A_csr + jitter * speye(n, format="csr")
        vals, vecs = eigsh(A_jit, k=k, which=which, maxiter=maxiter, tol=tol)
        return vals, vecs
    except ArpackNoConvergence:
        pass
    except Exception:
        pass

    # 3) Dense fallback if dimension is modest; otherwise reduce k and retry
    if n <= 3000:  # adjust threshold if your Colab RAM is tight
        A_dense = A_csr.toarray()
        smallest = (which.upper() == "SM")
        vals, vecs = _dense_eigh_wrapper(A_dense, k=k, smallest=smallest)
        return vals, vecs

    # 4) Last-resort: reduce k progressively and try ARPACK again
    k_try = k
    while k_try >= 1:
        try:
            vals, vecs = eigsh(A_csr, k=k_try, which=which, maxiter=maxiter*2, tol=tol*10)
            # pad with NaNs if reduced
            if k_try < k:
                pad = np.full((n, k - k_try), np.nan)
                vecs = np.concatenate([vecs, pad], axis=1)
                vals = np.concatenate([vals, np.full(k - k_try, np.nan)])
            return vals, vecs
        except Exception:
            k_try -= 1

    raise RuntimeError("robust_eigsh_sym: failed to recover eigenpairs after all fallbacks.")

from sklearn.cluster import KMeans
from scipy.sparse.csgraph import connected_components

def _largest_component_subgraph(W: csr_matrix):
    """Return indices of the giant component and the induced subgraph."""
    n_comp, comp_labels = connected_components(W, directed=False)
    if n_comp <= 1:
        idx = np.arange(W.shape[0])
        return idx, W
    # giant component
    vals, counts = np.unique(comp_labels, return_counts=True)
    giant = vals[np.argmax(counts)]
    idx = np.flatnonzero(comp_labels == giant)
    W_sub = W[idx][:, idx].tocsr()
    return idx, W_sub

def _labels_via_manual_spectral(W: csr_matrix, k: int, use_giant: bool = True) -> np.ndarray:
    #it normalizes graph taking top eigen vectors and running k-measn to form clusters
    if use_giant:
        idx_gc, W_use = _largest_component_subgraph(W)
    else:
        idx_gc, W_use = np.arange(W.shape[0]), W

    # Normalized adjacency
    deg = np.array(W_use.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D_inv_sqrt = csr_matrix((d_inv_sqrt, (np.arange(deg.size), np.arange(deg.size))), shape=W_use.shape)
    S = D_inv_sqrt @ W_use @ D_inv_sqrt

    # top-k eigenvectors of S (largest algebraic)
    # robust solver already imported earlier
    vals, vecs = robust_eigsh_sym(S, k=k, which="LA", maxiter=20000, tol=1e-6)
    U = vecs  # n x k

    # Row-normalize
    U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)

    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels_gc = km.fit_predict(U_norm)

    # Map back to full n
    labels_full = -1 * np.ones(W.shape[0], dtype=int)
    labels_full[idx_gc] = labels_gc
    # If we excluded small components, assign each remaining component to nearest cluster by mode (simple heuristic)
    if (labels_full == -1).any():
        # assign by nearest centroid in U-space if possible; else zero
        centroids = km.cluster_centers_
        #points tha are not connected are put in group 0
        labels_full[labels_full == -1] = 0
    return labels_full

#creation of a decision tree surrogate
def train_surrogate_tree_and_predict(
    X: np.ndarray, labels: np.ndarray,
    max_depth: int | None = None,
    max_leaf_nodes: int = 20,
    min_samples_leaf: int = 4,
    sample_weight: np.ndarray | None = None
):
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE
    )
    tree.fit(X, labels, sample_weight=sample_weight)
    pred = tree.predict(X)
    acc = float((pred == labels).mean())
    rules = export_text(tree, feature_names=[f"x{i}" for i in range(X.shape[1])])
    return pred, acc, rules

def _spectral_embedding_from_W(W: csr_matrix, k: int) -> np.ndarray:
    if not issparse(W): W = csr_matrix(W)
    n = W.shape[0]
    k = max(2, min(k, n - 1))

    deg = np.array(W.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D_inv_sqrt = csr_matrix((d_inv_sqrt, (np.arange(deg.size), np.arange(deg.size))), shape=W.shape)
    L = csr_matrix(np.eye(n)) - D_inv_sqrt @ W @ D_inv_sqrt

    # smallest k+1 eigenpairs of L, drop trivial first
    vals, vecs = robust_eigsh_sym(L, k=min(k + 1, max(2, n - 1)), which="SM", maxiter=20000, tol=1e-6)
    use_cols = min(vecs.shape[1] - 1, k) if vecs.shape[1] > 1 else vecs.shape[1]
    Z = vecs[:, 1:1 + use_cols] if vecs.shape[1] > 1 else vecs
    return Z


def _boundary_weights(W: csr_matrix, labels: np.ndarray) -> np.ndarray:
    n = W.shape[0]
    w = np.zeros(n, dtype=float)
    for c in np.unique(labels):
        S = np.flatnonzero(labels == c)
        notS = np.flatnonzero(labels != c)
        if S.size and notS.size:
            cross = W[S][:, notS].sum(axis=1).A.ravel()
            w[S] += cross
    w = w + 1e-6
    return w / w.mean()

def estimate_k_by_eigengap(W, kmax=10):
    from scipy.sparse import csr_matrix
    from numpy.linalg import eigh

    if not issparse(W):
        W = csr_matrix(W)

    deg = np.array(W.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D_inv_sqrt = csr_matrix((d_inv_sqrt, (np.arange(deg.size), np.arange(deg.size))))
    L = csr_matrix(np.eye(W.shape[0])) - D_inv_sqrt @ W @ D_inv_sqrt

    # smallest kmax+1 eigenvalues
    vals, _ = robust_eigsh_sym(L, k=min(kmax+1, W.shape[0]-1), which="SM")
    vals = np.sort(vals)

    gaps = np.diff(vals[:kmax+1])
    return int(np.argmax(gaps) + 1)

# === SpectralResult + ARPACK hot-fix bundle ===
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix

# Requires these to already exist in your notebook:
# RANDOM_STATE, standardize, build_affinity, graph_connectivity_stats,
# estimate_k_by_eigengap, _labels_via_manual_spectral, evaluate_unsupervised,
# _spectral_embedding_from_W, _boundary_weights, DecisionTreeClassifier, export_text,
# objective_ncut

@dataclass
class SpectralResult:
    labels: np.ndarray
    params: dict
    graph_mode: str
    metrics: dict
    surrogate_rules: str
    surrogate_acc: float
    pox_abs: float
    pox_rel: float
    obj_original: float
    obj_tree: float
    objective_name: str
    conn: dict
    tree_leaves: int
    tree_depth: int

# 1) Always use robust manual NJW path (no sklearn ARPACK)
def fit_spectral_with_graph(
    X: np.ndarray,
    graph_mode: str,        # 'radius' | 'knn' | 'mutual_knn' | 'local_scaling'
    eps: float | None = None,
    min_samples: int | None = None,
    n_clusters: int | None = None,
    sigma: float | None = None,
    ls_k: int | None = None,
    assign_labels: str = "discretize",
    **kwargs
) -> Tuple[SpectralResult, np.ndarray, csr_matrix]:
    Xs = standardize(X)
    W = build_affinity(Xs, mode=graph_mode, eps=eps, min_samples=min_samples, sigma=sigma, ls_k=ls_k)
    conn = graph_connectivity_stats(W)

    # k via robust eigengap
    k = n_clusters if (n_clusters is not None and n_clusters>=2) else estimate_k_by_eigengap(W, kmax=10)

    # Manual NJW on giant component (robust_eigsh + KMeans)
    labels = _labels_via_manual_spectral(W, k=k, use_giant=True)

    metrics = evaluate_unsupervised(Xs, labels)

    # Surrogate tree over spectral embedding (robust eig inside)
    Z = _spectral_embedding_from_W(W, k=int(len(np.unique(labels))))
    w = _boundary_weights(W, labels)
    tree = DecisionTreeClassifier(
        criterion="gini", max_depth=None, max_leaf_nodes=20, min_samples_leaf=4, random_state=RANDOM_STATE
    )
    tree.fit(Z, labels, sample_weight=w)
    pred = tree.predict(Z)
    acc = float((pred == labels).mean())
    rules = export_text(tree, feature_names=[f"x{i}" for i in range(Z.shape[1])])

    obj_original = objective_ncut(W, labels)
    obj_tree     = objective_ncut(W, pred)
    pox_abs = obj_tree - obj_original
    den = obj_original if abs(obj_original) > 1e-6 else np.nan
    pox_rel = pox_abs / den if not np.isnan(den) else np.nan

    result = SpectralResult(
        labels=labels,
        params={"eps":eps, "min_samples":min_samples, "sigma":sigma, "ls_k":ls_k, "n_clusters":k, "assign_labels":assign_labels},
        graph_mode=f"{graph_mode}+manualNJW",
        metrics=metrics,
        surrogate_rules=rules,
        surrogate_acc=acc,
        pox_abs=pox_abs,
        pox_rel=pox_rel,
        obj_original=obj_original,
        obj_tree=obj_tree,
        objective_name="Ncut",
        conn=conn,
        tree_leaves=tree.get_n_leaves(),
        tree_depth=tree.get_depth()
    )
    return result, Xs, W

# 2) Grid search preferring stable graphs (local_scaling → mutual_kNN → kNN → radius)
def spectral_grid_search(X: np.ndarray, eps_values: list, min_samples_values: list):
    best, best_score, best_W, best_Xs = None, -np.inf, None, None

    def score_of(res: SpectralResult) -> float:
        m, c = res.metrics, res.conn
        sil = m.get("silhouette") or -1.0
        ch  = m.get("calinski_harabasz") or 0.0
        db  = m.get("davies_bouldin")
        gcf = c.get("giant_component_frac", 0.0)
        return float(sil + 0.001*ch + (0.001*(1.0/db) if (db is not None and db>0) else 0.0) + 0.2*gcf)

    def try_cfg(**kwargs):
        nonlocal best, best_score, best_W, best_Xs
        res, Xs_tmp, W_tmp = fit_spectral_with_graph(X, **kwargs)
        s = score_of(res)
        if s > best_score:
            best, best_score, best_W, best_Xs = res, s, W_tmp, Xs_tmp

    for ms in min_samples_values:
        try_cfg(graph_mode="local_scaling", ls_k=ms)
    for ms in min_samples_values:
        try_cfg(graph_mode="mutual_knn", min_samples=ms)
    for ms in min_samples_values:
        try_cfg(graph_mode="knn", min_samples=ms)
    for e in eps_values:
        try_cfg(graph_mode="radius", eps=e)

    return best, best_Xs, best_W

def run_spectral_pipeline(name: str, df: pd.DataFrame):
    df = df.drop_duplicates()
    X, y, _ = maybe_extract_labels(df)
    n_samples = X.shape[0]

    if n_samples < 2:
        print(f"Skipping {name}: not enough samples ({n_samples})")
        return None, None

    # Adjust min_samples_values for kNN/local scaling
    min_samples_values_safe = [ms for ms in [3,5,7,10] if ms < n_samples]
    if not min_samples_values_safe:
        min_samples_values_safe = [2]

    eps_values_safe = [0.1, 0.3, 0.5, 0.7]

    # Run grid search over graph configurations
    res, Xs, W = spectral_grid_search(
        X,
        eps_values=eps_values_safe,
        min_samples_values=min_samples_values_safe
    )

    if res is None:
        print(f"No valid clustering produced for {name}")
        return None, None

    # Ensure n_clusters never exceeds n_samples
    if res.params["n_clusters"] > n_samples:
        print(f"Adjusting n_clusters for {name} from {res.params['n_clusters']} -> {n_samples}")
        res.params["n_clusters"] = n_samples
        # Optionally rerun KMeans on the eigenvectors with safe k
        k = n_samples
        U = _spectral_embedding_from_W(W, k=k)
        U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        res.labels = km.fit_predict(U_norm)

    return res, Xs

def _safe_min_samples_grid(n):
    base = [3, 4, 5, 8, 10]
    grid = [m for m in base if m < n]
    return grid if grid else [max(1, n-1)]

def _format_params(res):
    p = dict(res.params or {})
    p["graph_mode"] = res.graph_mode
    return p

def _print_rule_block(rules: str):
    print("\n--- Surrogate Decision Tree (DNF-style rules over spectral embedding) ---")
    print(rules.rstrip() if rules else "(no rules)")

def run_all_and_print(verbose_viz=True):
    prompt_upload_if_missing(list(DATASETS.values()))

    all_metrics = []
    label_frames = []
    rules_outputs = []

    for name, fname in DATASETS.items():
        # --- Load
        try:
            df = load_csv_with_fallback(fname)
        except FileNotFoundError as e:
            print(f"{name}: {e}")
            continue

        # --- Features
        df = df.drop_duplicates()
        X, y, _ = maybe_extract_labels(df)
        n = X.shape[0]
        if n < 2:
            print(f"Skipping {name}: not enough samples ({n})")
            continue

        # --- Candidate grids (data-driven eps, safe min_samples)
        Xs = standardize(X)
        eps_candidates = smart_eps_grid(Xs, k=4)
        # widen eps set slightly
        if len(eps_candidates) > 0:
            med = float(np.median(eps_candidates))
            eps_candidates = sorted(set(
                [round(e, 3) for e in eps_candidates + [0.9*med, 1.1*med]]
            ))
        ms_candidates = _safe_min_samples_grid(n)

        # --- Print header + candidates
        print(f"\n=== {name} (Spectral) ===")
        if eps_candidates:
            print(f"eps candidates (radius): {eps_candidates}")
        else:
            print(f"eps candidates (radius): []")
        print(f"min_samples candidates (kNN): {ms_candidates}")

        # --- Grid search across graph modes
        res, Xs_best, _W = spectral_grid_search(
            X,
            eps_values=eps_candidates if eps_candidates else [0.3, 0.5],
            min_samples_values=ms_candidates
        )
        if res is None:
            print(f"{name}: clustering failed (no valid configuration)")
            continue

        # --- Best params + metrics block
        best_params = _format_params(res)
        print(f"Best params: {best_params}")
        print(f"Internal metrics: {res.metrics}")

        # --- Surrogate fidelity + objectives
        acc = res.surrogate_acc if res.surrogate_acc is not None else float("nan")
        print(f"SURROGATE fidelity (tree vs spectral labels): {acc:.3f}")

        obj_o = res.obj_original
        obj_t = res.obj_tree
        pox_abs = res.pox_abs
        pox_rel = res.pox_rel if res.pox_rel is not None else float("nan")
        print(f"Ncut original: {obj_o:.6f} | tree: {obj_t:.6f} | PoX abs: {pox_abs:.6f} | PoX rel: {pox_rel:.6f}")

        # --- Rules
        _print_rule_block(res.surrogate_rules)

        # --- Optional 2D visualization
        if verbose_viz and len(np.unique(res.labels)) >= 2:
            try:
                quick_viz_2d(Xs_best, res.labels, title=f"Spectral on {name} (PCA)")
            except Exception as viz_err:
                print(f"(viz skipped: {viz_err})")

        # --- Collect rows for outputs
        m = res.metrics or {}
        c = res.conn or {}
        all_metrics.append({
            "dataset": name,
            "graph_mode": res.graph_mode,
            "n_clusters": int(len(np.unique(res.labels))),
            "silhouette": m.get("silhouette"),
            "calinski_harabasz": m.get("calinski_harabasz"),
            "davies_bouldin": m.get("davies_bouldin"),
            "surrogate_acc": round(float(acc), 3) if np.isfinite(acc) else None,
            "objective": res.objective_name,
            "obj_original": obj_o,
            "obj_tree": obj_t,
            "pox_abs": pox_abs,
            "pox_rel": pox_rel,
            "tree_leaves": res.tree_leaves,
            "tree_depth": res.tree_depth,
            "n_components": c.get("n_components"),
            "giant_component_frac": c.get("giant_component_frac"),
            "eps": res.params.get("eps"),
            "min_samples": res.params.get("min_samples"),
            "sigma": res.params.get("sigma"),
            "ls_k": res.params.get("ls_k"),
            "n_clusters_req": res.params.get("n_clusters"),
        })
        label_frames.append(pd.DataFrame({"dataset": name, "label": res.labels}))
        rules_outputs.append((f"{name}_rules_spectral.txt", res.surrogate_rules or ""))

    # --- Save outputs (exact filenames)
    metrics_df = pd.DataFrame(all_metrics)
    labels_df = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame(columns=["dataset","label"])
    metrics_df.to_csv("spectral_metrics_summary.csv", index=False)
    labels_df.to_csv("spectral_labels_all.csv", index=False)

    for fname, txt in rules_outputs:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(txt)

    print("\n Saved outputs:")
    print(" - spectral_metrics_summary.csv")
    print(" - spectral_labels_all.csv")
    print(" - *_rules_spectral.txt (decision tree rules per dataset)")

# Run it:
run_all_and_print(verbose_viz=True)
