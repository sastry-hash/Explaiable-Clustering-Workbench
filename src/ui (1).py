

# Explainable Clustering Workbench (ExKMC + IMM style)
#  — PoX for KMeans + Spectral + DBSCAN in one run
#    Surrogate label alignment, DBSCAN-safe metrics, clear plotting,


!pip install -q gradio pandas numpy scikit-learn matplotlib scipy

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile, os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    accuracy_score, f1_score, confusion_matrix
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)



def align_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tru_vals = np.unique(y_true); pred_vals = np.unique(y_pred)
    tru_to_idx = {v:i for i,v in enumerate(tru_vals)}
    pred_to_idx = {v:j for j,v in enumerate(pred_vals)}
    ti = np.vectorize(tru_to_idx.get)(y_true)
    pj = np.vectorize(pred_to_idx.get)(y_pred)
    C = np.zeros((len(tru_vals), len(pred_vals)), dtype=np.int64)
    np.add.at(C, (ti, pj), 1)
    r, c = linear_sum_assignment(C.max() - C)
    mapping = {pred_vals[cj]: tru_vals[rj] for rj, cj in zip(r, c)}
    return np.array([mapping.get(v, v) for v in y_pred])


# Helper Functions 


def _err_table(msg: str): return pd.DataFrame({"error": [str(msg)]})

def export_rules_sklearn(clf: DecisionTreeClassifier, feature_names):
    t = clf.tree_; rules = []
    def walk(node, path):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            cond = " AND ".join(path) if path else "TRUE"
            lab = int(np.argmax(t.value[node][0])); rules.append(f"IF {cond} THEN cluster = {lab}"); return
        j = t.feature[node]; thr = t.threshold[node]
        name = feature_names[j] if j < len(feature_names) else f"x{j}"
        walk(t.children_left[node],  path + [f"{name} <= {thr:.6g}"])
        walk(t.children_right[node], path + [f"{name} >  {thr:.6g}"])
    walk(0, []); return "\n".join(rules)

def ncut_objective(W: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, int); n = W.shape[0]; total = 0.0
    allmask = np.ones(n, dtype=bool)
    for c in np.unique(labels):
        S = (labels == c)
        if S.sum() == 0 or S.sum() == n: continue
        Sc = ~S; cut = float(W[np.ix_(S, Sc)].sum()); assoc = float(W[np.ix_(S, allmask)].sum())
        if assoc > 0: total += cut / assoc
    return total

def sse_fixed_to_centers(Xs: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    sse = 0.0
    for c in np.unique(labels):
        idx = (labels == c)
        if idx.any():
            R = Xs[idx] - centers[c]; sse += float((R * R).sum())
    return sse

def dbscan_density_objective(Xs, labels, eps, min_samples):
    labels = np.asarray(labels); mask = labels != -1
    if mask.sum() == 0: return 0.0
    Xn = Xs[mask]; Ln = labels[mask]
    nn = NearestNeighbors(radius=max(1e-8, eps)).fit(Xn)
    neigh_ix = nn.radius_neighbors(Xn, return_distance=False)
    total = 0.0; base = max(0, min_samples - 1)
    for i, nbrs in enumerate(neigh_ix):
        same = np.sum(Ln[nbrs] == Ln[i]) - 1
        total += max(0, same - base)
    return float(total)

def safe_internal(Xs, labels):
    labels = np.asarray(labels)
    mask = labels != -1 if (labels == -1).any() else np.ones_like(labels, bool)
    uniq = np.unique(labels[mask])
    if len(uniq) < 2:
        return pd.DataFrame({
            "silhouette":[np.nan],
            "calinski_harabasz":[np.nan],
            "davies_bouldin":[np.nan],
            "n_clusters_detected":[int(len(uniq))],
            "noise_points":[int(np.sum(labels==-1))]
        })
    try: sil = silhouette_score(Xs[mask], labels[mask])
    except: sil = np.nan
    try: ch = calinski_harabasz_score(Xs[mask], labels[mask])
    except: ch = np.nan
    try: db = davies_bouldin_score(Xs[mask], labels[mask])
    except: db = np.nan
    return pd.DataFrame({
        "silhouette":[sil], "calinski_harabasz":[ch], "davies_bouldin":[db],
        "n_clusters_detected":[int(len(uniq))], "noise_points":[int(np.sum(labels==-1))]
    })

def count_leaves_sklearn(clf): return clf.tree_.n_leaves

def confusion_df(y_true, y_pred):
    tvals = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=tvals)
    return pd.DataFrame(cm, index=[f"T{l}" for l in tvals], columns=[f"P{l}" for l in tvals])

def _resolve_leaf_budget(k:int, use_kprime:bool, k_prime:int, force_k_leaves:bool):
    if force_k_leaves and use_kprime:
        use_kprime = True; force_k_leaves = False
    if force_k_leaves: return int(k)
    if use_kprime:     return int(max(int(k), int(k_prime)))
    return None

# --- normalize shapes for safe Gradio display ---
def _safe_df(df):
    if isinstance(df, pd.DataFrame) and not df.empty: return df
    return pd.DataFrame({"note": ["No data"]})

def _safe_fig(fig): return fig if fig else plt.figure()



def fit_tree_with_leaf_budget(
    X, y, budget: int | None,
    *, min_leaf: int = 3, max_depth: int = 20,
    random_state: int = 42,
    score_fn=None
):
    if budget is None:
        return DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, random_state=random_state
        ).fit(X, y)

    base = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_leaf,
        random_state=random_state
    ).fit(X, y)

    path = base.cost_complexity_pruning_path(X, y)
    alphas = np.unique(path.ccp_alphas)

    best = (None, -np.inf, None)
    for a in alphas:
        clf = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_leaf,
            random_state=random_state, ccp_alpha=float(a)
        ).fit(X, y)
        leaves = clf.tree_.n_leaves
        if leaves > budget: continue
        sc = score_fn(clf) if score_fn else clf.score(X, y)
        if sc > best[1]: best = (clf, sc, leaves)
    if best[0] is not None: return best[0]

    return DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_leaf,
        max_leaf_nodes=int(budget), random_state=random_state
    ).fit(X, y)


#  Per-algorithm computation

def compute_algo_block(algo_name, Xs, num_cols, params):
    """
    Returns a dict with:
      metrics_df, pox_df, fid_df, cm_df, rules_text, labels, fig (PCA), out_csv
    """
    try:
        if algo_name == "KMeans":
            k = int(params["k"])
            km = KMeans(n_clusters=k, random_state=GLOBAL_SEED, n_init=10).fit(Xs)
            labels = km.labels_; centers = km.cluster_centers_
            metrics = safe_internal(Xs, labels)

            # Surrogate (budget may be k or k′)
            budget = _resolve_leaf_budget(k, params["use_kprime"], params["k_prime"], params["force_k_leaves"])
            score_fn = lambda clf: -(sse_fixed_to_centers(Xs, clf.predict(Xs), centers) -
                                     sse_fixed_to_centers(Xs, labels, centers))
            clf = fit_tree_with_leaf_budget(Xs, labels, budget, score_fn=score_fn, random_state=GLOBAL_SEED)
            pred_full = clf.predict(Xs)

            # Fidelity with alignment
            y1_aligned = align_labels(labels, pred_full)
            acc = accuracy_score(labels, y1_aligned)
            f1m = f1_score(labels, y1_aligned, average="macro")
            fid = pd.DataFrame({"algorithm":["KMeans"],"surrogate_accuracy":[acc],
                                "surrogate_f1_macro":[f1m],
                                "surrogate_leaves":[count_leaves_sklearn(clf)],
                                "requested_budget":[budget if budget else -1],
                                "surrogate_depth":[clf.get_depth()]})
            cm = confusion_df(labels, y1_aligned)

            # PoX (SSE) – use aligned labels for tree
            sse_orig = sse_fixed_to_centers(Xs, labels, centers)
            sse_tree = sse_fixed_to_centers(Xs, y1_aligned, centers)
            pox = pd.DataFrame({"algorithm":["KMeans"],"objective":["SSE"],
                                "orig":[sse_orig],"tree":[sse_tree],
                                "PoX_rel":[(sse_tree - sse_orig)/max(sse_orig,1e-12)]})

            # Rules
            rules = export_rules_sklearn(clf, num_cols)

            # PCA figure
            pca = PCA(n_components=2, random_state=GLOBAL_SEED); X2 = pca.fit_transform(Xs)
            fig, ax = plt.subplots(figsize=(6,5))
            ax.scatter(X2[:,0], X2[:,1], c=labels, s=25); ax.set_title("KMeans — PCA"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

            return {"metrics_df":add_algo_col(metrics,"KMeans"),
                    "pox_df":pox,"fid_df":fid,"cm_df":cm,"rules_text":rules,
                    "labels":labels,"fig":fig}

        elif algo_name == "Spectral Clustering":
            k = int(params["k"]); gamma = float(params["gamma"])
            sc = SpectralClustering(n_clusters=k, assign_labels="kmeans",
                                    affinity="rbf", gamma=gamma,
                                    random_state=GLOBAL_SEED)
            labels = sc.fit_predict(Xs); W = rbf_kernel(Xs, gamma=gamma)
            metrics = safe_internal(Xs, labels)

            budget = _resolve_leaf_budget(k, params["use_kprime"], params["k_prime"], params["force_k_leaves"])
            score_fn = lambda clf: -(ncut_objective(W, clf.predict(Xs)) - ncut_objective(W, labels))
            clf = fit_tree_with_leaf_budget(Xs, labels, budget, score_fn=score_fn, random_state=GLOBAL_SEED)
            pred_full = clf.predict(Xs)

            # Fidelity (permutation-invariant objective, but align for metrics)
            y1_aligned = align_labels(labels, pred_full)
            acc = accuracy_score(labels, y1_aligned)
            f1m = f1_score(labels, y1_aligned, average="macro")
            fid = pd.DataFrame({"algorithm":["Spectral"],"surrogate_accuracy":[acc],
                                "surrogate_f1_macro":[f1m],
                                "surrogate_leaves":[count_leaves_sklearn(clf)],
                                "requested_budget":[budget if budget else -1],
                                "surrogate_depth":[clf.get_depth()]})
            cm = confusion_df(labels, y1_aligned)

            # PoX (Ncut) – permutation-invariant, no need to align
            ncut_orig = ncut_objective(W, labels)
            ncut_tree = ncut_objective(W, pred_full)
            pox = pd.DataFrame({"algorithm":["Spectral"],"objective":["Ncut"],
                                "orig":[ncut_orig],"tree":[ncut_tree],
                                "PoX_rel":[(ncut_tree - ncut_orig)/max(ncut_orig,1e-12)]})

            rules = export_rules_sklearn(clf, num_cols)

            pca = PCA(n_components=2, random_state=GLOBAL_SEED); X2 = pca.fit_transform(Xs)
            fig, ax = plt.subplots(figsize=(6,5))
            ax.scatter(X2[:,0], X2[:,1], c=labels, s=25); ax.set_title("Spectral — PCA"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

            return {"metrics_df":add_algo_col(metrics,"Spectral"),
                    "pox_df":pox,"fid_df":fid,"cm_df":cm,"rules_text":rules,
                    "labels":labels,"fig":fig}

        else:  # DBSCAN
            eps = float(params["eps"]); min_samples = int(params["min_samples"])
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(Xs)
            labels = db.labels_
            metrics = safe_internal(Xs, labels)

            # Surrogate (no leaf budget for DBSCAN)
            mask = labels != -1
            if mask.sum() >= 2 and len(np.unique(labels[mask])) >= 2:
                score_fn = lambda clf: -(dbscan_density_objective(Xs[mask], clf.predict(Xs[mask]), eps, min_samples) -
                                         dbscan_density_objective(Xs[mask], labels[mask], eps, min_samples))
                clf = fit_tree_with_leaf_budget(Xs[mask], labels[mask], budget=None,
                                                score_fn=score_fn, random_state=GLOBAL_SEED)
                pred_full = clf.predict(Xs)
                mt = mask
                y1_aligned = align_labels(labels[mt], pred_full[mt])
                acc = accuracy_score(labels[mt], y1_aligned)
                f1m = f1_score(labels[mt], y1_aligned, average="macro")
                fid = pd.DataFrame({"algorithm":["DBSCAN"],"surrogate_accuracy":[acc],
                                    "surrogate_f1_macro":[f1m],
                                    "surrogate_leaves":[count_leaves_sklearn(clf)],
                                    "requested_budget":[-1],
                                    "surrogate_depth":[clf.get_depth()]})
                cm = confusion_df(labels[mt], y1_aligned)

                obj_orig = dbscan_density_objective(Xs, labels, eps, min_samples)
                obj_tree = dbscan_density_objective(Xs, pred_full, eps, min_samples)
                pox = pd.DataFrame({"algorithm":["DBSCAN"],"objective":["density"],
                                    "orig":[obj_orig],"tree":[obj_tree],
                                    "PoX_rel":[(obj_tree - obj_orig)/max(abs(obj_orig),1e-12)]})
                rules = export_rules_sklearn(clf, num_cols)
            else:
                # Not enough non-noise structure
                fid = pd.DataFrame({"algorithm":["DBSCAN"],"surrogate_accuracy":[np.nan],
                                    "surrogate_f1_macro":[np.nan],
                                    "surrogate_leaves":[np.nan],
                                    "requested_budget":[-1],
                                    "surrogate_depth":[np.nan]})
                cm = pd.DataFrame({"note":["Insufficient non-noise structure for surrogate"]})
                pox = pd.DataFrame({"algorithm":["DBSCAN"],"objective":["density"],
                                    "orig":[np.nan],"tree":[np.nan],"PoX_rel":[np.nan]})
                rules = "No rules: insufficient non-noise structure."

            pca = PCA(n_components=2, random_state=GLOBAL_SEED); X2 = pca.fit_transform(Xs)
            fig, ax = plt.subplots(figsize=(6,5))
            is_noise = (labels == -1)
            ax.scatter(X2[~is_noise,0], X2[~is_noise,1], c=labels[~is_noise], s=25)
            if is_noise.any(): ax.scatter(X2[is_noise,0], X2[is_noise,1], s=10, marker='x')
            ax.set_title("DBSCAN — PCA (x = noise)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

            return {"metrics_df":add_algo_col(metrics,"DBSCAN"),
                    "pox_df":pox,"fid_df":fid,"cm_df":cm,"rules_text":rules,
                    "labels":labels,"fig":fig}
    except Exception as e:
        # On per-algo failure, return a row with error info
        return {"metrics_df": pd.DataFrame({"note":[f"{algo_name} error: {repr(e)}"]}),
                "pox_df": pd.DataFrame({"algorithm":[algo_name],"objective":[np.nan],"orig":[np.nan],"tree":[np.nan],"PoX_rel":[np.nan]}),
                "fid_df": pd.DataFrame({"algorithm":[algo_name],"surrogate_accuracy":[np.nan],"surrogate_f1_macro":[np.nan],
                                        "surrogate_leaves":[np.nan],"requested_budget":[np.nan],"surrogate_depth":[np.nan]}),
                "cm_df": pd.DataFrame({"note":[f"{algo_name}: no confusion matrix"]}),
                "rules_text": f"{algo_name}: no rules",
                "labels": None, "fig": None}

def add_algo_col(df, algo_name):
    df = df.copy()
    df.insert(0, "algorithm", algo_name)
    return df



def run_clustering(file, algo_selected, k, gamma, eps, min_samples, explain,
                   force_k_leaves, use_kprime, k_prime):

    try:
        if file is None:
            return (_safe_df(_err_table("Upload a CSV file.")),
                    _safe_fig(plt.figure()),
                    "No rules generated",
                    _safe_df(pd.DataFrame()),
                    _safe_df(pd.DataFrame()),
                    _safe_df(pd.DataFrame()),
                    None)

        path = getattr(file, "name", None) or file
        df = pd.read_csv(path)

        # Keep only numeric features (retain names for rules)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return (_safe_df(_err_table("No numeric columns found")),
                    _safe_fig(plt.figure()),
                    "No rules generated",
                    _safe_df(pd.DataFrame()),
                    _safe_df(pd.DataFrame()),
                    _safe_df(pd.DataFrame()),
                    None)

        X = df[num_cols].to_numpy(dtype=float)
        Xs = StandardScaler().fit_transform(X)

        # Shared params for all algos
        params = {"k": int(k), "gamma": float(gamma),
                  "eps": float(eps), "min_samples": int(min_samples),
                  "force_k_leaves": bool(force_k_leaves),
                  "use_kprime": bool(use_kprime), "k_prime": int(k_prime)}

        # Compute blocks for all three
        block_km   = compute_algo_block("KMeans", Xs, num_cols, params)
        block_spec = compute_algo_block("Spectral Clustering", Xs, num_cols, params)
        block_db   = compute_algo_block("DBSCAN", Xs, num_cols, params)

        # Combine tables
        metrics_all  = pd.concat([block_km["metrics_df"], block_spec["metrics_df"], block_db["metrics_df"]],
                                 ignore_index=True, sort=False)
        pox_all      = pd.concat([block_km["pox_df"], block_spec["pox_df"], block_db["pox_df"]],
                                 ignore_index=True, sort=False)
        fid_all      = pd.concat([block_km["fid_df"], block_spec["fid_df"], block_db["fid_df"]],
                                 ignore_index=True, sort=False)

        # Choose which detailed outputs to show based on selection
        sel = algo_selected
        sel_block = block_km if sel == "KMeans" else (block_spec if sel == "Spectral Clustering" else block_db)

        # Export CSV only for selected algorithm
        out_df = df.copy()
        if sel_block["labels"] is not None:
            out_df["cluster"] = sel_block["labels"]
            tmpdir = tempfile.mkdtemp(prefix="clusters_")
            base = sel.replace(" ","_").lower()
            out_csv = os.path.join(tmpdir, f"{base}_clusters.csv")
            out_df.to_csv(out_csv, index=False)
        else:
            out_csv = None

        # If explain is False, still return PoX/metrics (already computed),
        # but rules/CM/plot will be minimal.
        rules_text = sel_block["rules_text"] if explain else "Explainability disabled."
        fig = sel_block["fig"] if sel_block["fig"] is not None else plt.figure()
        cm_df = sel_block["cm_df"] if explain else pd.DataFrame({"note":["Explainability disabled"]})

        return (
            _safe_df(metrics_all),
            _safe_fig(fig),
            rules_text or "No rules generated",
            _safe_df(pox_all),
            _safe_df(fid_all),
            _safe_df(cm_df),
            out_csv
        )

    except Exception as e:
        err = _err_table(repr(e))
        return (_safe_df(err), _safe_fig(plt.figure()), "",
                _safe_df(pd.DataFrame()), _safe_df(pd.DataFrame()), _safe_df(pd.DataFrame()), None)


# Gradio Interface 


with gr.Blocks(title="Explainable Clustering Workbench") as demo:
    gr.Markdown("## Explainable Clustering Workbench (Pruned Surrogate Trees) — PoX for All Algos")

    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
    with gr.Row():
        algo = gr.Radio(["KMeans","Spectral Clustering","DBSCAN"], value="KMeans", label="Inspect algorithm (plot/rules/CM show this)")
        explain = gr.Checkbox(True, label="Explainability (rules + PoX)")

        # Leaf-budget knobs affect KMeans & Spectral only
        force_k_leaves = gr.Checkbox(False, label="Constrain surrogate to ≤ k leaves (IMM)")
        use_kprime = gr.Checkbox(True, label="Constrain surrogate to ≤ k′ leaves (ExKMC)")

    with gr.Row():
        k = gr.Slider(2, 20, value=5, step=1, label="k (KMeans/Spectral)")
        k_prime = gr.Slider(2, 80, value=12, step=1, label="k′ (leaf budget)")
        gamma = gr.Slider(0.1, 5.0, value=1.4, step=0.1, label="Gamma (Spectral)")
        eps = gr.Slider(0.1, 5.0, value=0.5, step=0.1, label="Epsilon (DBSCAN)")
        min_samples = gr.Slider(2, 20, value=2, step=1, label="Min Samples (DBSCAN)")

    run_btn = gr.Button("Run Clustering")

    with gr.Row():
        metrics_output = gr.Dataframe(label="Clustering Metrics (All Algorithms)")
        plot_output = gr.Plot(label="PCA Scatter Plot (Selected Algorithm)")

    gr.Markdown("### Explainable Surrogate Rules (Selected Algorithm)")
    rules_box = gr.Textbox(lines=12, label="Rules", placeholder="Surrogate decision rules")

    with gr.Row():
        pox_output = gr.Dataframe(label="Price of Explainability (PoX) — All Algorithms")
        fidelity_output = gr.Dataframe(label="Surrogate Fidelity — All Algorithms")
    cm_output = gr.Dataframe(label="Confusion Matrix (Original vs Surrogate) — Selected Algorithm")
    download_output = gr.File(label="Download Clustered CSV (Selected Algorithm)")

    run_btn.click(
        fn=run_clustering,
        inputs=[file_input, algo, k, gamma, eps, min_samples, explain,
                force_k_leaves, use_kprime, k_prime],
        outputs=[metrics_output, plot_output, rules_box,
                 pox_output, fidelity_output, cm_output,
                 download_output]
    )

demo.launch(share=True)
