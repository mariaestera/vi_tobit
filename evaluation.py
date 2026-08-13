import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc



def confusion_stats(sig_true, sig_vi):
    """
    Compute confusion matrix counts (TP, TN, FP, FN) at the variable level.

    Parameters
    ----------
    sig_true : array-like, shape (d,)
        Boolean (or 0/1) mask indicating whether each variable is a true signal.
    sig_vi : array-like, shape (d,)
        Boolean (or 0/1) mask indicating whether each variable was selected by the model.

    Returns
    -------
    TP, TN, FP, FN : int
        True positive, true negative, false positive, and false negative counts.
    """
    sig_true = np.asarray(sig_true).astype(bool)
    sig_vi = np.asarray(sig_vi).astype(bool)

    assert sig_true.shape == sig_vi.shape, "sig_true and sig_vi must have the same shape"

    TP = (sig_true & sig_vi).sum()
    TN = (~sig_true & ~sig_vi).sum()
    FP = (~sig_true & sig_vi).sum()
    FN = (sig_true & ~sig_vi).sum()

    return TP, TN, FP, FN

def compute_stats(TP, TN, FP, FN):
    n = TP + TN + FP + FN
    precision   = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    recall      = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    npv         = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    fdr         = FP / (TP + FP) if (TP + FP) > 0 else np.nan
    fpr         = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    fnr         = FN / (FN + TP) if (FN + TP) > 0 else np.nan
    accuracy    = (TP + TN) / n if n > 0 else np.nan
    f1          = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else np.nan
    mcc_denom   = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc         = (TP*TN - FP*FN) / mcc_denom if mcc_denom > 0 else np.nan
    stats = pd.DataFrame({
        "Metric": ["Precision", "Recall", "Specificity",
                   "NPV", "FDR", "FPR", "FNR", "Accuracy", "F1-score", "MCC"],
        "Value": [precision, recall, specificity, npv, fdr, fpr, fnr, accuracy, f1, mcc]
    })
    stats["Value"] = stats["Value"].round(3)
    confusion = pd.DataFrame(
        [[TP, FN], [FP, TN]],
        index=["Actual: significant", "Actual: non-significant"],
        columns=["Predicted: significant", "Predicted: non-significant"]
    )
    return confusion, stats
    

def block_confusion_stats(sig_true, sig_vi, block_ids):
    """
    Block-level confusion statistics for correlated variable groups.

    Parameters
    ----------
    sig_true : array-like, shape (d,)
        Boolean (or 0/1) mask indicating whether each variable is a true signal.
    sig_vi : array-like, shape (d,)
        Boolean (or 0/1) mask indicating whether each variable was selected by the model.
    block_ids : array-like, shape (d,)
        Block label for each variable (arbitrary labels, not necessarily 0..n_blocks-1;
        blocks may have different sizes and need not be contiguous in the index).

    A block is classified as:
      - true-positive block: contains >=1 true signal AND >=1 selected variable
      - false-negative block: contains >=1 true signal, but NO variable was selected
      - false-positive block: contains no signal, but >=1 variable was selected
      - true-negative block: contains no signal and no variable was selected

    Returns
    -------
    TP, TN, FP, FN, n_blocks
    """
    sig_true = np.asarray(sig_true).astype(bool)
    sig_vi = np.asarray(sig_vi).astype(bool)
    block_ids = np.asarray(block_ids)

    assert sig_true.shape == sig_vi.shape == block_ids.shape, \
        "sig_true, sig_vi and block_ids must have the same shape"

    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    n_blocks = len(unique_blocks)

    # Per-block "any" aggregation, without assuming a fixed block size
    block_has_true = np.zeros(n_blocks, dtype=bool)
    block_has_vi = np.zeros(n_blocks, dtype=bool)

    np.logical_or.at(block_has_true, inverse, sig_true)
    np.logical_or.at(block_has_vi, inverse, sig_vi)

    TP = (block_has_true & block_has_vi).sum()
    TN = ((~block_has_true) & (~block_has_vi)).sum()
    FP = ((~block_has_true) & block_has_vi).sum()
    FN = (block_has_true & (~block_has_vi)).sum()

    return TP, TN, FP, FN, n_blocks


def compute_block_ids(X, tol=None, alpha=0.05):
    """
    Derive block ids (groups of correlated variables) from the empirical
    correlation structure of X, assuming the correlation matrix is
    block-diagonal (i.e. variables are already ordered so that each block
    occupies a contiguous range of column indices).

    Under this assumption, a new block starts whenever two consecutive
    variables (i, i+1) are not significantly correlated, so only the
    off-diagonal band (i, i+1) needs to be checked instead of the full
    pairwise correlation matrix. This is O(d) instead of O(d^2) and avoids
    building a graph / connected-components computation.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Data matrix (n samples, d variables/columns), with variables already
        ordered so that correlated blocks are contiguous.
    tol : float or None, default None
        Threshold on the absolute empirical correlation |corr(i, i+1)| above
        which two consecutive variables are considered part of the same block.
    alpha : float, default 0.05
        Significance level used to compute the default `tol`, when `tol`
        is not provided explicitly. Ignored if `tol` is given.

    Returns
    -------
    block_ids : ndarray, shape (d,)
        Block label (integer, 0-indexed) for each variable/column of X.
    tol_used : float
        The threshold value that was actually used.
    corr : ndarray, shape (d, d)
        The empirical correlation matrix of X (returned for inspection/debugging).
    """
    X = np.asarray(X)
    n, d = X.shape
    corr = np.corrcoef(X, rowvar=False)

    if tol is None:
        df = n - 2
        if df <= 0:
            raise ValueError("Sample size too small to estimate correlation significance (n <= 2).")
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        tol = t_crit / np.sqrt(df + t_crit ** 2)

    # Correlation between consecutive variables (the only off-diagonal
    # entries that matter under the block-diagonal assumption)
    consecutive_corr = np.abs(np.diag(corr, k=1))  # length d-1

    # A new block starts wherever consecutive variables are NOT connected
    block_boundary = consecutive_corr <= tol  # True => break between i and i+1

    block_ids = np.zeros(d, dtype=int)
    block_ids[1:] = np.cumsum(block_boundary)

    return block_ids, tol, corr

def count_misattributed_errors(sig_true, sig_vi, block_ids):
    """
    Count errors due to attributing a true signal to the wrong column
    within the same block: cases where a block correctly contains a
    detected signal (block-level TP), but the specific selected variable(s)
    do not overlap with the true signal variable(s) in that block.

    Parameters
    ----------
    sig_true : array-like, shape (d,)
        Boolean mask of true signal variables.
    sig_vi : array-like, shape (d,)
        Boolean mask of selected variables.
    block_ids : array-like, shape (d,)
        Block label for each variable.

    Returns
    -------
    n_misattributed_blocks : int
        Number of blocks where the signal was detected (block-level TP)
        but assigned to the wrong variable within the block (no overlap
        between true and selected columns in that block).
    n_misattributed_vars : int
        Total number of true-signal variables "missed" due to this kind
        of misattribution.
    pct_of_selected_blocks : float
        Percentage of blocks with at least one selected variable
        (block-level positives) that are misattributed.
    pct_of_selected_columns : float
        Percentage of selected columns (sig_vi == True) that belong to a
        misattributed block.
    """
    sig_true = np.asarray(sig_true).astype(bool)
    sig_vi = np.asarray(sig_vi).astype(bool)
    block_ids = np.asarray(block_ids)

    n_misattributed_blocks = 0
    n_misattributed_vars = 0
    n_selected_blocks = 0  # blocks with at least one selected variable

    for b in np.unique(block_ids):
        idx = block_ids == b
        true_in_block = sig_true[idx]
        vi_in_block = sig_vi[idx]

        block_has_true = true_in_block.any()
        block_has_vi = vi_in_block.any()
        overlap = (true_in_block & vi_in_block).any()

        if block_has_vi:
            n_selected_blocks += 1

        # Block-level TP, but no column-level overlap => wrong variable selected
        if block_has_true and block_has_vi and not overlap:
            n_misattributed_blocks += 1
            n_misattributed_vars += true_in_block.sum()

    n_selected_columns = sig_vi.sum()

    pct_of_selected_blocks = (
        100 * n_misattributed_blocks / n_selected_blocks if n_selected_blocks > 0 else np.nan
    )
    pct_of_selected_columns = (
        100 * n_misattributed_vars / n_selected_columns if n_selected_columns > 0 else np.nan
    )

    return n_misattributed_blocks, n_misattributed_vars, pct_of_selected_blocks, pct_of_selected_columns


def compare_effect_sizes(beta_true, beta_vi, sig_true, sig_vi, label, ax=None, legend = True):

    # Boolean masks (assumed already computed, e.g. sig_true = ..., sig_vi = model.sig_vi(threshold=0.95))
    TP_mask = sig_true & sig_vi
    FN_mask = sig_true & ~sig_vi
    FP_mask = ~sig_true & sig_vi
    TN_mask = ~sig_true & ~sig_vi
    
    lo = min(beta_true.min(), beta_vi.min())-1
    hi = max(beta_true.max(), beta_vi.max())+1
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    # Regression line fitted only on True Positives
    if TP_mask.sum() > 1:
        slope, intercept = np.polyfit(beta_true[TP_mask], beta_vi[TP_mask], 1)
        x_line = np.linspace(lo,hi, 100)
        ax.plot(x_line, slope * x_line + intercept, color="green", linewidth=1, linestyle = "dashed",
                label=f"TP fit (slope={slope:.2f})")
    
    ax.scatter(beta_true[FP_mask], beta_vi[FP_mask], s=10, alpha=0.4, color="red", label="False Positive")
    ax.scatter(beta_true[FN_mask], beta_vi[FN_mask], s=10, alpha=0.4, color="gold", label="False Negative")
    ax.scatter(beta_true[TP_mask], beta_vi[TP_mask], s=10, alpha=0.4, color="green", label="True Positive")
    ax.scatter(beta_true[TN_mask], beta_vi[TN_mask], s=10, alpha=0.4, color="blue", label="True Negative")
    
    
    ax.set_xlabel("True beta")
    ax.set_ylabel(f"Estimated beta ({label})")
    if legend: 
        ax.legend()    
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if ax is None:
        ax.set_title(f"True vs Estimated Coefficients by Detection Outcome using {label}")
        plt.show()
    else:
        return ax



def plot_roc_curve(sig_true, rho, ax=None, label=None):
    """
    Plot the ROC curve based on posterior inclusion probabilities (rho)
    against the true signal indicator.

    Parameters
    ----------
    sig_true : array-like, shape (d,)
        Boolean (or 0/1) mask indicating whether each variable is a true signal.
    rho : array-like, shape (d,)
        Posterior inclusion probability for each variable (model.rho),
        used as the continuous score for the ROC curve.
    ax : matplotlib.axes.Axes or None, default None
        Axis to plot on. If None, a new figure and axis are created.
    label : str or None, default None
        Optional label for the ROC curve (e.g. model name), shown in the legend.
        If None, only the AUC value is shown.

    Returns
    -------
    fpr : ndarray
        False positive rates at each threshold.
    tpr : ndarray
        True positive rates at each threshold.
    roc_auc : float
        Area under the ROC curve.
    """
    sig_true = np.asarray(sig_true).astype(bool)
    rho = np.asarray(rho)

    fpr, tpr, thresholds = roc_curve(sig_true, rho)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    curve_label = f"{label} (AUC = {roc_auc:.3f})" if label else f"AUC = {roc_auc:.3f}"
    ax.plot(fpr, tpr, linewidth=2, label=curve_label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance level")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    return fpr, tpr, roc_auc

def plot_precision_recall_curve(sig_true, rho, ax=None, label=None, target_fdr=None):

    sig_true = np.asarray(sig_true).astype(bool)
    rho = np.asarray(rho)

    precision, recall, thresholds = precision_recall_curve(sig_true, rho)
    pr_auc = auc(recall, precision)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    curve_label = f"{label} (AUC = {pr_auc:.3f})" if label else f"AUC = {pr_auc:.3f}"
    ax.plot(recall, precision, linewidth=2, label=curve_label)

    baseline = sig_true.mean()
    ax.axhline(baseline, linestyle="--", color="gray", linewidth=1,
               label=f"Chance level (baseline = {baseline:.3f})")

    if target_fdr is not None:
        target_precision = 1 - target_fdr
        ax.axhline(target_precision, linestyle=":", color="red", linewidth=1.5,
                   label=f"Target precision (FDR={target_fdr:.2f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (TPR)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="upper right")

    return precision, recall, pr_auc


def viz_gibbs(sig_true, burn_in, model_gibbs, path = None):

    # Pick one "significant" (true signal) and one "non-significant" variable index
    sig_true = np.asarray(sig_true).astype(bool)  # from earlier: beta_true != 0
    idx_sig = np.where(sig_true)[0][0]        # first true-signal variable
    idx_nonsig = np.where(~sig_true)[0][0]    # first null variable
    
    beta_history = np.array(model_gibbs.beta_history)     # shape (n_iter, d)
    gamma_history = np.array(model_gibbs.gamma_history)   # shape (n_iter, d)
    sigma_history = np.array(model_gibbs.sigma_history)    # shape (n_iter,)
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Beta - significant variable
    axes[0].plot(beta_history[:, idx_sig], color="green", linewidth=0.8)
    axes[0].axvline(burn_in, color="red", linestyle="--", label="Burn-in")
    axes[0].set_ylabel(f"beta[{idx_sig}]")
    axes[0].set_title(f"Beta trace — significant variable (index {idx_sig})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Beta - non-significant variable
    axes[1].plot(beta_history[:, idx_nonsig], color="gray", linewidth=0.8)
    axes[1].axvline(burn_in, color="red", linestyle="--", label="Burn-in")
    axes[1].set_ylabel(f"beta[{idx_nonsig}]")
    axes[1].set_title(f"Beta trace — non-significant variable (index {idx_nonsig})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Gamma - both variables
    axes[2].plot(gamma_history[:, idx_sig], color="green", linewidth=0.8, alpha=0.7,
                 label=f"gamma[{idx_sig}] (significant)")
    axes[2].plot(gamma_history[:, idx_nonsig], color="gray", linewidth=0.8, alpha=0.7,
                 label=f"gamma[{idx_nonsig}] (non-significant)")
    axes[2].axvline(burn_in, color="red", linestyle="--", label="Burn-in")
    axes[2].set_ylabel("gamma")
    axes[2].set_title("Gamma trace")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Sigma^2 of the latent variable
    axes[3].plot(sigma_history, color="blue", linewidth=0.8)
    axes[3].axvline(burn_in, color="red", linestyle="--", label="Burn-in")
    axes[3].set_ylabel("sigma^2 (y*)")
    axes[3].set_xlabel("Iteration")
    axes[3].set_title("Sigma^2 trace (latent variable)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()