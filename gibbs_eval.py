import numpy as np
import argparse
from scipy.stats import truncnorm, invgamma
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from tqdm.auto import tqdm
from scipy.special import expit, logit
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
from scipy.stats import linregress
from sklearn.metrics import matthews_corrcoef, r2_score

def cond_hdi(beta_samples, gamma_samples, prob=0.95):

    n_iter, d = beta_samples.shape

    cond_hdi = np.full((d, 2), np.nan)

    for j in range(d):
        mask = gamma_samples[:, j] == 1
        if mask.sum() > 1:  # potrzeba min. kilku próbek, żeby HDI miało sens
            cond_hdi[j] = az.hdi(beta_samples[mask, j], prob=prob)

    return cond_hdi

def hdi(samples, prob=0.95):
    samples = np.asarray(samples)
    if samples.ndim == 1:
        return az.hdi(samples, prob=prob)
    n_iter, d = samples.shape
    result = np.empty((d, 2))
    for j in range(d):
        result[j] = az.hdi(samples[:, j], prob=prob)
    return result

def confusion(true_sig, pip, pip_tr,
              beta_true, beta_cond_mean,
              X_train, y_latent_train,
              X_test, y_latent_test):

    true_sig = np.asarray(true_sig).astype(bool)
    pip = np.asarray(pip)
    pip_tr = np.asarray(pip_tr)
    beta_true = np.asarray(beta_true)
    beta_cond_mean = np.asarray(beta_cond_mean)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_latent_train = np.asarray(y_latent_train)
    y_latent_test = np.asarray(y_latent_test)

    rows = []

    for tr in pip_tr:
        pred_sig = pip >= tr

        TP = int(np.sum(pred_sig & true_sig))
        TN = int(np.sum(~pred_sig & ~true_sig))
        FP = int(np.sum(pred_sig & ~true_sig))
        FN = int(np.sum(~pred_sig & true_sig))

        # --- klasyfikacyjne ---
        precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        accuracy = (TP + TN) / len(true_sig)

        if len(np.unique(pred_sig)) > 1 and len(np.unique(true_sig)) > 1:
            mcc = matthews_corrcoef(true_sig, pred_sig)
        else:
            mcc = np.nan

        active_idx = np.where(pred_sig)[0]
        if len(active_idx) >= 2:
            bt = beta_true[active_idx]
            bc = beta_cond_mean[active_idx]

            rmse_beta = np.sqrt(np.mean((bt - bc) ** 2))
            r2_beta = r2_score(bt, bc)
            slope_beta = linregress(bt, bc).slope
        elif len(active_idx) == 1:
            bt = beta_true[active_idx]
            bc = beta_cond_mean[active_idx]
            rmse_beta = np.sqrt(np.mean((bt - bc) ** 2))
            r2_beta = np.nan
            slope_beta = np.nan
        else:
            rmse_beta = np.nan
            r2_beta = np.nan
            slope_beta = np.nan

        beta_masked = np.where(pred_sig, beta_cond_mean, 0.0)

        y_pred_train = X_train @ beta_masked
        y_pred_test = X_test @ beta_masked if X_test else np.nan

        r2_train = r2_score(y_latent_train, y_pred_train) 
        r2_test = r2_score(y_latent_test, y_pred_test) if X_test else np.nan

        rows.append({
            "pip_tr": tr,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "MCC": mcc,
            "RMSE_beta_active": rmse_beta,
            "R2_beta_active": r2_beta,
            "slope_beta_active": slope_beta,
            "R2_train": r2_train,
            "R2_test": r2_test,
        })

    return pd.DataFrame(rows)


    
def _post_burn(chain, burn_in, n_iter):
    """
    Returns the post-burn-in portion of a saved chain, regardless of
    whether the full chain (with burn-in) or only the post-burn-in
    samples were saved to disk.
    """
    if chain.shape[0] == n_iter + burn_in:
        return chain[burn_in:]
    return chain  # already contains only post burn-in samples


def viz_beta(beta_samples, gamma_samples, beta_true, beta_cond_mean, pip,
             burn_in, n_iter, output_folder, n_show=5):
    """
    1) Histograms of up to n_show truly significant betas (beta_true != 0),
       with the conditional posterior mean and the true value as vertical lines.
    2) Histograms of up to n_show truly nonsignificant betas (beta_true == 0),
       annotated with PIP.
    """
    beta_post = _post_burn(beta_samples, burn_in, n_iter)

    sig_idx = np.where(beta_true != 0)[0]
    nonsig_idx = np.where(beta_true == 0)[0]

    # ---- 1) significant betas ----
    n_sig = min(n_show, sig_idx.size)
    if n_sig > 0:
        sel = sig_idx[:n_sig]
        fig, axes = plt.subplots(1, n_sig, figsize=(4 * n_sig, 4), squeeze=False)
        axes = axes[0]
        for k, j in enumerate(sel):
            ax = axes[k]
            ax.hist(beta_post[:, j], bins=40, color="steelblue", alpha=0.7, density=True)
            ax.axvline(beta_cond_mean[j], color="darkorange", linestyle="--",
                       linewidth=2, label="posterior cond. mean")
            ax.axvline(beta_true[j], color="black", linestyle="-",
                       linewidth=2, label="true value")
            ax.set_title(f"beta_{j} (PIP={pip[j]:.2f})")
        axes[-1].legend()  # legend only on the last panel
        fig.suptitle("True significant betas")
        fig.tight_layout()
        fig.savefig(f"{output_folder}/beta_significant.png", dpi=150)
        plt.close(fig)
    else:
        print("No truly significant variables in beta_true - skipping plot.")

    # ---- 2) nonsignificant betas ----
    n_nonsig = min(n_show, nonsig_idx.size)
    if n_nonsig > 0:
        sel = nonsig_idx[:n_nonsig]
        fig, axes = plt.subplots(1, n_nonsig, figsize=(4 * n_nonsig, 4), squeeze=False)
        axes = axes[0]
        for k, j in enumerate(sel):
            ax = axes[k]
            ax.hist(beta_post[:, j], bins=40, color="firebrick", alpha=0.7, density=True)
            ax.axvline(0.0, color="black", linewidth=1)
            ax.set_title(f"beta_{j} (PIP={pip[j]:.2f})")
        fig.suptitle("True nonsignificant betas")
        fig.tight_layout()
        fig.savefig(f"{output_folder}/beta_nonsignificant.png", dpi=150)
        plt.close(fig)
    else:
        print("No truly nonsignificant variables in beta_true - skipping plot.")


def viz_scalars(sigma_samples, pi0_samples, 
                 sigma_mean, sigma_hdi, pi0_mean, pi0_hdi,
                 burn_in, n_iter, output_folder):
    """
    3) Histograms of the scalar parameters (sigma, pi0, g) with posterior
       mean and 95% HDI marked.
    """
    params = [
        ("sigma", sigma_samples, sigma_mean, sigma_hdi, "seagreen"),
        ("pi0", pi0_samples, pi0_mean, pi0_hdi, "mediumpurple"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, samples, mean_val, hdi_val, color) in zip(axes, params):
        post = _post_burn(samples, burn_in, n_iter)
        ax.hist(post, bins=40, color=color, alpha=0.7, density=True)
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=2, label="mean")
        ax.axvspan(hdi_val[0], hdi_val[1], color="gray", alpha=0.25, label="95% HDI")
        ax.set_title(name)
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(f"{output_folder}/scalar_posteriors.png", dpi=150)
    plt.close(fig)


def viz_gibbs(sigma_samples, pi0_samples, gamma_samples,
              burn_in, n_iter, output_folder, gamma_idx=None, rng=None):
    """
    4) Trace plots (including burn-in, if available) for sigma, pi0, g,
       and 4 example gamma_j chains. Vertical line marks the burn-in
       cutoff, horizontal line marks the post-burn-in mean.
    """
    has_burn_in = sigma_samples.shape[0] == n_iter + burn_in

    if gamma_idx is None:
        d = gamma_samples.shape[1]
        n_pick = min(4, d)
        if rng is not None:
            gamma_idx = rng.choice(d, size=n_pick, replace=False)
        else:
            gamma_idx = np.arange(n_pick)

    series = [("sigma", sigma_samples), ("pi0", pi0_samples)]
    series += [(f"gamma_{j}", gamma_samples[:, j]) for j in gamma_idx]

    fig, axes = plt.subplots(len(series), 1, figsize=(8, 2.2 * len(series)), sharex=True)
    for ax, (name, chain) in zip(axes, series):
        ax.plot(chain, color="steelblue", linewidth=0.8)
        post = _post_burn(chain, burn_in, n_iter)
        post_mean = post.mean()
        if has_burn_in:
            ax.axvline(burn_in, color="red", linestyle="--", linewidth=1, label="burn-in cutoff")
        ax.axhline(post_mean, color="black", linestyle=":", linewidth=1, label="post burn-in mean")
        ax.set_ylabel(name)

    if not has_burn_in:
        fig.suptitle("Gibbs sampler trace plots (burn-in not saved)")
    else:
        fig.suptitle("Gibbs sampler trace plots")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(f"{output_folder}/gibbs_traces.png", dpi=150)
    plt.close(fig)

def compute_ess(samples):
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]          # (1, n_samples)
        return az.ess(samples, chain_axis=0, draw_axis=1)
    # samples: (n_samples, d) -> dodaj wymiar chain
    samples = samples[np.newaxis, :, :]            # (1, n_samples, d)
    return az.ess(samples, chain_axis=0, draw_axis=1)



def gibbs_eval(model, args, total_time,n,d):

    input_folder = args.input_folder
    output_folder = args.output_folder
    burn_in = args.burn_in
    n_iter = args.n_iter

    beta_samples, gamma_samples, sigma_samples = np.array(model.beta_history), np.array(model.gamma_history), np.array(model.sigma_history)
    l, u,  sigma_y_true = list(np.load(f"{input_folder}/l_u_sigma.npy"))

    if args.save_chain:
        np.save(f"{output_folder}/beta_samples.npy", beta_samples)
        np.save(f"{output_folder}/sigma_samples.npy", sigma_samples)
        np.save(f"{output_folder}/gamma_samples.npy", gamma_samples)
        np.save(f"{output_folder}/pi0_samples.npy", pi0_samples)

    #---- estimates -----------------------
    gamma_b = gamma_samples[burn_in:]
    pip = np.mean(gamma_b, axis=0)
    beta_b = beta_samples[burn_in:]
    
    beta_mean = np.mean(beta_b, axis=0)
    beta_std = np.std(beta_b, axis=0)
    beta_cond_hdi = cond_hdi(beta_b, gamma_b, prob = args.hdi)

    sigma_mean = sigma_samples[burn_in:].mean()
    sigma_std = sigma_samples[burn_in:].std()
    sigma_hdi = hdi(sigma_samples[burn_in:], prob = args.hdi)
    
    #pi0 = pi0_samples[burn_in:].mean()
    #pi0_std = pi0_samples[burn_in:].std()
    #pi0_hdi = hdi(pi0_samples[burn_in:], prob = args.hdi)

    sigma_ess = compute_ess(sigma_samples[burn_in:])
    #pi0_ess = compute_ess(pi0_samples[burn_in:])
    beta_ess = compute_ess(beta_b)       # (d,) 
    gamma_ess = compute_ess(gamma_b.astype(float))  # (d,)

    
    #computing conditional beta stats
    n_included = gamma_b.sum(axis=0)   

    masked = np.where(gamma_b == 1, beta_b, np.nan)
    with np.errstate(invalid="ignore"):
        beta_cond_mean = np.nanmean(masked, axis=0)
        beta_cond_std = np.nanstd(masked, axis=0)
    beta_cond_mean[n_included == 0] = np.nan
    beta_cond_std[n_included <= 1] = np.nan

    beta_true = np.load(f"{input_folder}/beta.npy")
    beta_stats_df = pd.DataFrame({
        "var_idx": np.arange(d),
        "pip": pip,
        "true_sig": (beta_true != 0).astype(bool),
        "true_size": beta_true,
        "beta_marg_mean": beta_mean,
        "beta_marg_std": beta_std,
        "beta_cond_mean": beta_cond_mean,
        "beta_cond_std": beta_cond_std,
        "beta_cond_hdi_low": beta_cond_hdi[:, 0],
        "beta_cond_hdi_high": beta_cond_hdi[:, 1],
        "ess": beta_ess,
    })
    
    beta_stats_df.to_csv(f"{output_folder}/gibbs_beta.csv", index=False)

    with open(f"{output_folder}/gibbs_scalars.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "true_value", "mean", "std", "hdi_low", "hdi_high", "ess"])
        writer.writerow(["sigma", sigma_y_true, sigma_mean, sigma_std, sigma_hdi[0], sigma_hdi[1], sigma_ess])
        #writer.writerow(["pi0", pi0, pi0_std, pi0_hdi[0], pi0_hdi[1], pi0_ess])


    # --- stats -------------------------------------------------------------------
    pip_tr = [0.1, 0.5, 0.9, 0.95]

    X_train, y_latent_train = np.load(f"{input_folder}/X.npy"), np.load(f"{input_folder}/y_latent.npy")
    
    try:
        X_test, y_latent_test = np.load(f"{input_folder}/test/X.npy"), np.load(f"{input_folder}/test/y_latent.npy")
    except:
        X_test, y_latent_test = None, None

    true_sig = (beta_true !=0).astype(int)
    df = confusion(true_sig, pip, pip_tr,
               beta_true, beta_cond_mean,
               X_train, y_latent_train,
               X_test, y_latent_test)
    df.to_csv(f"{output_folder}/gibbs_stats.csv")
    
    # ---- time --------------------------------------------------------------------
    row = {
        "n": n,
        "d": d,
        "mean_sparsity": pip.mean(),
        "total_time": total_time,
        "total_fit_time": model.total_fit_time,
        "n_iters_with_burn_in": n_iter + burn_in,
        "ystar_fit_time": model.ystar_fit_time,
        "gamma_fit_time": model.gamma_fit_time,
        "gamma_batch_size": args.gamma_batch,
        "beta_fit_time": model.beta_fit_time,
        "seed": args.seed,
    }
    
    pd.DataFrame([row]).to_csv(f"{output_folder}/gibbs_time.csv", index=False)
    

    # --- visualization ----
    if args.viz:    
        pi0_samples = np.zeros(n_iter + burn_in)
        viz_beta(beta_samples, gamma_samples, beta_true, beta_cond_mean, pip,
                 burn_in, n_iter, output_folder)
    
        #tymczasowo: 
        pi0_hdi = [0,1]
    
        viz_scalars(sigma_samples, pi0_samples,
                    sigma_mean, sigma_hdi, args.pi0, pi0_hdi,
                    burn_in, n_iter, output_folder)

        viz_gibbs(sigma_samples, pi0_samples, gamma_samples,
                      burn_in, n_iter, output_folder)
    