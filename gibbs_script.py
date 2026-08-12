import numpy as np
import argparse
from scipy.stats import truncnorm, invgamma
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from tqdm.auto import tqdm
from scipy.special import expit, logit
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate effect sizes using Gibbs sampler")
    # input and output
    parser.add_argument("-input_folder", type=str, required=True, help="folder with X_design, y_latent, l, u, sigma_y_true")
    parser.add_argument("-output_folder", type=str, required=True, help="folder with estimates")

    # gibbs sampler params
    parser.add_argument("-n_iter", type=int, required=True, help="number of sampler iterations after burn-in")
    parser.add_argument("-burn_in", type=int, required=True, help="number of burn-in iterations")
    parser.add_argument("-gamma_bath", type=int, required=False, default=10, help="number of the gamma_i sampled together")

    # initialization
    parser.add_argument("--tau2", type=float, required=False, default=100, help="Initial prior variance")
    parser.add_argument("--pi0", type=float, required=False, default=0.1, help="Initial prior probability of inclusion")
    parser.add_argument("--eps", type=float, required=False, default=0.01, help="sigma^2 ~ InvGamma(eps, eps)")

    #stats
    parser.add_argument("--hdi", type=float, required=False, default=0.95, help = "Level of HDI")
    parser.add_argument("--viz", type=bool, required=False, default = False, help = "Generate histograms of scalar params and example betas, and chains for them")

    # other settings
    parser.add_argument("--seed", type=int, required=False, default=None, help = "random seed")
    parser.add_argument("--save_chain", type=bool, required=False, default=False, help = "saving chain for diagnostics: True/False")

    args = parser.parse_args()
    return args


def data_aug(X, y, beta, l, u, sigma, mask_l, mask_u, rng):
    if (beta != 0).sum() > 0:
        nonzero_idx = np.nonzero(beta)[0]
        mu = X[:, nonzero_idx] @ beta[nonzero_idx]
    else:
        mu = np.zeros(X.shape[0])

    # Lower-censored: y*_i <= l
    if mask_l.any():
        a = -np.inf
        b = (l - mu[mask_l]) / sigma
        y[mask_l] = truncnorm.rvs(
            a, b, loc=mu[mask_l], scale=sigma, random_state=rng
        )

    # Upper-censored: y*_i >= u
    if mask_u.any():
        a = (u - mu[mask_u]) / sigma
        b = np.inf
        y[mask_u] = truncnorm.rvs(
            a, b, loc=mu[mask_u], scale=sigma, random_state=rng
        )

    return y


# Numerical floors to keep log() / division well-defined under rounding error
_EPS_DEN = 1e-10
_EPS_LOG = 1e-12


def sample_gamma_batch(X, XtX, y, g, pi0, n, batch_idx, base_idx):
    
    SSy = y @ y
    k_base = base_idx.size
    Xbatch = X[:, batch_idx]

    if k_base > 0:
        Xb = X[:, base_idx]
        A = XtX[np.ix_(base_idx, base_idx)]
        L, lower = cho_factor(A, lower=True)
        L = np.tril(L)  # cho_factor leaves the other triangle undefined

        c_base = Xb.T @ y
        w_base = cho_solve((L, lower), c_base)
        q_base = c_base @ w_base

        cross = Xb.T @ Xbatch                             # (k_base, m)
        V = solve_triangular(L, cross, lower=True)         # (k_base, m)

        diag_batch = np.einsum("ij,ij->j", Xbatch, Xbatch)     # x_j^T x_j, (m,)
        d = diag_batch - np.einsum("ij,ij->j", V, V)            # Schur complement, (m,)
        d = np.maximum(d, _EPS_DEN)

        cty_batch = Xbatch.T @ y                            # x_j^T y, (m,)
        num = cty_batch - V.T @ w_base                      # (m,)
        q_add = q_base + num**2 / d
    else:
        diag_batch = np.einsum("ij,ij->j", Xbatch, Xbatch)
        diag_batch = np.maximum(diag_batch, _EPS_DEN)
        cty_batch = Xbatch.T @ y
        q_base = 0.0
        q_add = cty_batch**2 / diag_batch

    term = g / (1.0 + g)
    inside1 = np.maximum(SSy - term * q_add, _EPS_LOG)
    inside0 = np.maximum(SSy - term * q_base, _EPS_LOG)

    log_val1 = -0.5 * (k_base + 1) * np.log1p(g) - 0.5 * n * np.log(inside1)
    log_val0 = -0.5 * k_base * np.log1p(g) - 0.5 * n * np.log(inside0)

    log_odds = log_val1 - log_val0 + logit(pi0)
    rho = expit(log_odds)
    return rho




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
        # dodaj wymiar "chain" = 1
        samples = samples[np.newaxis, :]
        return az.ess(samples, chain_axis=0, draw_axis=1)
    return az.ess(samples)

def main():

    args = parse_args()

    # loading data
    input_folder = args.input_folder

    X = np.load(f"{input_folder}/X.npy")
    y = np.load(f"{input_folder}/y_latent.npy")
    l, u = list(np.load(f"{input_folder}/l_u_sigma.npy"))[:2]

    y = np.clip(y, l, u)

    # initialize
    rng = np.random.default_rng(args.seed)

    n, d = X.shape
    n_iter, burn_in = args.n_iter, args.burn_in

    mask_l = (y == l)
    mask_u = (y == u)
    mask_mid = ~mask_l & ~mask_u

    gamma = (rng.random(d) < args.pi0).astype(int)

    beta = np.zeros(d)
    active = gamma == 1
    beta[active] = rng.normal(0, np.sqrt(args.tau2), active.sum())

    sigma2 = np.var(y)
    sigma = np.sqrt(sigma2)

    pi0 = args.pi0
    logit_pi0 = np.log(pi0 / (1 - pi0))
    tau2 = args.tau2
    
    n_saved_iter = n_iter + burn_in

    beta_samples = np.zeros((n_saved_iter, d))
    sigma_samples = np.zeros(n_saved_iter)
    gamma_samples = np.zeros((n_saved_iter, d), dtype=int)
    pi0_samples = np.zeros(n_saved_iter)

    XtX_diag = (X ** 2).sum(axis=0)   # x_j^T x_j for all j, precomputed once
    
    sample_idx = 0
    pbar = tqdm(range(n_iter + burn_in), desc="Gibbs")
    for it in pbar:
        # ---- 1. Data augmentation for censored latent outcomes y*_i ----
        ystar = data_aug(X, y, beta, l, u, sigma, mask_l, mask_u, rng)

        # ---- 2. Batched update of inclusion indicators gamma (k << d) ----
        eta_full = X[:, active] @ beta[active]

        order = rng.permutation(d)
        for start in range(0, d, args.gamma_bath):
            batch_idx = order[start:start + args.gamma_bath]

            Xb = X[:, batch_idx]
            beta_b = beta[batch_idx]
            gamma_b = gamma[batch_idx]

            r_base = ystar - eta_full
            linear_term = (beta_b / sigma2) * (r_base @ Xb)

            active_in_batch = gamma_b == 1
            if active_in_batch.any():
                idx_active = batch_idx[active_in_batch]
                beta_a = beta_b[active_in_batch]
                linear_term[active_in_batch] += (beta_a ** 2 / sigma2) * XtX_diag[idx_active]

            quad_term = (beta_b ** 2 / (2 * sigma2)) * XtX_diag[batch_idx]
            logodds = logit_pi0 + linear_term - quad_term

            p_b = expit(logodds)
            p_b = np.clip(p_b, 1e-10, 1 - 1e-10)
            gamma_b_new = rng.binomial(1, p_b)

            if active_in_batch.any():
                eta_full -= (Xb[:, active_in_batch] * beta_b[active_in_batch]).sum(axis=1)
            newly_active = gamma_b_new == 1
            if newly_active.any():
                eta_full += (Xb[:, newly_active] * beta_b[newly_active]).sum(axis=1)

            gamma[batch_idx] = gamma_b_new

        k = gamma.sum()
        active = gamma == 1

        # ---- 3. Inclusion probability pi0 ----
        # no prior on pi0 yet -> kept fixed
        # pi0 = rng.beta(args.a + k, args.b + d - k)
        # logit_pi0 = np.log(pi0 / (1 - pi0))

        # ---- 4. Effect sizes beta (active/inactive block split) ----
        beta_new = np.zeros(d)
        if k > 0:
            X_gamma = X[:, active]
            precision_active = (X_gamma.T @ X_gamma) / sigma2 + np.eye(k) / tau2
            S_gamma = np.linalg.inv(precision_active)
            S_gamma = (S_gamma + S_gamma.T) / 2
            m_gamma = S_gamma @ (X_gamma.T @ ystar) / sigma2
            beta_new[active] = rng.multivariate_normal(m_gamma, S_gamma)
        if k < d:
            beta_new[~active] = rng.normal(0, np.sqrt(tau2), d - k)
        beta = beta_new

        # ---- 5. Residual variance sigma2 ----
        resid = ystar - X @ (gamma * beta)   # = y* - X_gamma @ beta_active
        RSS = resid @ resid
        sigma2 = invgamma.rvs(args.eps + n / 2, scale=args.eps + RSS / 2, random_state=rng)
        sigma = np.sqrt(sigma2)

        # ---- 6. Store samples ----
        y = ystar
        beta_samples[sample_idx] = beta
        sigma_samples[sample_idx] = sigma
        gamma_samples[sample_idx] = gamma
        pi0_samples[sample_idx] = pi0
        sample_idx += 1
        pbar.set_postfix(k=k, mean_rho=f"{gamma.mean():.3f}", sigma=f"{sigma:.3f}")

    # ---- saving results ----
    output_folder = args.output_folder

    if args.save_chain:
        np.save(f"{output_folder}/beta_samples.npy", beta_samples)
        np.save(f"{output_folder}/sigma_samples.npy", sigma_samples)
        np.save(f"{output_folder}/gamma_samples.npy", gamma_samples)
        np.save(f"{output_folder}/pi0_samples.npy", pi0_samples)

    #---- computing stats ----
    beta_b = beta_samples[burn_in:]
    gamma_b = gamma_samples[burn_in:]
    
    pip = np.mean(gamma_b, axis=0)
     
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

    
    # ---- computing conditional beta stats ----
    n_included = gamma_b.sum(axis=0)   

    masked = np.where(gamma_b == 1, beta_b, np.nan)
    with np.errstate(invalid="ignore"):
        beta_cond_mean = np.nanmean(masked, axis=0)
        beta_cond_std = np.nanstd(masked, axis=0)
    beta_cond_mean[n_included == 0] = np.nan
    beta_cond_std[n_included <= 1] = np.nan

    # ---- saving stats as CSV ----
    import csv
    
    beta_stats_df = pd.DataFrame({
        "var_idx": np.arange(d),
        "pip": pip,
        "n_included": n_included.astype(int),
        "beta_marg_mean": beta_mean,
        "beta_marg_std": beta_std,
        "beta_cond_mean": beta_cond_mean,
        "beta_cond_std": beta_cond_std,
        "beta_cond_hdi_low": beta_cond_hdi[:, 0],
        "beta_cond_hdi_high": beta_cond_hdi[:, 1],
        "ess": beta_ess,
    })
    
    beta_stats_df.to_csv(f"{output_folder}/beta_stats.csv", index=False)

    with open(f"{output_folder}/scalar_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "mean", "std", "hdi_low", "hdi_high", "ess"])
        writer.writerow(["sigma", sigma_mean, sigma_std, sigma_hdi[0], sigma_hdi[1], sigma_ess])
        #writer.writerow(["pi0", pi0, pi0_std, pi0_hdi[0], pi0_hdi[1], pi0_ess])

    
    # --- visualization ----
    if args.viz:
            beta_true = np.load(f"{input_folder}/beta.npy")
    
            viz_beta(beta_samples, gamma_samples, beta_true, beta_cond_mean, pip,
                     burn_in, n_iter, output_folder)
        
            #tymczasowo: 
            pi0_hdi = [0,1]
        
            viz_scalars(sigma_samples, pi0_samples,
                        sigma_mean, sigma_hdi, pi0, pi0_hdi,
                        burn_in, n_iter, output_folder)
    
            viz_gibbs(sigma_samples, pi0_samples, gamma_samples,
                      burn_in, n_iter, output_folder, rng=rng)

if __name__ == "__main__":
    main()