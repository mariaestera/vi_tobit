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


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate effect sizes using Gibbs sampler")
    # input and output
    parser.add_argument("-input_folder", type=str, required=True, help="folder with X_design, y_latent, l, u, sigma_y_true")
    parser.add_argument("-output_folder", type=str, required=True, help="folder with estimates")

    # gibbs sampler params
    parser.add_argument("-n_iter", type=int, required=True, help="number of sampler iterations after burn-in")
    parser.add_argument("-burn_in", type=int, required=True, help="number of burn-in iterations")
    parser.add_argument("--gamma_batch", type=int, required=False, default=-1, help="number of the gamma_i sampled together; -1 - fully parralel update")

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

def save_args_command(args, script_name="gibbs_script.py"):
    """Save all arguments (including defaults) as a copy-paste-ready command."""
    flag_map = {
        "input_folder": "-input_folder", "output_folder": "-output_folder",
        "n_iter": "-n_iter", "burn_in": "-burn_in", "gamma_batch": "-gamma_batch",
        "tau2": "--tau2", "pi0": "--pi0", "eps": "--eps",
        "hdi": "--hdi", "viz": "--viz",
        "seed": "--seed", "save_chain": "--save_chain",
    }
    args_dict = vars(args)
    command_parts = [f"python {script_name}"]
    for key, value in args_dict.items():
        command_parts.append(f"{flag_map[key]} {value}")
    command_str = " \\\n    ".join(command_parts)
    with open(f"{args.output_folder}/args.txt", "w") as f:
        f.write(command_str + "\n")


# Numerical floors to keep log() / division well-defined under rounding error
_EPS_DEN = 1e-10
_EPS_LOG = 1e-12


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


class SparseTobitGibbs():
    """
    Gibbs sampler for Tobit regression with a discrete spike-and-slab prior
    on beta, via variable-inclusion indicators gamma.

    Model:
        y_i = clip(y*_i; l, u)
        y*_i | beta, gamma ~ N(x_i^T Gamma beta, sigma2)
        beta ~ N(0, tau2 I)
        gamma_j ~ Bernoulli(pi0), iid
        sigma2 ~ InvGamma(delta, eps)
        
    """

    def __init__(self, X, y, l=None, u=None, tau2=25.0, pi0=0.1,
                 delta=0.01, eps=0.01, seed=None):
        self.X = X
        self.y = np.asarray(y, dtype=float)
        self.n, self.d = X.shape
        self.rng = np.random.default_rng(seed)
        # Hyperparameters
        self.tau2 = tau2
        self.pi0 = pi0
        self.logit_pi0 = logit(pi0)
        self.delta = delta
        self.eps = eps
        # Censorship thresholds
        if l is None:
            l = np.min(self.y) if (self.y == np.min(self.y)).sum() > 5 else -np.inf
        if u is None:
            u = np.max(self.y) if (self.y == np.max(self.y)).sum() > 5 else np.inf
        self.l, self.u = l, u
        self.mask_l = (self.y == l)
        self.mask_u = (self.y == u)
        self.mask_mid = ~(self.mask_l | self.mask_u)
        # other quantities
        self.XtX_diag = np.sum(X**2, axis=0)
        self._init_params()
        self.beta_history = []
        self.gamma_history = []
        self.sigma_history = []
        
        self.total_fit_time = 0
        self.beta_fit_time = 0
        self.gamma_fit_time = 0
        self.ystar_fit_time = 0

    # ------------------------------------------------------------------
    def _init_params(self):
        d, n = self.d, self.n

        self.gamma = self.rng.binomial(1, self.pi0, d).astype(float)
        self.active = self.gamma == 1
        self.beta = self.rng.normal(0, np.sqrt(self.tau2), d)
        self.sigma2 = np.var(self.y)

        self.ystar = np.empty(n, dtype=float)
        self.ystar[self.mask_mid] = self.y[self.mask_mid]
        self.ystar[self.mask_l] = self.l
        self.ystar[self.mask_u] = self.u

    # ------------------------------------------------------------------
    def update_ystar(self):
        """
        Sample latent y* for censored observations from the appropriate
        truncated normal, conditional on current beta, gamma, sigma2.
        Observations in (l, u) are fixed at their observed value (not latent).
        """
        eta = self.X[:,self.active] @ self.beta[self.active]
        sigma = np.sqrt(self.sigma2)

        # Lower-censored: y*_i <= l
        if self.mask_l.any():
            idx = self.mask_l
            a = (-np.inf - eta[idx]) / sigma  # = -inf
            b = (self.l - eta[idx]) / sigma
            self.ystar[idx] = truncnorm.rvs(
                a, b, loc=eta[idx], scale=sigma, random_state=self.rng
            )

        # Upper-censored: y*_i >= u
        if self.mask_u.any():
            idx = self.mask_u
            a = (self.u - eta[idx]) / sigma
            b = (np.inf - eta[idx]) / sigma  # = inf
            self.ystar[idx] = truncnorm.rvs(
                a, b, loc=eta[idx], scale=sigma, random_state=self.rng
            )

        # Uncensored observations remain fixed at y_i
        self.ystar[self.mask_mid] = self.y[self.mask_mid]

    # ------------------------------------------------------------------
    def update_beta(self):
        """
        Sample beta | y*, gamma, sigma2 ~ N(m, S), using the effective
        design matrix X_tilde = X * Gamma (columns zeroed where gamma_j=0).
        """
        active = self.active
        n_active = int(active.sum())
    
        beta_new = np.empty(self.d)
    
        if n_active > 0:
            X_a = self.X[:, active]                                    # n x n_active
            precision_a = (X_a.T @ X_a) / self.sigma2 + np.eye(n_active) / self.tau2
            S_a = np.linalg.inv(precision_a)
            S_a = (S_a + S_a.T) / 2                                     # symetryzacja
            m_a = S_a @ (X_a.T @ self.ystar) / self.sigma2
            beta_new[active] = self.rng.multivariate_normal(m_a, S_a)
    
        if n_active < self.d:
            beta_new[~active] = self.rng.normal(0.0, np.sqrt(self.tau2), self.d - n_active)
    
        self.beta = beta_new

    # ------------------------------------------------------------------
    def update_gamma_seq(self):
        """
        Sample gamma_j | beta, gamma_{-j}, y*, sigma2 SEQUENTIALLY,
        one variable at a time, each conditioned on the most recently
        updated values of the others (theoretically correct Gibbs sweep).
        """
        # Current linear predictor with full Gamma applied
        eta_full = self.X[:,self.active] @ self.beta[self.active]

        for j in self.rng.permutation(self.d):
            xj = self.X[:, j]
            beta_j = self.beta[j]
        
            eta_minus_j = eta_full - self.gamma[j] * beta_j * xj
            r_minus_j = self.ystar - eta_minus_j
        
            linear_term = (beta_j / self.sigma2) * (xj @ r_minus_j)
            quad_term = (beta_j**2 / (2 * self.sigma2)) * self.XtX_diag[j]
            logodds = self.logit_pi0 + linear_term - quad_term
            p_j = np.clip(expit(logodds), 1e-10, 1 - 1e-10)
        
            gamma_j_new = self.rng.binomial(1, p_j)

            eta_full = eta_full + (gamma_j_new - self.gamma[j]) * self.beta[j] * xj 
            self.gamma[j] = gamma_j_new

    def update_gamma_batch(self, batch_size =10):
        """
        Sample gamma_j | beta, gamma_{-j}, y*, sigma2 in bathes,
        batch_size variables at a time, each conditioned on the most recently
        updated values of the others (theoretically correct Gibbs sweep).
        """
        eta_full = self.X[:,self.active] @ self.beta[self.active]

        perm = self.rng.permutation(self.d)
    
        for start in range(0, self.d, batch_size):
            idx = perm[start:start + batch_size]
            X_b = self.X[:, idx]              # n x b
            beta_b = self.beta[idx]           # b
            gamma_b_old = self.gamma[idx]     # b
    
            eta_minus_j = eta_full[:, None] - gamma_b_old * beta_b * X_b   # n x b
            r_minus_j = self.ystar[:, None] - eta_minus_j                  # n x b
    
            linear_term = (beta_b / self.sigma2) * np.einsum('nb,nb->b', X_b, r_minus_j)
            quad_term = (beta_b**2 / (2 * self.sigma2)) * self.XtX_diag[idx]
            logodds = self.logit_pi0 + linear_term - quad_term
            p_b = np.clip(expit(logodds), 1e-10, 1 - 1e-10)
    
            gamma_b_new = self.rng.binomial(1, p_b).astype(float)
    
            eta_full = eta_full + X_b @ ((gamma_b_new - gamma_b_old) * beta_b)
            
            self.gamma[idx] = gamma_b_new

    def update_gamma_parallel(self):
        """
        Sample gamma_j | beta, gamma_{-j}, y*, sigma2 IN PARALLEL for all j
        """
        eta_full = self.X[:,self.active] @ self.beta[self.active]
    
        resid_base = self.ystar - eta_full
        correction = self.gamma * self.beta * self.XtX_diag
        linear_term = (self.beta / self.sigma2) * (self.X.T @ resid_base + correction)
        quad_term = (self.beta**2 / (2 * self.sigma2)) * self.XtX_diag
    
        logodds = self.logit_pi0 + linear_term - quad_term
        p = expit(logodds)
        p = np.clip(p, 1e-10, 1 - 1e-10)
    
        self.gamma = self.rng.binomial(1, p).astype(float)

    def update_gamma(self, gamma_batch_size):
        
        if gamma_batch_size <0:
            self.update_gamma_parallel()
            
        elif gamma_batch_size > 1:
            self.update_gamma_batch(gamma_batch_size)
            
        else:
            self.update_gamma_seq()

        self.active = self.gamma.astype(bool)
        
    # ------------------------------------------------------------------
    def update_sigma(self):
        """
        Sample sigma2 | beta, gamma, y* ~ InvGamma(delta + n/2,
        eps + 0.5 * ||y* - X Gamma beta||^2).
        """
        resid = self.ystar - self.X @ (self.gamma * self.beta)
        a_post = self.delta + self.n / 2
        b_post = self.eps + 0.5 * np.sum(resid**2)
        self.sigma2 = invgamma(a_post, scale=b_post).rvs(random_state=self.rng)

                
    # ------------------------------------------------------------------
    def step(self, gamma_batch_size):
        """One full Gibbs sweep: y* -> beta -> gamma (sequential/batch) -> sigma2,
        optionally followed by an EM step updating tau2 and pi0."""

        start = time.perf_counter()
        self.update_ystar()
        self.ystar_fit_time += time.perf_counter() - start
        
        start = time.perf_counter()
        self.update_beta()
        self.beta_fit_time += time.perf_counter() - start

        start = time.perf_counter()
        self.update_gamma(gamma_batch_size)
        self.gamma_fit_time += time.perf_counter() - start
        
        self.update_sigma()
    
        self.beta_history.append(self.beta.copy())
        self.gamma_history.append(self.gamma.copy())
        self.sigma_history.append(self.sigma2)
    
    # ------------------------------------------------------------------
    def fit(self, n_iter=2000, burn_in=500, gamma_batch_size = -1, verbose=True):

        start = time.perf_counter()
    
        from tqdm import tqdm
        pbar = tqdm(range(n_iter), desc="Gibbs", disable=not verbose)
        
        for it in pbar:
            self.step(gamma_batch_size)
    
        beta_samples = np.array(self.beta_history[burn_in:])
        gamma_samples = np.array(self.gamma_history[burn_in:])
        sigma_samples = np.array(self.sigma_history[burn_in:])

        self.total_fit_time = time.perf_counter() - start
        
        return beta_samples, gamma_samples, sigma_samples

    
def main():
    
    args = parse_args()
    save_args_command(args)

    # loading data
    input_folder = args.input_folder

    X = np.load(f"{input_folder}/X.npy")
    n,d = X.shape
    ystar = np.load(f"{input_folder}/y_latent.npy")
    l, u,  sigma_y_true = list(np.load(f"{input_folder}/l_u_sigma.npy"))
    y = np.clip(ystar, l, u).copy()

    n_iter, burn_in = args.n_iter, args.burn_in

    model_gibbs = SparseTobitGibbs(
        X, y,
        tau2 = args.tau2,
        pi0 = args.pi0,
        seed = args.seed,
    )

    beta_samples, gamma_samples, sigma_samples = model_gibbs.fit(
        n_iter = burn_in + n_iter, 
        burn_in = 0, 
        gamma_batch_size = args.gamma_batch
    )

    pi0_samples = np.zeros(n_iter+burn_in)
    
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
    
    beta_stats_df.to_csv(f"{output_folder}/beta_stats.csv", index=False)

    with open(f"{output_folder}/scalar_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "true_value", "mean", "std", "hdi_low", "hdi_high", "ess"])
        writer.writerow(["sigma", sigma_y_true, sigma_mean, sigma_std, sigma_hdi[0], sigma_hdi[1], sigma_ess])
        #writer.writerow(["pi0", pi0, pi0_std, pi0_hdi[0], pi0_hdi[1], pi0_ess])


    # --- visualization ----
    if args.viz:    
            viz_beta(beta_samples, gamma_samples, beta_true, beta_cond_mean, pip,
                     burn_in, n_iter, output_folder)
        
            #tymczasowo: 
            pi0_hdi = [0,1]
        
            viz_scalars(sigma_samples, pi0_samples,
                        sigma_mean, sigma_hdi, args.pi0, pi0_hdi,
                        burn_in, n_iter, output_folder)
    
            viz_gibbs(sigma_samples, pi0_samples, gamma_samples,
                      burn_in, n_iter, output_folder)
    
    row = {
        "n": n,
        "d": d,
        "mean_sparsity": pip.mean(),
        "total_fit_time": model_gibbs.total_fit_time,
        "n_iters_with_burn_in": n_iter + burn_in,
        "ystar_fit_time": model_gibbs.ystar_fit_time,
        "gamma_fit_time": model_gibbs.gamma_fit_time,
        "gamma_batch_size": args.gamma_batch,
        "beta_fit_time": model_gibbs.beta_fit_time,
        "seed": args.seed,
    }
    
    pd.DataFrame([row]).to_csv(f"{output_folder}/time.csv", index=False)

if __name__ == "__main__":
    main()