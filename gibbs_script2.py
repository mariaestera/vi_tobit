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
    
    args = parser.parse_args()
    return args

def save_args_command(args, script_name="gibbs_script.py"):
    """Save all arguments (including defaults) as a copy-paste-ready command."""
    flag_map = {
        "input_folder": "-input_folder", "output_folder": "-output_folder",
        "n_iter": "-n_iter", "burn_in": "-burn_in", "gamma_bath": "-gamma_bath",
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


class gibbs_disc_spike_slab():
    """
    Gibbs sampler for Tobit regression with a discrete spike-and-slab prior
    on beta, via variable-inclusion indicators gamma.

    Model:
        y_i = clip(y*_i; l, u)
        y*_i | beta, gamma ~ N(x_i^T Gamma beta, sigma2)
        beta ~ N(0, tau2 I)
        gamma_j ~ Bernoulli(pi0), iid
        sigma2 ~ InvGamma(delta, eps)

    Gamma updates are done SEQUENTIALLY (one variable at a time, each
    conditioned on the just-updated values of the others within the same
    sweep) to preserve the theoretical correctness of the Gibbs sampler.
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
        self.G = X.T @ X
        self.XtX_diag = np.diag(self.G)
        self._init_params()
        self.beta_history = []
        self.gamma_history = []
        self.sigma_history = []

    # ------------------------------------------------------------------
    def _init_params(self):
        d, n = self.d, self.n

        self.gamma = self.rng.binomial(1, self.pi0, d).astype(float)
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
        eta = self.X @ (self.gamma * self.beta)
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
        Xt = self.X * self.gamma[None, :]  # X_tilde = X Gamma
        precision = (Xt.T @ Xt) / self.sigma2 + np.eye(self.d) / self.tau2
        S = np.linalg.inv(precision)
        m = S @ (Xt.T @ self.ystar) / self.sigma2

        # Symmetrize for numerical stability before Cholesky/MVN sampling
        S = (S + S.T) / 2
        self.beta = self.rng.multivariate_normal(m, S)

    # ------------------------------------------------------------------
    def update_gamma_seq(self):
        """
        Sample gamma_j | beta, gamma_{-j}, y*, sigma2 SEQUENTIALLY,
        one variable at a time, each conditioned on the most recently
        updated values of the others (theoretically correct Gibbs sweep).
        """
        # Current linear predictor with full Gamma applied
        eta_full = self.X @ (self.gamma * self.beta)

        for j in self.rng.permutation(self.d):
            xj = self.X[:, j]
            beta_j = self.beta[j]

            # Residual excluding variable j's current contribution
            r_minus_j = self.ystar - eta_full + self.gamma[j] * beta_j * xj

            linear_term = (beta_j / self.sigma2) * (xj @ r_minus_j)
            quad_term = (beta_j**2 / (2 * self.sigma2)) * self.XtX_diag[j]

            logodds = self.logit_pi0 + linear_term - quad_term
            p_j = expit(logodds)
            p_j = np.clip(p_j, 1e-10, 1 - 1e-10)

            gamma_j_new = self.rng.binomial(1, p_j)

            # Update eta_full incrementally to reflect the new gamma_j
            eta_full = r_minus_j - self.gamma[j] * beta_j * xj  # remove old contrib (already excluded above, safe reset)
            eta_full = eta_full + gamma_j_new * beta_j * xj      # add new contrib

            self.gamma[j] = gamma_j_new

    def update_gamma(self):
        """
        Sample gamma_j | beta, gamma_{-j}, y*, sigma2 IN PARALLEL for all j
        """
        eta_full = self.X @ (self.gamma * self.beta)
    
        # Residual excluding each variable's own contribution, computed for
        # all j at once using the OLD (pre-sweep) gamma vector
        resid_base = self.ystar - eta_full
        correction = self.gamma * self.beta * self.XtX_diag
        linear_term = (self.beta / self.sigma2) * (self.X.T @ resid_base + correction)
        quad_term = (self.beta**2 / (2 * self.sigma2)) * self.XtX_diag
    
        logodds = self.logit_pi0 + linear_term - quad_term
        p = expit(logodds)
        p = np.clip(p, 1e-10, 1 - 1e-10)
    
        self.gamma = self.rng.binomial(1, p).astype(float)

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
    def update_tau2_em(self, damping= 0.3):

        tau2_new = np.mean(self.beta**2)
        self.tau2 = (1 - damping) * self.tau2 + damping * tau2_new
    
    def update_pi0_em(self, damping= 0.3):

        pi0_new = np.clip(self.gamma.mean(), 1e-4, 1 - 1e-4)
        self.pi0 = (1 - damping) * self.pi0 + damping * pi0_new
        self.logit_pi0 = logit(self.pi0)


    # ------------------------------------------------------------------
    def step(self, gamma_seq, em=False, em_damping=1.0):
        """One full Gibbs sweep: y* -> beta -> gamma (sequential) -> sigma2,
        optionally followed by an EM step updating tau2 and pi0."""
        self.update_ystar()
        self.update_beta()
        if gamma_seq:
            self.update_gamma_seq()
        else:
            self.update_gamma()
        self.update_sigma()
    
        if em:
            self.update_tau2_em(damping=em_damping)
            self.update_pi0_em(damping=em_damping)
    
        self.beta_history.append(self.beta.copy())
        self.gamma_history.append(self.gamma.copy())
        self.sigma_history.append(self.sigma2)
    
    # ------------------------------------------------------------------
    def fit(self, n_iter=2000, burn_in=500, gamma_seq=False, verbose=True,
            em=False, em_damping = 0.3, em_warmup=100):
        
        assert em_warmup <= burn_in, "em_warmup should not exceed burn_in"
    
        from tqdm import tqdm
        pbar = tqdm(range(n_iter), desc="Gibbs", disable=not verbose)
        
        for it in pbar:
            do_em = em and (em_warmup <= it < burn_in)
            self.step(gamma_seq, em=do_em, em_damping=em_damping)
    
        beta_samples = np.array(self.beta_history[burn_in:])
        gamma_samples = np.array(self.gamma_history[burn_in:])
        sigma_samples = np.array(self.sigma_history[burn_in:])
        
        return beta_samples, gamma_samples, sigma_samples



def main():    
    args = parse_args()
    save_args_command(args)

    # loading data
    input_folder = args.input_folder
    output_folder = args.output_folder

    X = np.load(f"{input_folder}/X.npy")
    ystar = np.load(f"{input_folder}/y_latent.npy")
    l, u = list(np.load(f"{input_folder}/l_u_sigma.npy"))[:2]
    y = np.clip(ystar, l, u).copy()

    n,d = X.shape

    model_gibbs = gibbs_disc_spike_slab(
        X, y,
        tau2=args.tau2,
        pi0=args.pi0,
        seed=args.seed,
    )

    beta_samples, gamma_samples, sigma_samples = model_gibbs.fit(
        n_iter = args.n_iter, 
        em_warmup = 0,
        burn_in = args.burn_in, 
        gamma_seq = True,
        em = False
    )


    #---- computing stats ----
    burn_in = args.burn_in
    
    beta_b = beta_samples
    gamma_b = gamma_samples
    
    pip = np.mean(gamma_b, axis=0)
     
    beta_mean = np.mean(beta_b, axis=0)
    beta_std = np.std(beta_b, axis=0)

    sigma_mean = sigma_samples.mean()
    sigma_std = sigma_samples.std()

    
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
        "beta_cond_std": beta_cond_std
    })
    
    beta_stats_df.to_csv(f"{output_folder}/beta_stats.csv", index=False)
    

if __name__ == "__main__":
    main()