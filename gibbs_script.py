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
from gibbs_eval import gibbs_eval


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
        "seed": "--seed", "save_chain": "--save_chain"
    }
    args_dict = vars(args)
    command_parts = [f"python {script_name}"]
    for key, value in args_dict.items():
        command_parts.append(f"{flag_map[key]} {value}")
    command_str = " \\\n    ".join(command_parts)
    with open(f"{args.output_folder}/args.txt", "w") as f:
        f.write(command_str + "\n")


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
        self.update_gamma(gamma_batch_size)
        self.gamma_fit_time += time.perf_counter() - start

        start = time.perf_counter()
        self.update_beta()
        self.beta_fit_time += time.perf_counter() - start
        
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

    total_time = time.perf_counter()
    
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

    total_time -= time.perf_counter()
    
    gibbs_eval(model_gibbs, args, total_time,n,d)

if __name__ == "__main__":
    main()