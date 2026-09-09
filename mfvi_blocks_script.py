import numpy as np
from scipy.stats import norm
from scipy.special import logit, expit, log_ndtr, gammaln, digamma
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
import aux_functions as aux_f
from mfvi_eval import mfvi_eval

def parse_args():
    parser = argparse.ArgumentParser(description="Estimate effect sizes using Gibbs sampler")
    # input and output
    parser.add_argument("-input_folder", type=str, required=True, help="folder with X_design, y_latent, l, u, sigma_y_true")
    parser.add_argument("-output_folder", type=str, required=True, help="folder with estimates")

    # sampler params
    parser.add_argument("--n_iter", type=int, required=False, default = 1000, help="maximum number of CAVI full iterations after EM warmup")
    parser.add_argument("--em", type=int, required=False, default = 1, help= "em: 1 - with EM steps, 0 - without EM")
    parser.add_argument("--em-warm_up", type=int, required=False, default = 100, help= "number of initial iterations without update hyperparams")
    parser.add_argument("--gamma_batch", type=int, required=False, default=-1, help="number of the gamma_i updated together; -1 - fully parralel update")
    parser.add_argument("--tol", type=float, required=False, default = 0.01, help="treshold for ELBO divergence")

    # initialization
    parser.add_argument("--tau2", type=float, required=False, default=100, help="Initial prior variance")
    parser.add_argument("--pi0", type=float, required=False, default=0.1, help="Initial prior probability of inclusion")
    parser.add_argument("--eps", type=float, required=False, default=0.01, help="sigma^2 ~ InvGamma(eps, eps)")

    #stats
    parser.add_argument("--hdi", type=float, required=False, default=0.95, help = "Level of HDI")

    # other settings
    parser.add_argument("--seed", type=int, required=False, default=None, help = "random seed")

    args = parser.parse_args()
    return args, parser
        
def inv_mills_ratio(z, upper=False):
    if upper:
        return np.exp(norm.logpdf(z) - log_ndtr(-z))
    return np.exp(norm.logpdf(z) - log_ndtr(z))

class SparseTobitStructuredVI:
    """
    Mean-field variational inference for Tobit regression with spike-and-slab prior
    on beta coefficients, using a block-structured variational factor for beta.

    Variational parameters:

    m : (d,) - mean q(beta) [slab], concatenated across blocks
    S : list of (|B_k|,|B_k|) arrays - block-diagonal covariance matrices q(beta_{B_k}),
        one dense matrix per block k = 1,...,K
    S_diag : (d,) - diagonal of the full block-diagonal covariance, in original
        coordinate order (i.e. the marginal variances S_{jj})
    rho : (d,) - inclusion probabilities q(gamma_j=1)
    a, b : scalars - parameters q(sigma^2) ~ InvGamma(a,b)
    mu : (n,) - E_q[y*] (latent y)
    Sigma_ii: (n,) - Var_q(y*)

    Prior hyperparameters:
    tau2 - Gaussian prior variance on beta (slab)
    pi0 - prior on the probability of variable inclusion
    delta, eps - prior parameters InvGamma(delta, eps) on sigma^2

    Block structure:
    beta_blocks : (d,) - array assigning each coordinate j = 0,...,d-1 to a block id,
        e.g. [0, 0, 0, 1, 1, 2, 2, 2, ...] groups the first 3 coordinates into block 0,
        the next 2 into block 1, and so on. Block ids need not be contiguous or sorted,
        but each unique value defines one block B_k over which q(beta_{B_k}) is jointly
        Gaussian; coordinates in different blocks are assumed independent under q(beta).
        If None, all coordinates are placed in a single block (i.e. the non-blocked,
        fully joint q(beta)).
    """

    def __init__(self, X, y, l=None, u=None, beta_blocks = None, tau2=1.0, pi0=0.1,
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

        # Censorship tresholds
        if l is None:
            l = np.min(self.y) if (self.y == np.min(self.y)).sum() > 5 else -np.inf
        if u is None:
            u = np.max(self.y) if (self.y == np.max(self.y)).sum() > 5 else np.inf
        self.l, self.u = l, u

        self.mask_l = (self.y == l)
        self.mask_u = (self.y == u)
        self.mask_mid = ~(self.mask_l | self.mask_u)

        # other quantities
        self.beta_blocks = beta_blocks if beta_blocks is not None else np.zeros(self.d)
        self.unique_blocks = np.unique(self.beta_blocks)
        self.block_idx = [np.where(self.beta_blocks == k)[0] for k in self.unique_blocks]
        
        self.G = self.X.T @ self.X
        self.XtX_diag =(self.X**2).sum(axis=0)
        

        self._init_params()
        self.elbo_history = []
        self.covergence = None

        self.total_fit_time = 0
        self.beta_fit_time = 0
        self.gamma_fit_time = 0
        self.ystar_fit_time = 0
        

    # ------------------------------------------------------------------
    def _init_params(self):
        d, n = self.d, self.n

        self.a = self.delta + n / 2
        self.b = np.var(self.y) * self.a  

        self.rho = self.rng.uniform(0, 1, d)
        beta0 = self.rng.normal(0, np.sqrt(self.tau2), d)
        self.m = self.rho * beta0
        
        self.S = [
            self.tau2 * np.eye(np.sum(self.beta_blocks == k))
            for k in self.unique_blocks
        ]
        self.S_diag = self.tau2 * np.ones(len(self.beta_blocks))

        self.eta = self.X @ self.m
        self.nu = np.sqrt(self.b / self.a)

        self.mu = np.empty(n, dtype=float)
        self.mu[self.mask_mid] = self.y[self.mask_mid]
        self.Sigma_ii = np.zeros(n, dtype=float)            
        
        self._update_ystar_censored()


    def _update_ystar_censored(self):
        """Aktualizuje mu, Sigma_ii, lambda_l/u, z_l/u dla obserwacji przyciętych."""
        nu = self.nu
        eta, l, u = self.eta, self.l, self.u

        if self.mask_l.any():
            self.z_l = (l - eta[self.mask_l]) / nu
            self.lambda_l = inv_mills_ratio(self.z_l)
            self.mu[self.mask_l] = eta[self.mask_l] - nu * self.lambda_l
            self.Sigma_ii[self.mask_l] = nu**2 * (1 - self.lambda_l * (self.lambda_l + self.z_l))

        if self.mask_u.any():
            self.z_u = (u - eta[self.mask_u]) / nu
            self.lambda_u = inv_mills_ratio(self.z_u, upper=True)
            self.mu[self.mask_u] = eta[self.mask_u] + nu * self.lambda_u
            self.Sigma_ii[self.mask_u] = nu**2 * (1 - self.lambda_u * (self.lambda_u - self.z_u))

    # ------------------------------------------------------------------
    def update_beta(self):
        """q(beta): block-coordinate updates of S, m."""
        a_over_b = self.a / self.b

        Xt_mu = self.X.T @ self.mu  # (d,)
        self.R = np.zeros((self.d, self.d))

        for i, k in enumerate(self.unique_blocks):
            idx = np.where(self.beta_blocks == k)[0]      # coordinates in block k
            idx_c = np.where(self.beta_blocks != k)[0]    # coordinates outside block k

            X_B = self.X[:, idx]
            rho_B = self.rho[idx]
            rho_c = self.rho[idx_c]

            # R_{B,B} = G_{B,B} * [diag(rho_B(1-rho_B)) + rho_B rho_B^T]
            G_BB = self.G[np.ix_(idx, idx)]
            R_BB = G_BB * (np.diag(rho_B * (1 - rho_B)) + np.outer(rho_B, rho_B))

            # R_{B,-B} = G_{B,-B} * (rho_B rho_{-B}^T)
            G_B_c = self.G[np.ix_(idx, idx_c)]
            R_B_c = G_B_c * np.outer(rho_B, rho_c)

            # S_B = (a/b * R_BB + 1/tau2 * I)^{-1}
            S_B_inv = a_over_b * R_BB + np.eye(len(idx)) / self.tau2
            S_B = np.linalg.inv(S_B_inv)

            # m_B = a/b * S_B * (P_B X_B^T mu - R_{B,-B} m_{-B})
            m_B = a_over_b * S_B @ (rho_B * Xt_mu[idx] - R_B_c @ self.m[idx_c])

            self.S[i] = S_B
            self.m[idx] = m_B
            self.S_diag[idx] = np.diag(S_B)

            self.R[np.ix_(idx, idx)] = R_BB
            self.R[np.ix_(idx, idx_c)] = R_B_c
            self.R[np.ix_(idx_c, idx)] = R_B_c.T

        # Refresh eta = X @ m, used by q(y*) updates
        self.eta = self.X @ self.m        
        

    def update_ystar(self):
        """q(y*): updates eta, nu, mu, Sigma_ii  for censored observations."""
        self.eta = self.X @ (self.rho * self.m)
        self.nu = np.sqrt(self.b / self.a)
        self._update_ystar_censored()

    def update_gamma_parallel(self):
        
        resid_base = self.mu - self.eta
        Xt_resid_base = self.X.T @ resid_base
        correction = self.rho * self.m * self.XtX_diag
        linear_term = (self.a / self.b) * self.m * (Xt_resid_base + correction)
        quad_term = (self.a / (2 * self.b)) * (self.m**2 + self.S_diag) * self.XtX_diag
        self.rho = expit(self.logit_pi0 - quad_term + linear_term)
        self.rho = np.clip(self.rho, 1e-10, 1 - 1e-10)


    def update_gamma_batch(self, batch_size):
        order = self.rng.permutation(self.d)
    
        for start in range(0, self.d, batch_size):
            batch = order[start:start + batch_size]
    
            rho_old_batch = self.rho[batch].copy()
            rho_new_batch = np.empty_like(rho_old_batch)
    
            for k, j in enumerate(batch):
                xj = self.X[:, j]
                resid_j = self.mu - self.eta + rho_old_batch[k] * self.m[j] * xj
    
                linear_term = (self.a / self.b) * self.m[j] * (xj @ resid_j)
                quad_term = (self.a / (2 * self.b)) * (self.m[j]**2 + self.S_diag[j]) * self.XtX_diag[j]
    
                rho_j_new = expit(self.logit_pi0 - quad_term + linear_term)
                rho_new_batch[k] = np.clip(rho_j_new, 1e-10, 1 - 1e-10)
    
            delta = (rho_new_batch - rho_old_batch) * self.m[batch]
            self.eta = self.eta + self.X[:, batch] @ delta
    
            self.rho[batch] = rho_new_batch


    def update_gamma_seq(self):
        for j in self.rng.permutation(self.d):
            xj = self.X[:, j]
    
            # Residual excluding variable j's current contribution
            resid_j = self.mu - self.eta + self.rho[j] * self.m[j] * xj
    
            linear_term = (self.a / self.b) * self.m[j] * (xj @ resid_j)
            quad_term = (self.a / (2 * self.b)) * (self.m[j]**2 + self.S_diag[j]) * self.XtX_diag[j]
    
            rho_j_new = expit(self.logit_pi0 - quad_term + linear_term)
            rho_j_new = np.clip(rho_j_new, 1e-10, 1 - 1e-10)
    
            # Update eta incrementally to reflect the new rho_j
            #self.eta = resid_j - self.rho[j] * self.m[j] * xj + rho_j_new * self.m[j] * xj
            self.eta = self.eta - (self.rho[j] - rho_j_new) * self.m[j] * xj
    
            self.rho[j] = rho_j_new

    def update_gamma(self, batch_size):
        
        if batch_size <0:
            self.update_gamma_parallel()
            
        elif batch_size > 1:
            self.update_gamma_batch(batch_size)
            
        else:
            self.update_gamma_seq()
    

    def update_sigma(self):
        """q(sigma^2): updates b (a is constant, given by n)."""

        trace_RS = sum(
            np.trace(self.R[np.ix_(idx, idx)] @ self.S[i])
            for i, idx in enumerate(self.block_idx)
        )
        self.b = self.eps + 0.5 * (
            np.sum(self.Sigma_ii + self.mu**2)
            - 2 * self.m @ (self.rho * (self.X.T @ self.mu))
            + trace_RS
            + self.m @ self.R @ self.m
        )

    # ------------------------------------------------------------------
    def compute_elbo(self):
        # trace(S) and log|S|, computed block-wise since S is block-diagonal
        trace_S = sum(np.trace(S_k) for S_k in self.S)
        logdet_S = sum(np.linalg.slogdet(S_k)[1] for S_k in self.S)

        # trace(R @ S), using only the diagonal blocks of R (off-diagonal blocks
        # of S are zero, so they don't contribute)
        trace_RS = sum(
            np.trace(self.R[np.ix_(idx, idx)] @ self.S[i])
            for i, idx in enumerate(self.block_idx)
        )

        # E_q[log p(beta)]
        q_beta = (
            -self.d / 2 * np.log(2 * np.pi * self.tau2)
            - 1 / (2 * self.tau2) * (
                trace_S
                + self.m.T @ self.m
            )
        )

        # E_q[log p(gamma)]
        q_gamma = np.sum(
            self.rho * np.log(self.pi0)
            + (1 - self.rho) * np.log(1 - self.pi0)
        )

        # E_q[log p(sigma_y^2)]
        q_sigma = (
            self.delta * np.log(self.eps)
            - gammaln(self.delta)
            - (self.delta + 1) * (
                np.log(self.b) - digamma(self.a)
            )
            - self.a * self.eps / self.b
        )

        # E_q[log p(y* | beta, sigma^2)]
        q_y_star = (
            -self.n / 2 * np.log(2 * np.pi)
            - self.n / 2 * (
                np.log(self.b) - digamma(self.a)
            )
            - self.a / (2 * self.b) * (
                self.Sigma_ii.sum()
                + np.sum(self.mu**2)
                - 2 * self.m.T @ (self.rho * (self.X.T @ self.mu))
                + trace_RS
                + self.m.T @ self.R @ self.m
            )
        )

        # E_q[log q(beta)], entropy computed block-wise via logdet_S
        q_beta_entropy = (
            -self.d / 2 * np.log(2 * np.pi)
            - 0.5 * logdet_S
            - self.d / 2
        )

        # E_q[log q(gamma)]
        q_gamma_entropy = -np.sum(
            self.rho * np.log(self.rho)
            + (1 - self.rho) * np.log(1 - self.rho)
        )

        # E_q[log q(sigma_y^2)]
        q_sigma_entropy = (
            -gammaln(self.a)
            - np.log(self.b)
            + (self.a + 1) * digamma(self.a)
            - self.a
        )
        # E_q[log q(y^*)]
        q_y_star_entropy = 0.0
        # lower censored observations
        if self.mask_l.any():
            q_y_star_entropy += np.sum(
                -0.5 * np.log(2 * np.pi * self.nu**2)
                - log_ndtr(self.z_l)
                - 0.5 * (
                    1
                    - self.z_l * self.lambda_l
                    - self.lambda_l**2
                )
            )

        # upper censored observations
        if self.mask_u.any():
            q_y_star_entropy += np.sum(
                -0.5 * np.log(2 * np.pi * self.nu**2)
                - norm.logsf(self.z_u)
                - 0.5 * (
                    1
                    + self.z_u * self.lambda_u
                    - self.lambda_u**2
                )
            )

        elbo = (
            q_y_star
            + q_beta
            + q_gamma
            + q_sigma
            - q_y_star_entropy
            - q_beta_entropy
            - q_gamma_entropy
            - q_sigma_entropy
        )
        return elbo
                    
    # ------------------------------------------------------------------
    def update_tau2(self, damping=0.3):
        """
        M-step with damping: move only partway toward the new EM estimate,
        to avoid destabilizing CAVI with abrupt hyperparameter jumps.
        
        damping : float in (0, 1]
            1.0 = full EM update (no damping)
            smaller values = more conservative, slower update
        """
        tau2_new = np.mean(self.m**2 + self.S_diag)
        self.tau2 = (1 - damping) * self.tau2 + damping * tau2_new
    
    
    def update_pi0(self, damping=0.3):
        pi0_new = np.clip(self.rho.mean(), 1e-4, 1 - 1e-4)
        self.pi0 = (1 - damping) * self.pi0 + damping * pi0_new
        self.logit_pi0 = logit(self.pi0)
        
    
    # ------------------------------------------------------------------
    def step(self, em, damping, gamma_batch_size):

        start = time.perf_counter()
        self.update_beta()
        self.beta_fit_time += time.perf_counter() - start

        start = time.perf_counter()
        self.update_ystar()
        self.ystar_fit_time += time.perf_counter() - start

        start = time.perf_counter()
        self.update_gamma(gamma_batch_size)
        self.gamma_fit_time += time.perf_counter() - start
        
        self.update_sigma()
        
        self.elbo_history.append(self.compute_elbo())

        if em:
            self.update_tau2(damping)
            self.update_pi0(damping)

    def fit(self, n_iter=1000, em = True, em_warmup=0, damping=0.3, tol = 1e-5, gamma_batch_size=-1, verbose=True):

        start = time.perf_counter()
        
        n_it = em_warmup if em else n_iter
        desc = "MFVI (em_warmup)" if em else "MFVI"
        
        pbar = tqdm(range(n_it), desc = desc, disable=not verbose)
        
        for it in pbar:
            
            self.step(em=False, damping = damping, gamma_batch_size = gamma_batch_size)
            
            pbar.set_postfix(elbo=f"{self.elbo_history[it]:.4f}")

            rel_change = abs(self.elbo_history[it] - self.elbo_history[it-1]) / abs(self.elbo_history[it-1])
                
            if rel_change < tol  and it >10:
                
                if em:
                    print("Early stopping em_warmup")

                else:   
                    self.covergence = True
                    print("Early stopping")
                    
                break
        else:
            if not em:
                self.covergence = False
                print("ELBO didn't converge")

        if em:
            n_it = n_iter - em_warmup
            desc = "MFVI (EM)"
            
            pbar = tqdm(range(n_it), desc = desc, disable=not verbose)
            
            for it in pbar:
                
                self.step(em=False, damping = damping, gamma_batch_size = gamma_batch_size)
                
                pbar.set_postfix(elbo=f"{self.elbo_history[it]:.4f}")
    
                rel_change = abs(self.elbo_history[it] - self.elbo_history[it-1]) / abs(self.elbo_history[it-1])
                    
                if rel_change < tol and it >10:
                    print("Early stopping")
                    break
            

        self.total_fit_time = time.perf_counter() - start
        
        return self

    # ------------------------------------------------------------------
    def sig_vi(self, threshold=0.9):

        return self.rho > threshold

    def summary(self):
        return {
            "rho": self.rho.copy(),
            "m": self.m.copy(),
            "S_diag": self.S_diag.copy(),
            "a": self.a,
            "b": self.b,
            "E_sigma2": self.b / (self.a - 1) if self.a > 1 else np.nan,
            "covergence": self.covergence
        }

        
def summary_orig(summary, mu_y, sd_y, intercept_idx=None):
    m_orig = summary["m"].copy() * sd_y
    S_diag_orig = summary["S_diag"].copy() * sd_y**2 

    if intercept_idx is not None:
        m_orig[intercept_idx] += mu_y
        
    a_orig = summary["a"]
    b_orig = summary["b"] * sd_y**2
    E_sigma2_orig = b_orig / (a_orig - 1) if a_orig > 1 else np.nan

    return {
        "rho": summary["rho"].copy(),
        "m": m_orig,
        "S_diag": S_diag_orig,
        "a": a_orig,
        "b": b_orig,
        "E_sigma2": E_sigma2_orig,
        "n_iters": summary["n_iters"],
        "covergence": summary["covergence"],
    }


def main():
    
    args, parser = parse_args()
    aux_f.save_args_command(args,parser, "mfvi_script.py")

    # loading data
    input_folder = args.input_folder
    output_folder = args.output_folder

    X = np.load(f"{input_folder}/X.npy")
    n,d = X.shape
    ystar = np.load(f"{input_folder}/y_latent.npy")
    l, u,  sigma_y_true = list(np.load(f"{input_folder}/l_u_sigma.npy"))
    y = np.clip(ystar, l, u).copy()
    beta_blocks = np.load(f"{input_folder}/beta_blocks.npy")

    start = time.perf_counter()
    
    model_vi = SparseTobitStructuredVI(
        X, y,
        beta_blocks = beta_blocks,
        tau2= args.tau2,
        pi0= args.pi0, 
        seed= args.seed,
        delta=args.eps, 
        eps=args.eps
    )
    
    model_vi.fit(
        n_iter = args.n_iter, 
        em = False if args.em == 0 else True,
        em_warmup = args.em_warm_up, 
        gamma_batch_size = args.gamma_batch,
        tol = args.tol
    )

    total_time = start - time.perf_counter()

    summary = model_vi.summary()
    summary["n_iters"] = len(model_vi.elbo_history)

    comput_time= {
        "total": total_time,
        "fit": model_vi.total_fit_time,
        "gamma": model_vi.gamma_fit_time
    }
    
    mfvi_eval(summary, X, ystar, comput_time, args)

if __name__ == "__main__":
    main()