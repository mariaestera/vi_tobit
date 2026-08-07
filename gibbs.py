import numpy as np
from scipy.stats import truncnorm, invgamma
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.special import expit, logit


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





def gibbs_spike_slab(X, y,
                     n_iter=5000,
                     burn_in=1000,
                     tau0=0.1,
                     tau1=10,
                     pi=0.01,
                     a0=0.01,
                     b0=0.01,
                    seed = None):

    rng = np.random.seed(42)

    n, p = X.shape
    l, u = y.min(), y.max()
    y = y.copy()

    mask_l = (y == l)
    mask_u = (y == u)
    mask_mid = ~mask_l & ~mask_u

    beta = np.zeros(p)
    sigma2 = 1.0
    gamma = np.ones(p, dtype=int)

    beta_samples = np.zeros((n_iter - burn_in, p))
    sigma_samples = np.zeros(n_iter - burn_in)
    gamma_samples = np.zeros((n_iter - burn_in, p), dtype=int)

    XtX = X.T @ X

    sample_idx = 0
    pbar = tqdm(range(n_iter), desc="Gibbs")

    for it in pbar:

        ###############################################
        # Step 0. Data augumentation
        ###############################################

        mu = X @ beta
        sigma = np.sqrt(sigma2)

        a = (l - mu[mask_l]) / sigma
        b = np.inf

        y[mask_l] = truncnorm.rvs(
            a=a,
            b=b,
            loc=mu[mask_l],
            scale=sigma,
        )

        a = -np.inf
        b = (u - mu[mask_u]) / sigma

        y[mask_u] = truncnorm.rvs(
            a=a,
            b=b,
            loc=mu[mask_u],
            scale=sigma,
        )

        Xty = X.T @ y

        ###############################################
        # Step 1. Sample beta
        ###############################################

        tau2 = np.where(gamma == 1, tau1**2, tau0**2)

        A = XtX.copy()
        A[np.diag_indices(p)] += 1.0 / tau2

        c, lower = cho_factor(A, lower=True, check_finite=False)

        m = cho_solve((c, lower), Xty, check_finite=False)

        z = np.random.randn(p)
        beta = m + np.sqrt(sigma2) * solve_triangular(
            c.T, z, lower=False, check_finite=False
        )

        ###############################################
        # Step 2. Sample sigma²
        ###############################################

        resid = y - X @ beta

        shape = a0 + (n + p) / 2

        quad = np.sum(beta**2 / tau2)

        scale = b0 + 0.5 * (resid @ resid + quad)

        sigma2 = invgamma.rvs(a=shape, scale=scale)

        ###############################################
        # Step 3. Sample gamma
        ###############################################

        for j in range(p):

            log_p1 = (
                np.log(pi)
                -0.5*np.log(sigma2*tau1**2)
                -beta[j]**2/(2*sigma2*tau1**2)
            )
            
            log_p0 = (
                np.log(1-pi)
                -0.5*np.log(sigma2*tau0**2)
                -beta[j]**2/(2*sigma2*tau0**2)
            )

            mlog = max(log_p1, log_p0)

            p1 = np.exp(log_p1 - mlog)
            p0 = np.exp(log_p0 - mlog)

            gamma[j] = np.random.binomial(1, p1 / (p1 + p0))
            
        pi = (gamma.sum() + 1) / (p + 2)

        active = gamma == 1

        if active.any():
            tau1 = np.clip(
                np.sqrt(np.mean(beta[active]**2)/sigma2),
                1,
                100
            )

        ###############################################
        # Save samples
        ###############################################

        if it >= burn_in:
            beta_samples[sample_idx] = beta
            sigma_samples[sample_idx] = sigma2
            gamma_samples[sample_idx] = gamma
            sample_idx += 1

        pbar.set_postfix(
            sigma2=f"{sigma2:.3f}",
            active=int(gamma.sum())
        )

    return {
        "beta": beta_samples,
        "sigma2": sigma_samples,
        "gamma": gamma_samples
    }

def trace_plot(samples, names=None, figsize=(12,4)):

    if samples.ndim == 1:
        samples = samples[:,None]

    n_param = samples.shape[1]

    fig, axes = plt.subplots(
        n_param,
        1,
        figsize=figsize,
        squeeze=False
    )

    for i in range(n_param):

        axes[i,0].plot(samples[:,i])

        if names:
            axes[i,0].set_title(names[i])
        else:
            axes[i,0].set_title(f"parameter {i}")

        axes[i,0].set_xlabel("iteration")

    plt.tight_layout()
    plt.show()





def autocorrelation(x, max_lag=100):
    """
    Compute the autocorrelation function (ACF) of a 1D chain up to max_lag.

    Parameters
    ----------
    x : array-like, shape (n_iter,)
        Scalar chain values (e.g. one beta_j coefficient, or sigma2).
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    acf : ndarray, shape (max_lag+1,)
        Autocorrelation values for lags 0, 1, ..., max_lag (acf[0] = 1).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    var = np.dot(x, x) / n

    acf = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.dot(x[:-lag], x[lag:]) / (n * var)
    return acf


def gelman_rubin(chains):
    """
    Compute the Gelman-Rubin R-hat statistic for multiple chains of a
    scalar quantity, to assess convergence (R-hat close to 1 = good).

    Parameters
    ----------
    chains : array-like, shape (n_chains, n_iter)
        Post-burn-in samples of a scalar quantity from multiple independent
        chains (e.g. different seeds / initializations).

    Returns
    -------
    r_hat : float
    """
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape  # m chains, n samples each

    chain_means = chains.mean(axis=1)
    grand_mean = chain_means.mean()

    # Between-chain variance
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)

    # Within-chain variance
    W = np.mean(chains.var(axis=1, ddof=1))

    var_hat = (1 - 1 / n) * W + B / n
    r_hat = np.sqrt(var_hat / W) if W > 0 else np.nan
    return r_hat


def diagnose_burn_in(chains, quantity_name="beta_0", max_lag=100,
                      acf_threshold=0.1, plot=True):
    """
    Diagnose whether burn-in / chain length is sufficient by:
      1. Plotting trace plots for multiple chains (visual mixing check).
      2. Plotting the autocorrelation function (ACF) per chain, and
         reporting the lag at which ACF first drops below acf_threshold
         (a rough proxy for the "effective" decorrelation time).
      3. Computing the Gelman-Rubin R-hat statistic across chains
         (values > ~1.1 typically indicate insufficient burn-in / mixing).

    Parameters
    ----------
    chains : array-like, shape (n_chains, n_iter)
        Post-burn-in samples of a SCALAR quantity (e.g. a single beta_j,
        or sigma2) from multiple independent Gibbs chains (different seeds).
    quantity_name : str
        Label for the quantity being diagnosed (used in plot titles).
    max_lag : int
        Maximum lag for the ACF plot.
    acf_threshold : float
        ACF level considered "sufficiently decorrelated" (used to report
        an approximate decorrelation lag per chain).
    plot : bool
        Whether to produce diagnostic plots.

    Returns
    -------
    diagnostics : dict
        {
          "r_hat": float,
          "decorrelation_lag": list of int (per chain, first lag where
              |ACF| < acf_threshold; np.nan if never reached within max_lag),
          "acf_per_chain": ndarray, shape (n_chains, max_lag+1)
        }
    """
    chains = np.asarray(chains, dtype=float)
    n_chains, n_iter = chains.shape

    # --- Gelman-Rubin R-hat ---
    r_hat = gelman_rubin(chains)

    # --- Autocorrelation per chain ---
    acf_per_chain = np.array([autocorrelation(chains[c], max_lag=max_lag)
                               for c in range(n_chains)])

    decorrelation_lag = []
    for c in range(n_chains):
        below = np.where(np.abs(acf_per_chain[c]) < acf_threshold)[0]
        decorrelation_lag.append(below[0] if len(below) > 0 else np.nan)

    # --- Reporting ---
    print(f"Diagnostics for '{quantity_name}' ({n_chains} chains, {n_iter} samples each)")
    print(f"Gelman-Rubin R-hat: {r_hat:.4f} "
          f"({'OK' if r_hat < 1.1 else 'WARNING: chains may not have converged'})")
    for c, lag in enumerate(decorrelation_lag):
        lag_str = f"{lag}" if not np.isnan(lag) else f">{max_lag} (not reached)"
        print(f"  Chain {c}: decorrelation lag (|ACF| < {acf_threshold}) = {lag_str}")

    # --- Plots ---
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Trace plot
        for c in range(n_chains):
            axes[0].plot(chains[c], alpha=0.7, linewidth=0.8, label=f"Chain {c}")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel(quantity_name)
        axes[0].set_title(f"Trace plot ({quantity_name})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ACF plot
        lags = np.arange(max_lag + 1)
        for c in range(n_chains):
            axes[1].plot(lags, acf_per_chain[c], alpha=0.7, label=f"Chain {c}")
        axes[1].axhline(acf_threshold, linestyle="--", color="red", linewidth=1,
                        label=f"Threshold ({acf_threshold})")
        axes[1].axhline(-acf_threshold, linestyle="--", color="red", linewidth=1)
        axes[1].set_xlabel("Lag")
        axes[1].set_ylabel("Autocorrelation")
        axes[1].set_title(f"ACF ({quantity_name})")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        "r_hat": r_hat,
        "decorrelation_lag": decorrelation_lag,
        "acf_per_chain": acf_per_chain,
    }