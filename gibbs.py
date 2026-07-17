import numpy as np
from scipy.stats import truncnorm, invgamma
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


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