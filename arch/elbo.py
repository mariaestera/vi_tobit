import numpy as np
from scipy.stats import norm
from scipy.special import ndtr
from numpy.polynomial.hermite import hermgauss
from tqdm import tqdm


def gh_integral(f, mu, s, n_gh = 20):
    """Gauss-Hermite quadrature of ∫ f(z) N(z;mu,s) dz.
    n_gh - number of Gauss-Hermite nodes
    """

    gh_nodes, gh_weights = hermgauss(n_gh) # GH nodes & weights for  ∫ f(z) N(z;μ,s) dz ≈ Σ w_k f(μ + √(2s) t_k) / √π
    z = mu + np.sqrt(2 * s) * gh_nodes

    return np.dot(gh_weights, f(z)) / np.sqrt(np.pi)


def elbo(X, y,l,u, tau2, m, log_s, log_sigma2_y):
    s = np.exp(log_s)  # diagonal of S  (>0)
    sigma2_y = np.exp(log_sigma2_y)  # noise variance (>0)
    sigma_y = np.sqrt(sigma2_y)

    mu_pred = X @ m  # shape (n,)
    s_pred = np.einsum('ij,j,ij->i', X, s, X)  # x_i^T diag(s) x_i

    val = 0.0

    mask_l = (y == l)
    mask_u = (y == u)
    mask_mid = ~mask_l & ~mask_u

    n,d = X.shape

    # ── uncensored likelihood ────────────────────────────────────────────────
    if mask_mid.any():
        r = y[mask_mid] - mu_pred[mask_mid]
        val += np.sum(-0.5 * np.log(2 * np.pi * sigma2_y)
                      - (r ** 2 + s_pred[mask_mid]) / (2 * sigma2_y))

    # ── lower-censored likelihood (GH) ──────────────────────────────────────
    for i in np.where(mask_l)[0]:
        def f_l(z, _mu=mu_pred[i], _s=s_pred[i]):
            arg = (l - z) / sigma_y
            log_phi = norm.logcdf(arg)
            return log_phi

        val += gh_integral(f_l, mu_pred[i], s_pred[i])

    # ── upper-censored likelihood (GH) ──────────────────────────────────────
    for i in np.where(mask_u)[0]:
        def f_u(z, _mu=mu_pred[i], _s=s_pred[i]):
            arg = (z - u) / sigma_y
            log_phi = norm.logcdf(arg)
            return log_phi

        val += gh_integral(f_u, mu_pred[i], s_pred[i])

    # ── KL(q || p) ──────────────────────────────────────────────────────────
    log_det_S = np.sum(log_s)
    kl = 0.5 * (d * np.log(tau2) - log_det_S - d
                + np.sum(s) / tau2 + np.dot(m, m) / tau2)
    val -= kl

    return val

def elbo_grad(X, y,l,u, tau2,m, log_s, log_sigma2_y, eps=1e-5):
    """Numerical gradient via central differences."""
    d = len(m)

    params = np.concatenate([m, log_s, [log_sigma2_y]])
    grad   = np.zeros_like(params)
    f0     = elbo(X, y,l,u, tau2, params[:d], params[d:2*d], params[-1])
    for k in range(len(params)):
        p_plus       = params.copy(); p_plus[k]  += eps
        p_minus      = params.copy(); p_minus[k] -= eps
        grad[k] = (elbo(X, y,l,u, tau2, p_plus[:d],  p_plus[d:2*d],  p_plus[-1]) -
                   elbo(X, y,l,u, tau2,p_minus[:d], p_minus[d:2*d], p_minus[-1])) / (2 * eps)
    return grad

def elbo_grad_analytic(X, y, l, u, tau2, m, log_s, log_sigma2_y, n_gh = 20):
    """
    Analytic gradient of ELBO w.r.t. (m, log_s, log_sigma2_y).

    Parametrisation:
        s        = exp(log_s)        -- diagonal of S  (elementwise)
        sigma2_y = exp(log_sigma2_y) -- noise variance

    Chain rule gives:
        d ELBO / d log_s_ii      = s_ii      * (d ELBO / d s_ii)
        d ELBO / d log_sigma2_y  = sigma2_y  * (d ELBO / d sigma2_y)
    """
    n, d = X.shape
    s        = np.exp(log_s)          # (d,)
    sigma2_y = np.exp(log_sigma2_y)   # scalar
    sigma_y  = np.sqrt(sigma2_y)      # scalar

    mask_l = (y == l)
    mask_u = (y == u)
    mask_mid = ~mask_l & ~mask_u

    mu_pred = X @ m                                        # (n,)  x_i^T m
    s_pred  = np.einsum('ij,j,ij->i', X, s, X)            # (n,)  x_i^T S x_i

    # accumulators
    grad_m        = np.zeros(d)
    grad_s        = np.zeros(d)   # w.r.t. s_ii (will be chain-ruled at the end)
    grad_sigma2_y = 0.0

    # ── precompute GH nodes mapped to each observation ───────────────────────
    # For obs i:  z = mu_pred[i] + sqrt(2 * s_pred[i]) * t,  t = gh_nodes
    # weight factor: w_k / sqrt(pi)

    gh_nodes, gh_weights = hermgauss(n_gh)

    T  = gh_nodes                          # (n_gh,)
    W  = gh_weights / np.sqrt(np.pi)       # (n_gh,)  normalised

    # ── 1. uncensored observations  y_i ∈ (l, u) ────────────────────────────
    if mask_mid.any():
        Xi   = X[mask_mid]                         # (n_mid, d)
        ri   = y[mask_mid] - mu_pred[mask_mid]     # (n_mid,)
        spri = s_pred[mask_mid]                    # (n_mid,)

        # grad m:  sum_i  (y_i - mu_i) / sigma2_y  * x_i
        grad_m += Xi.T @ (ri / sigma2_y)

        # grad s_jj:  sum_i  -1/(2 sigma2_y) * x_ij^2
        #   (because d s_pred_i / d s_jj = x_ij^2)
        grad_s += np.einsum('ij,ij->j', Xi**2, -0.5 / sigma2_y * np.ones((mask_mid.sum(), d)))

        # grad sigma2_y:  sum_i [ -1/(2 sigma2_y) + (ri^2 + s_pred_i)/(2 sigma2_y^2) ]
        grad_sigma2_y += np.sum(-0.5 / sigma2_y
                                + (ri**2 + spri) / (2 * sigma2_y**2))

    # ── 2. lower-censored  y_i = l ──────────────────────────────────────────
    # E_q[log Phi((l-z)/sigma_y)]  with  z ~ N(mu_i, s_pred_i)
    #
    # d/d mu_i  = E_q[ log Phi(...) * (z - mu_i) / s_pred_i ]
    # d/d s_jj  = E_q[ log Phi(...) * (z-mu_i)^2 - s_pred_i) / (2 s_pred_i^2) ] * x_ij^2
    # d/d sigma2_y = E_q[ -phi(...)/Phi(...) * (l-z)/(2 sigma_y^3) ]

    for i in np.where(mask_l)[0]:
        mu_i  = mu_pred[i]
        sp_i  = s_pred[i]
        z_i   = mu_i + np.sqrt(2 * sp_i) * T       # (n_gh,)
        arg   = (l - z_i) / sigma_y
        log_Phi = norm.logcdf(arg)                  # (n_gh,)
        ratio   = np.exp(norm.logpdf(arg) - norm.logcdf(arg))  # phi/Phi

        # d ELBO / d mu_i
        dmu = np.dot(W, log_Phi * (z_i - mu_i) / sp_i)
        grad_m += X[i] * dmu

        # d ELBO / d s_jj  (via s_pred_i = x_i^T diag(s) x_i)
        ds_pred = np.dot(W, log_Phi * ((z_i - mu_i)**2 - sp_i) / (2 * sp_i**2))
        grad_s  += X[i]**2 * ds_pred

        # d ELBO / d sigma2_y
        grad_sigma2_y += np.dot(W, -ratio * (l - z_i) / (2 * sigma_y**3))

    # ── 3. upper-censored  y_i = u ──────────────────────────────────────────
    # E_q[log Phi((z-u)/sigma_y)]
    #
    # Same structure, opposite sign on the Phi argument.

    for i in np.where(mask_u)[0]:
        mu_i  = mu_pred[i]
        sp_i  = s_pred[i]
        z_i   = mu_i + np.sqrt(2 * sp_i) * T
        arg   = (z_i - u) / sigma_y
        log_Phi = norm.logcdf(arg)
        ratio   = np.exp(norm.logpdf(arg) - norm.logcdf(arg))  # phi/Phi

        dmu = np.dot(W, log_Phi * (z_i - mu_i) / sp_i)
        grad_m += X[i] * dmu

        ds_pred = np.dot(W, log_Phi * ((z_i - mu_i)**2 - sp_i) / (2 * sp_i**2))
        grad_s  += X[i]**2 * ds_pred

        grad_sigma2_y += np.dot(W, ratio * (z_i - u) / (2 * sigma_y**3))

    # ── 4. KL(q || p) gradients  (minus sign — KL is subtracted from ELBO) ──
    # KL = 0.5 * [ p*log(tau2) - log|S| - p + tr(S)/tau2 + ||m||^2/tau2 ]
    #
    # d KL / d m_j       =  m_j / tau2
    # d KL / d s_jj      =  1/(2 tau2)  -  1/(2 s_jj)   [from -log|S| + tr(S)/tau2]
    # d KL / d sigma2_y  =  0

    grad_m -= m / tau2
    grad_s -= (0.5 / tau2 - 0.5 / s)   # note: subtracting d_KL/d_s

    # ── 5. chain rule to log-parametrisation ─────────────────────────────────
    # d ELBO / d log_s_ii     = s_ii     * (d ELBO / d s_ii)
    # d ELBO / d log_sigma2_y = sigma2_y * (d ELBO / d sigma2_y)

    grad_log_s        = s * grad_s
    grad_log_sigma2_y = sigma2_y * grad_sigma2_y

    return np.array([*grad_m, *grad_log_s, grad_log_sigma2_y])

def adam(X, y,l,u, tau2,m0, log_s0, log_sigma2_y0,
         lr=0.05, n_iter=2000, beta1=0.9, beta2=0.999, eps_adam=1e-8, early_stop = 0.1):

    d = len(m0)

    params = np.concatenate([m0, log_s0, [log_sigma2_y0]])
    mt_adam = np.zeros_like(params)
    vt_adam = np.zeros_like(params)
    elbo_hist = [-1_000_000]

    for t in range(1, n_iter + 1):
        g = elbo_grad_analytic(X, y,l,u, tau2, params[:d], params[d:2 * d], params[-1])
        mt_adam = beta1 * mt_adam + (1 - beta1) * g
        vt_adam = beta2 * vt_adam + (1 - beta2) * g ** 2
        m_hat = mt_adam / (1 - beta1 ** t)
        v_hat = vt_adam / (1 - beta2 ** t)
        params += lr * m_hat / (np.sqrt(v_hat) + eps_adam)  # maximise

        if t % 100 == 0:
            ev = elbo(X, y,l,u, tau2, params[:d], params[d:2 * d], params[-1])
            elbo_hist.append(ev)
            print(f"  iter {t:4d}  ELBO = {ev:.4f}")

            if elbo_hist[-1] - elbo_hist[-2] < early_stop:
                print("Early stopping")
                break

    return params[:d], params[d:2 * d], params[-1], elbo_hist[1:]

### Spike and slab ####

import numpy as np
from scipy.stats import norm

def _unpack(theta, d):
    m = theta[0:d]
    log_s2 = theta[d:2*d]
    logit_gamma = theta[2*d:3*d]
    log_sigma_y2 = theta[3*d]
    s2 = np.exp(log_s2)
    gamma = 1.0 / (1.0 + np.exp(-logit_gamma))
    sigma_y2 = np.exp(log_sigma_y2)
    return m, s2, gamma, sigma_y2

def _pack(m, s2, gamma, sigma_y2):
    logit_gamma = np.log(gamma) - np.log(1.0 - gamma)
    log_s2 = np.log(s2)
    log_sigma_y2 = np.log(sigma_y2)
    return np.concatenate([m, log_s2, logit_gamma, [log_sigma_y2]])

def elbo_spike_and_slab(theta, X, y, l, u, pi0, tau2, n_mc=20, seed=None):
    n, d = X.shape
    m, s2, gamma, sigma_y2 = _unpack(theta, d)
    sigma_y = np.sqrt(sigma_y2)

    Sigma_diag = gamma * s2
    mu = X @ m
    var = (X ** 2) @ Sigma_diag

    idx_l = (y == l)
    idx_u = (y == u)
    idx_c = ~(idx_l | idx_u)

    rng = np.random.default_rng(seed)

    ll = 0.0

    if idx_c.any():
        yi = y[idx_c]
        mui = mu[idx_c]
        vari = var[idx_c]
        ll += np.sum(
            -0.5 * np.log(2 * np.pi * sigma_y2)
            - ((yi - mui) ** 2 + vari) / (2 * sigma_y2)
        )

    def censored_term(idx, boundary, sign):
        if not idx.any():
            return 0.0
        mui = mu[idx]
        sdi = np.sqrt(var[idx])
        eps = rng.standard_normal((n_mc, idx.sum()))
        ui = mui[None, :] + sdi[None, :] * eps
        z = sign * (boundary - ui) / sigma_y
        log_phi = norm.logcdf(z)
        return np.sum(np.mean(log_phi, axis=0))

    ll += censored_term(idx_l, l, 1.0)
    ll += censored_term(idx_u, u, -1.0)

    kl = np.sum(
        gamma * (
            np.log(gamma / pi0)
            + 0.5 * np.log(tau2 / s2)
            + (s2 + m ** 2) / (2 * tau2)
            - 0.5
        )
        + (1 - gamma) * np.log((1 - gamma) / (1 - pi0))
    )

    return ll - kl

def elbo_spike_and_slab_grad(theta, X, y, l, u, pi0, tau2, n_mc=20, seed=None):
    n, d = X.shape
    m, s2, gamma, sigma_y2 = _unpack(theta, d)
    sigma_y = np.sqrt(sigma_y2)

    Sigma_diag = gamma * s2
    mu = X @ m
    var = (X ** 2) @ Sigma_diag

    idx_l = (y == l)
    idx_u = (y == u)
    idx_c = ~(idx_l | idx_u)

    rng = np.random.default_rng(seed)

    d_mu = np.zeros(n)
    d_var = np.zeros(n)
    d_sigma_y2 = 0.0

    if idx_c.any():
        yi = y[idx_c]
        mui = mu[idx_c]
        vari = var[idx_c]
        d_mu[idx_c] = (yi - mui) / sigma_y2
        d_var[idx_c] = -1.0 / (2 * sigma_y2)
        d_sigma_y2 += np.sum(
            -1.0 / (2 * sigma_y2)
            + ((yi - mui) ** 2 + vari) / (2 * sigma_y2 ** 2)
        )

    def censored_grad(idx, boundary, sign):
        nonlocal d_sigma_y2
        if not idx.any():
            return
        mui = mu[idx]
        sdi = np.sqrt(var[idx])
        eps = rng.standard_normal((n_mc, idx.sum()))
        ui = mui[None, :] + sdi[None, :] * eps
        z = sign * (boundary - ui) / sigma_y
        pdf_cdf_ratio = np.exp(norm.logpdf(z) - norm.logcdf(z))

        dz_dmu = sign * (-1.0) / sigma_y
        dz_dsd = sign * (-eps) / sigma_y
        dz_dsigma_y = -z / sigma_y

        g = pdf_cdf_ratio
        d_mu[idx] += np.mean(g * dz_dmu, axis=0)

        safe_sdi = np.where(sdi > 0, sdi, 1.0)
        d_sd = np.mean(g * dz_dsd, axis=0)
        d_var_i = d_sd / (2 * safe_sdi)
        d_var_i = np.where(sdi > 0, d_var_i, 0.0)
        d_var[idx] += d_var_i

        d_sigma_y_scalar = np.sum(np.mean(g * dz_dsigma_y, axis=0))
        d_sigma_y2 += d_sigma_y_scalar / (2 * sigma_y)

    censored_grad(idx_l, l, 1.0)
    censored_grad(idx_u, u, -1.0)

    d_m = X.T @ d_mu
    d_Sigma_diag = (X ** 2).T @ d_var

    d_gamma_ll = s2 * d_Sigma_diag
    d_s2_ll = gamma * d_Sigma_diag

    d_m_kl = gamma * m / tau2
    d_s2_kl = gamma * (1.0 / (2 * tau2) - 1.0 / (2 * s2))
    d_gamma_kl = (
        np.log(gamma / (1 - gamma))
        - np.log(pi0 / (1 - pi0))
        + 0.5 * np.log(tau2 / s2)
        + (s2 + m ** 2) / (2 * tau2)
        - 0.5
    )

    d_m_total = d_m - d_m_kl
    d_s2_total = d_s2_ll - d_s2_kl
    d_gamma_total = d_gamma_ll - d_gamma_kl

    d_log_s2 = d_s2_total * s2
    d_logit_gamma = d_gamma_total * gamma * (1 - gamma)
    d_log_sigma_y2 = d_sigma_y2 * sigma_y2

    grad = np.concatenate([d_m_total, d_log_s2, d_logit_gamma, [d_log_sigma_y2]])
    return grad

def adam_vi(X, y, l, u, pi0=0.5, tau2=1.0, n_iter=2000, lr=0.01,
            seed=None, theta0=None, n_mc=20,
            beta1=0.9, beta2=0.999, eps=1e-8):
    n, d = X.shape
    rng = np.random.default_rng(seed)

    if theta0 is None:
        m0 = np.zeros(d)
        s2_0 = np.ones(d) * tau2
        gamma0 = np.ones(d) * pi0
        sigma_y2_0 = np.var(y[(y > l) & (y < u)]) if np.any((y > l) & (y < u)) else 1.0
        theta = _pack(m0, s2_0, gamma0, sigma_y2_0)
    else:
        theta = theta0.copy()

    p = theta.shape[0]
    mt = np.zeros(p)
    vt = np.zeros(p)

    elbo_history = np.zeros(n_iter)

    pbar = tqdm(range(1, n_iter + 1), desc="ADAM VI")
    for it in pbar:
        step_seed = None if seed is None else seed + it
        grad = elbo_spike_and_slab_grad(theta, X, y, l, u, pi0, tau2,
                                         n_mc=n_mc, seed=step_seed)
        grad = -grad

        mt = beta1 * mt + (1 - beta1) * grad
        vt = beta2 * vt + (1 - beta2) * (grad ** 2)
        mt_hat = mt / (1 - beta1 ** it)
        vt_hat = vt / (1 - beta2 ** it)

        theta = theta - lr * mt_hat / (np.sqrt(vt_hat) + eps)

        elbo_history[it - 1] = elbo_spike_and_slab(theta, X, y, l, u, pi0, tau2,
                                                    n_mc=n_mc, seed=step_seed)

    d = X.shape[1]
    m, s2, gamma, sigma_y2 = _unpack(theta, d)

    return {
        "theta": theta,
        "m": m,
        "s2": s2,
        "gamma": gamma,
        "sigma_y2": sigma_y2,
        "elbo_history": elbo_history,
    }