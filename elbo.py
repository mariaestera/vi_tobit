import numpy as np
from scipy.stats import norm
from scipy.special import ndtr
from numpy.polynomial.hermite import hermgauss


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