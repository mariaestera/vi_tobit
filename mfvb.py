import numpy as np
from scipy.stats import norm
from scipy.special import logit, expit, log_ndtr
from tqdm.auto import tqdm


def inv_mills_ratio(z, upper=False):
    """Numerycznie stabilny odwrotny Mills ratio: pdf(z)/cdf(z) (lub /(1-cdf(z)) dla upper=True)."""
    if upper:
        return np.exp(norm.logpdf(z) - log_ndtr(-z))
    return np.exp(norm.logpdf(z) - log_ndtr(z))


class q_tobit:
    """
    Mean-field variational inference for Tobit regression with spike-and-slab prior
    on beta coefficients.
    
    Variation parameters:
    
    m : (d,) - mean q(beta) [slab]
    S : (d,d) - covariance matrix q(beta)
    rho : (d,) - inclusion probabilities q(gamma_j=1)
    a, b : scalars - parameters q(sigma^2) ~ InvGamma(a,b)
    mu : (n,) - E_q[y*] (latent y)
    Sigma_ii: (n,) - Var_q(y*)
    Prior hyperparameters:
    tau2 - Gaussian prior variance on beta (slab)
    pi0 - prior on the probability of variable inclusion
    delta, eps - prior parameters InvGamma(delta, eps) on sigma^2
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
        self.G = X.T @ X
        self.XtX_diag = np.diag(self.G)

        self._init_params()
        self.elbo_history = []

    # ------------------------------------------------------------------
    def _init_params(self):
        d, n = self.d, self.n

        self.a = self.delta + n / 2
        self.b = np.var(self.y) *self.a  

        self.rho = self.rng.uniform(0, 1, d)
        beta0 = self.rng.normal(0, np.sqrt(self.tau2), d)
        self.m = self.rho * beta0
        self.S = np.eye(d) * self.tau2
        self.S_diag = np.diag(self.S)

        self.R = self._build_R(self.rho)

        self.eta = self.X @ self.m
        self.nu = np.sqrt(self.b / self.a)

        self.mu = np.empty(n, dtype=float)
        self.mu[self.mask_mid] = self.y[self.mask_mid]
        self.Sigma_ii = np.zeros(n, dtype=float)

        self._update_ystar_censored()

    def _build_R(self, rho):
        """R = E_q[gamma gamma^T] elementwise w G, dla P = diag(rho)."""
        R = self.G * np.outer(rho, rho)
        np.fill_diagonal(R, self.XtX_diag * rho)
        return R

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
        """q(beta): aktualizuje S, m."""
        d = self.d
        self.S = np.linalg.inv((self.a / self.b) * self.R + (1 / self.tau2) * np.eye(d))
        self.S_diag = np.diag(self.S)
        self.m = (self.a / self.b) * self.S @ (self.rho * (self.X.T @ self.mu))

    def update_ystar(self):
        """q(y*): aktualizuje eta, nu, mu, Sigma_ii dla obserwacji przyciętych."""
        self.eta = self.X @ (self.rho * self.m)
        self.nu = np.sqrt(self.b / self.a)
        self._update_ystar_censored()

    def update_gamma(self):
        """q(gamma): aktualizuje rho oraz zależne od niego R."""
        resid_base = self.mu - self.eta
        Xt_resid_base = self.X.T @ resid_base
        correction = self.rho * self.m * self.XtX_diag

        linear_term = (self.a / self.b) * self.m * (Xt_resid_base + correction)
        quad_term = (self.a / (2 * self.b)) * (self.m**2 + self.S_diag) * self.XtX_diag

        self.rho = expit(self.logit_pi0 - quad_term + linear_term)
        self.R = self._build_R(self.rho)

    def update_sigma(self):
        """q(sigma^2): aktualizuje parametr b (a jest stałe, ustalone przez n)."""
        trace_RS = np.sum(self.R * self.S)
        self.b = self.eps + 0.5 * (
            np.sum(self.Sigma_ii + self.mu**2)
            - 2 * self.m @ (self.rho * (self.X.T @ self.mu))
            + trace_RS
            + self.m @ self.R @ self.m
        )

    # ------------------------------------------------------------------
    def compute_elbo(self):
        # E_q[log p(beta)]
        q_beta = (
            -self.d / 2 * np.log(2 * np.pi * self.tau2)
            - 1 / (2 * self.tau2) * (
                np.trace(self.S)
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
            - 0.5 * (
                np.log(self.b) - digamma(self.a)
            )
            - self.a / (2 * self.b) * (
                self.Sigma_ii.sum()
                + np.sum(self.mu**2)
                - 2 * self.m.T @ np.diag(self.rho) @ self.X.T @ self.mu
                + np.trace(self.R @ self.S)
                + self.m.T @ self.R @ self.m
            )
        )

        q_beta_entropy = (
            -self.d / 2 * np.log(2 * np.pi)
            - 0.5 * np.linalg.slogdet(self.S)[1]
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
    def step(self):
        """Jeden pełny cykl aktualizacji CAVI: beta -> y* -> gamma -> sigma."""
        self.update_beta()
        self.update_ystar()
        self.update_gamma()
        self.update_sigma()
        self.elbo_history.append(self.compute_elbo())

    def fit(self, n_iter=1000, verbose=True):
        pbar = tqdm(range(n_iter), desc="MFVI", disable=not verbose)
        for _ in pbar:
            self.step()
        return self

    # ------------------------------------------------------------------
    @property
    def sig_vi(self, threshold=0.95):
        """Zwraca maskę zmiennych uznanych za istotne (rho > threshold)."""
        return self.rho > threshold

    def summary(self):
        return {
            "rho": self.rho.copy(),
            "m": self.m.copy(),
            "S_diag": self.S_diag.copy(),
            "a": self.a,
            "b": self.b,
            "E_sigma2": self.b / (self.a - 1) if self.a > 1 else np.nan,
        }