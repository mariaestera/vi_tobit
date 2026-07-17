import numpy as np


def X_basic(n, d, intercept=True, seed=None):
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, d))
    if intercept:
        X = np.hstack([np.ones((n, 1)), X])  # prepend intercept column

    return X


import numpy as np


def X_corr_blocks(n, d, k, corr, intercept=True, seed=None):
    """
    Tworzy macierz eksperymentu o wymiarach (n, d), w której kolumny
    są skorelowane w blokach o rozmiarze k z siłą korelacji corr.
    """
    rng = np.random.default_rng(seed)

    # 1. Inicjalizacja macierzy kowariancji (d x d) jako macierzy jednostkowej
    Sigma = np.eye(d)

    # 2. Wypełnianie macierzy kowariancji blokami o maksymalnym rozmiarze k
    for i in range(0, d, k):
        # Zabezpieczenie na wypadek, gdyby d nie dzieliło się bez reszty przez k
        end = min(i + k, d)
        block_size = end - i

        # Tworzenie pojedynczego bloku kowariancji
        block_cov = np.full((block_size, block_size), corr)
        np.fill_diagonal(block_cov, 1.0)  # Na przekątnej zawsze 1

        # Wstawienie bloku do głównej macierzy
        Sigma[i:end, i:end] = block_cov

    # 3. Generowanie danych z wielowymiarowego rozkładu normalnego
    mean = np.zeros(d)
    X = rng.multivariate_normal(mean, cov=Sigma, size=n)

    # 4. Opcjonalne dodanie kolumny jedynek (wyrazu wolnego)
    if intercept:
        X = np.hstack([np.ones((n, 1)), X])

    return X

def beta_basic(d, beta_scale, seed = None):
    rng = np.random.default_rng(seed)

    beta_true = beta_scale * rng.standard_normal(d)

    return beta_true

def beta_sparse(d, beta_scale, perc, seed = None):
    assert perc > 0 and perc < 1
    rng = np.random.default_rng(seed)

    beta_raw = beta_basic(d, beta_scale, seed)
    indices = rng.choice([0,1], d, p = [1-perc,perc])

    return beta_raw * indices


def y_tobit(X, l_perc, u_perc, beta_true, sigma_y_true=1.0, seed=None):
    rng = np.random.default_rng(seed)

    n, d = X.shape

    # 3. Censoring thresholds & observed y
    y_latent = X @ beta_true + rng.normal(0, sigma_y_true, n)
    l, u = np.percentile(y_latent, l_perc), np.percentile(y_latent, u_perc)
    y = np.clip(y_latent, l, u)

    return y, l, u

