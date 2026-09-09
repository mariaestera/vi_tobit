import argparse
import numpy as np
import aux_functions as aux_f
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def parse_args():
    parser = argparse.ArgumentParser(description="Simulate Tobit data. Creates disagn matrix X, dependent latent variable y^*, censored variable y. Returns all of those arrays and vector of true betas")

    #X_design
    parser.add_argument("-n", type=int, required=True, help="n smples")
    parser.add_argument("-d", type=int, required=True, help="d dimensions")
    parser.add_argument("-X_structure", type=str, required=True, help="type of simulated X: 'basic', 'corr_blocks', 'diagonal', 'AR'")
    parser.add_argument("--k", type=int, required=False, default = 10, help="Size of correlated blocks / width of the correlated diagonal")
    parser.add_argument("--corr", type=float, required=False, default = 0.7, help="strength of correlation of predictors")
    parser.add_argument("--intercept", type=int, required=False, default = 1, help="inclusion of the intercept: 1 -True/0 -False")
    
    #y_censored
    parser.add_argument("-l_perc", type=float, required=True, help="lower percentile censoring threshold")
    parser.add_argument("-u_perc", type=float, required=True, help="upper percentile censoring threshold")

    #betas
    parser.add_argument("-snr", type=float, required=True, help="signal to noise ratio")
    parser.add_argument("--pi0", type=float, required=False, default = 0.1, help="percentage of the true significant effects")
    
    #other settings
    parser.add_argument("--seed", type=int, required=False, default = None, help="random seed")
    parser.add_argument("--input_folder", type=str, required=False, default = "Jupyter/vi_tobit/simulations", help="path to save results")
    parser.add_argument("--output_folder", type=str, required=False,default = "Jupyter/vi_tobit/simulations", help="path to future estimates")
    parser.add_argument("--test", type=int, required=False, default = 0, help="test dataset size")
    
    
    args = parser.parse_args()

    assert args.X_structure in ['basic', 'corr_blocks', 'diagonal', 'AR']
    assert (args.corr <= 1) and (args.corr >= 0)
    assert (args.snr > 0)
    assert (args.pi0 < 1) and (args.pi0 > 0)
    assert args.l_perc < args.u_perc
    assert (args.l_perc >= 0) and (args.u_perc <= 100)
    assert (args.intercept == 0) or (args.intercept == 1)

    return args, parser



def X_basic(n, d, intercept=True, rng=np.random.default_rng(None)):

    X = rng.standard_normal((n, d))
    
    if intercept:
        X = np.hstack([np.ones((n, 1)), X])  # prepend intercept column

    return X

def X_corr_blocks(n, d, k, corr, intercept=True, rng=np.random.default_rng(None)):

    Sigma = np.eye(d)

    for i in range(0, d, k):
        end = min(i + k, d)
        block_size = end - i

        block_cov = np.full((block_size, block_size), corr)
        np.fill_diagonal(block_cov, 1.0) 

        Sigma[i:end, i:end] = block_cov

    mean = np.zeros(d)
    X = rng.multivariate_normal(mean, cov=Sigma, size=n)

    if intercept:
        X = np.hstack([np.ones((n, 1)), X])

    return X

def X_corr_diag(n, d, k, corr, intercept=True, rng=np.random.default_rng(None)):

    Sigma = np.eye(d)
    for offset in range(1, k):
        idx = np.arange(d - offset)
        Sigma[idx, idx + offset] = corr
        Sigma[idx + offset, idx] = corr
    mean = np.zeros(d)
    X = rng.multivariate_normal(mean, cov=Sigma, size=n)
    if intercept:
        X = np.hstack([np.ones((n, 1)), X])
    return X


def X_ar(n, d, k, corr, intercept=False, rng=np.random.default_rng(None)):

    n_blocks = int(np.ceil(d / k))
    d_padded = n_blocks * k

    eps = rng.standard_normal((n, n_blocks, k))
    X_blocks = np.empty_like(eps)
    X_blocks[:, :, 0] = eps[:, :, 0]

    scale = np.sqrt(1 - corr**2)
    for t in range(1, k):
        X_blocks[:, :, t] = corr * X_blocks[:, :, t - 1] + scale * eps[:, :, t]

    X = X_blocks.reshape(n, d_padded)[:, :d]

    if intercept:
        X = np.hstack([np.ones((n, 1)), X])
    return X


def X_design(n, d, struct, k =10, corr = 0.7, intercept= False, rng=np.random.default_rng(None)):

    if struct == "basic":
        return X_basic(n, d, intercept, rng)
        
    elif struct == "corr_blocks":
        return X_corr_blocks(n, d, k, corr, intercept, rng)
   
    elif struct == "diagonal":
        return X_corr_diag(n, d, k, corr, intercept, rng)

    elif struct == "AR":
        return X_ar(n, d, k, corr, intercept, rng)

    else:
        print("Unknown type of X_design")


def spike_and_slab(d, pi0, mean =0, scale =1, rng = None):

    if rng is None:
        rng = np.random.default_rng()

    return rng.normal(mean, scale, d) * (rng.uniform(0, 1,d) < pi0).astype(int)


def beta_sparse(d, X, pi0, snr, rng=None):
    # X: raw (standardized) design matrix (n x d); R = X^T X / n is the correlation matrix
    if rng is None:
        rng = np.random.default_rng()

    beta_raw = spike_and_slab(d, pi0, rng=rng)

    h2 = snr / (snr + 1)

    has_intercept = np.all(X[:, 0] == 1)
    X_signal = X[:, 1:] if has_intercept else X
    
    Xb = X_signal @ beta_raw
    v = (Xb @ Xb) / X.shape[0]

    beta = np.sqrt(h2 / v) * beta_raw

    return beta, 1 - h2

def y_tobit(X, beta, l_perc, u_perc, noise_var, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    
    n, d = X.shape
    
    signal = X @ beta
    
    y_latent = signal + rng.normal(0, np.sqrt(noise_var), n)

    l, u = np.percentile(y_latent, l_perc), np.percentile(y_latent, u_perc)
    
    return y_latent, l, u

def estimate_blocks(X, tol=1e-8):

    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1.0  # avoid division by zero for all-zero columns
    X_normalized = X / norms

    G = X_normalized.T @ X_normalized
    adjacency = csr_matrix(np.abs(G) > tol)

    n_components, labels = connected_components(
        adjacency, directed=False, connection="weak"
    )

    beta_blocks = np.empty_like(labels)
    seen = {}
    next_id = 0
    for j, lbl in enumerate(labels):
        if lbl not in seen:
            seen[lbl] = next_id
            next_id += 1
        beta_blocks[j] = seen[lbl]

    return beta_blocks

def main():
    args, parser = parse_args()
    aux_f.save_args_command(args,parser, "simulate_data_script.py")

    n, d = args.n, args.d
    k = args.k
    seed = args.seed
    rng = np.random.default_rng(seed)
    folder = args.input_folder
    intercept = True if args.intercept == 1 else False
    

    X = X_design(n, d, args.X_structure, k = k, corr = args.corr, intercept=intercept, rng=rng)

    n, d_total = X.shape 
    
    beta, noise_var = beta_sparse(d, X, args.pi0, args.snr, rng=rng)
    
    if intercept:
        beta = np.concatenate([[0], beta])
        
    y_latent, l, u = y_tobit(X, beta, args.l_perc, args.u_perc, noise_var, rng)
    
    np.save(f"{folder}/X.npy", X)
    np.save(f"{folder}/beta.npy", beta)
    np.save(f"{folder}/y_latent.npy", y_latent)
    np.save(f"{folder}/l_u_sigma.npy", np.array([l, u, np.sqrt(noise_var)]))

    if args.X_structure == 'basic':
        beta_blocks = np.zeros(d, dtype=int)
        
    elif args.X_structure in ['corr_blocks', 'AR']:
        if k< d/2:
            beta_blocks = np.concatenate([[0], 1 + np.arange(d) // k]) if intercept else np.arange(d) // k
        else:
            beta_blocks = estimate_blocks(X, tol=0.5)
    else:
        print("Unknown X_design type")

    np.save(f"{folder}/beta_blocks.npy", beta_blocks)

    
    if args.test > 0:
        
        rng = np.random.default_rng(2*seed)
        
        X = X_design(args.test, d, args.X_structure, k = args.k, corr = args.corr, intercept=intercept, rng=rng)
        y_latent = X @ beta + rng.normal(0, np.sqrt(noise_var), args.test)
        
        np.save(f"{folder}/X_test.npy", X)
        np.save(f"{folder}/y_latent_test.npy", y_latent)
    

if __name__ == "__main__":
    main()