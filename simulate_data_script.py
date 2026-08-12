import argparse
import numpy as np


def X_basic(n, d, intercept=True, seed=None):
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, d))
    
    if intercept:
        X = np.hstack([np.ones((n, 1)), X])  # prepend intercept column

    return X

def X_corr_blocks(n, d, k, corr, intercept=True, seed=None):

    rng = np.random.default_rng(seed)

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

def X_corr_diag(n, d, k, corr, intercept=True, seed=None):
    rng = np.random.default_rng(seed)
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


def X_ar(n, d, k, corr, intercept=False, seed=None):
    
    rng = np.random.default_rng(seed)

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


def X_design(n, d, struct, k =10, corr = 0.7, intercept= False, seed=None):

    if struct == "basic":
        return X_basic(n, d, intercept, seed)
        
    elif struct == "corr_blocks":
        return X_corr_blocks(n, d, k, corr, intercept, seed)
   
    elif struct == "corr_diag":
        return X_corr_diag(n, d, k, corr, intercept, seed)

    elif struct == "AR":
        return X_ar(n, d, k, corr, intercept, seed)

    else:
        print("Unknown type of X_design")


def beta_sparse(d, beta_scale, perc, seed = None):
    
    rng = np.random.default_rng(seed)

    beta = beta_scale * rng.standard_normal(d)
    beta *= rng.random(d) < perc
    
    return beta


def y_tobit(X, beta_true, l_perc, u_perc, snr, seed=None):
    
    rng = np.random.default_rng(seed)
    n, d = X.shape
    
    signal = X @ beta_true
    signal_var = np.var(signal)

    sigma_y_true = np.sqrt(signal_var / snr)
    
    y_latent = signal + rng.normal(0, sigma_y_true, n)
    l, u = np.percentile(y_latent, l_perc), np.percentile(y_latent, u_perc)
    
    return y_latent, l, u, sigma_y_true


def main():
    parser = argparse.ArgumentParser(description="Simulate Tobit data. Creates disagn matrix X, dependent latent variable y^*, censored variable y. Returns all of those arrays and vector of true betas")

    #X_design
    parser.add_argument("-n", type=int, required=True, help="n smples")
    parser.add_argument("-d", type=int, required=True, help="d dimensions")
    parser.add_argument("-X_structure", type=str, required=True, help="type of simulated X: 'basic', 'corr_blocks', 'diagonal', 'AR'")
    parser.add_argument("--k", type=int, required=False, default = 10, help="Size of correlated blocks / width of the correlated diagonal")
    parser.add_argument("--corr", type=float, required=False, default = 0.7, help="strength of correlation of predictors")
    parser.add_argument("--intercept", type=bool, required=False, default = False, help="inclusion of the intercept: True/False")
    
    #y_censored
    parser.add_argument("-l_perc", type=float, required=True, help="lower percentile censoring threshold")
    parser.add_argument("-u_perc", type=float, required=True, help="upper percentile censoring threshold")

    #betas
    parser.add_argument("-snr", type=float, required=True, help="signal to noise ratio")
    parser.add_argument("--tau2", type=float, required=False, default = 100.0, help="variance of the true significant effects")
    parser.add_argument("--pi0", type=float, required=False, default = 0.1, help="percentage of the true significant effects")
    
    #other settings
    parser.add_argument("--seed", type=int, required=False, default = None, help="random seed")
    parser.add_argument("--folder", type=str, required=False, default = "Jupyter/vi_tobit/simulations", help="path to save results")
    
    
    args = parser.parse_args()

    assert args.X_structure in ['basic', 'corr_blocks', 'diagonal', 'AR']
    assert (args.corr <= 1) and (args.corr >= 0)
    assert (args.snr <= 1) and (args.snr >= 0)
    assert (args.pi0 < 1) and (args.pi0 > 0)
    assert args.tau2 > 0
    assert args.l_perc < args.u_perc
    assert (args.l_perc >= 0) and (args.u_perc <= 100)

    n, d = args.n, args.d
    seed = args.seed
    folder = args.folder
    

    X = X_design(n, d, args.X_structure, k = args.k, corr = args.corr, intercept=args.intercept, seed=seed)
        
    beta = beta_sparse(d, args.tau2, args.pi0, seed = seed)

    y_latent, l, u, sigma_y_true = y_tobit(X, beta, args.l_perc, args.u_perc, args.snr, seed)
    
    np.save(f"{folder}/X.npy", X)
    np.save(f"{folder}/beta.npy", beta)
    np.save(f"{folder}/y_latent.npy", y_latent)
    np.save(f"{folder}/l_u_sigma.npy", np.array([l, u, sigma_y_true]))

    # Save all arguments (including defaults) as a copy-paste-ready command
    flag_map = {
        "n": "-n", "d": "-d", "X_structure": "-X_structure",
        "k": "--k", "corr": "--corr", "intercept": "--intercept",
        "l_perc": "-l_perc", "u_perc": "-u_perc",
        "snr": "-snr", "tau2": "--tau2", "pi0": "--pi0",
        "seed": "--seed", "folder": "--folder",
    }
    args_dict = vars(args)
    command_parts = ["python simulate_data_script.py"]
    for key, value in args_dict.items():
        command_parts.append(f"{flag_map[key]} {value}")
    command_str = " \\\n    ".join(command_parts)

    with open(f"{folder}/args.txt", "w") as f:
        f.write(command_str + "\n")

if __name__ == "__main__":
    main()