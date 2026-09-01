import numpy as np
import argparse
from scipy.stats import truncnorm, invgamma, linregress, norm
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from tqdm.auto import tqdm
from scipy.special import expit, logit
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import matthews_corrcoef, r2_score
import aux_functions as aux_f


def mfvi_eval(summary, X_train, y_latent_train, time, args, model_name = "mfvi"):

    #simulation parameters
    input_folder = args.input_folder
    output_folder = args.output_folder
    burn_in = args.burn_in
    n_iter = args.n_iter

    #data
    n, d = X.shape
    l, u,  sigma_y_true = list(np.load(f"{input_folder}/l_u_sigma.npy"))

    try:
        X_test, y_latent_test = np.load(f"{input_folder}/X_test.npy"), np.load(f"{input_folder}/y_latent_test.npy")
    except:
        X_test, y_latent_test = None, None

    #---- estimates -----------------------

    pip = summary["rho"]

    
    beta_mean = summary["m"]
    beta_std = np.sqrt(summary["S_diag"])
    
    z = norm.ppf(0.5 + args.hdi / 2)

    beta_hdi = np.stack([
        beta_mean - z * beta_std,
        beta_mean + z * beta_std
    ], axis=1)

    a = summary["a"]
    b = summary["b"]
    alpha = 1 - args.hdi

    
    sigma_mean = summary["E_sigma2"]
    sigma_std = np.sqrt(b**2 / ((a - 1)**2 * (a - 2))) if a > 2 else np.nan
    sigma_hdi = invgamma.ppf([alpha / 2, 1 - alpha / 2], a=a, scale=b)

    
    beta_true = np.load(f"{input_folder}/beta.npy")
    
    beta_stats_df = pd.DataFrame({
        "var_idx": np.arange(d),
        "pip": pip,
        "true_sig": (beta_true != 0).astype(bool),
        "true_size": beta_true,
        "beta_mean": beta_mean,
        "beta_std": beta_std,
        "beta_hdi_low": beta_hdi[:, 0],
        "beta_hdi_high": beta_hdi[:, 1]
    })
    
    beta_stats_df.to_csv(f"{output_folder}/effect_sizes/{model_name}_{args.seed}_beta.csv", index=False)


    sigma_df = pd.DataFrame([{
        "parameter": "sigma",
        "true_value": sigma_y_true,
        "mean": sigma_mean,
        "std": sigma_std,
        "hdi_low": sigma_hdi[0],
        "hdi_high": sigma_hdi[1]
    }])
    
    sigma_df.to_csv(f"{output_folder}/{model_name}_{args.seed}_sigma.csv", index=False)


    # --- stats -------------------------------------------------------------------
    pip_tr = [0.1, 0.5, 0.9, 0.95]

    true_sig = (beta_true !=0).astype(int)
    df = stats(true_sig, pip, pip_tr,
               beta_true, beta_cond_mean,
               X_train, y_latent_train,
               X_test, y_latent_test)
    df.to_csv(f"{output_folder}/{model_name}_stats.csv")
    
    # ---- time --------------------------------------------------------------------
    row = {
        "n": n,
        "d": d,
        "mean_sparsity": pip.mean(),
        "n_iters": summary["n_iters",
        "coverged": summary["covergence"],
        "total_time": time["total"],
        "total_fit_time": time["fit"],
        "gamma_fit_time": time["gamma"],
        "gamma_batch_size": args.gamma_batch,
        "seed": args.seed,
    }
    
    pd.DataFrame([row]).to_csv(f"{output_folder}/{model_name}_{args.seed}_time.csv", index=False)
    