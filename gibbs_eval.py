import numpy as np
import argparse
from scipy.stats import truncnorm, invgamma
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from tqdm.auto import tqdm
from scipy.special import expit, logit
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.stats import linregress
from sklearn.metrics import matthews_corrcoef, r2_score
import aux_functions as aux_f

def cond_hdi(beta_samples, gamma_samples, prob=0.95):

    n_iter, d = beta_samples.shape

    cond_hdi = np.full((d, 2), np.nan)

    for j in range(d):
        mask = gamma_samples[:, j] == 1
        if mask.sum() > 1: 
            cond_hdi[j] = az.hdi(beta_samples[mask, j], prob=prob)

    return cond_hdi

def hdi(samples, prob=0.95):
    samples = np.asarray(samples)
    if samples.ndim == 1:
        return az.hdi(samples, prob=prob)
    n_iter, d = samples.shape
    result = np.empty((d, 2))
    for j in range(d):
        result[j] = az.hdi(samples[:, j], prob=prob)
    return result


def compute_ess(samples):
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]          # (1, n_samples)
        return az.ess(samples, chain_axis=0, draw_axis=1)
    # samples: (n_samples, d) -> dodaj wymiar chain
    samples = samples[np.newaxis, :, :]            # (1, n_samples, d)
    return az.ess(samples, chain_axis=0, draw_axis=1)



def gibbs_eval(samples, X_train, y_latent_train, time, args, model_name = "gibbs"):

    #simulation parameters
  
    input_folder = args.input_folder
    output_folder = args.output_folder
    burn_in = args.burn_in
    n_iter = args.n_iter

    #data
    n, d = X_train.shape
    l, u,  sigma_y_true = list(np.load(f"{input_folder}/l_u_sigma.npy"))

    try:
        X_test, y_latent_test = np.load(f"{input_folder}/X_test.npy"), np.load(f"{input_folder}/y_latent_test.npy")
    except:
        X_test, y_latent_test = None, None

    #---- estimates -----------------------

    samples = {name: chain[burn_in:] for name, chain  in samples.items()}

    gamma_b, beta_b, sigma_b = samples["gamma"], samples["beta"], samples["sigma"]
    
    pip = np.mean(gamma_b, axis=0)

    beta_mean = np.mean(beta_b, axis=0)
    beta_std = np.std(beta_b, axis=0)
    beta_cond_hdi = cond_hdi(beta_b, gamma_b, prob = args.hdi)

    sigma_mean = sigma_b.mean()
    sigma_std = sigma_b.std()
    sigma_hdi = hdi(sigma_b, prob = args.hdi)

    sigma_ess = compute_ess(sigma_b)
    beta_ess = compute_ess(beta_b)       
    gamma_ess = compute_ess(gamma_b.astype(float))  

    
    #computing conditional beta stats
    
    n_included = gamma_b.sum(axis=0)   
    masked = np.where(gamma_b == 1, beta_b, np.nan)

    with np.errstate(invalid="ignore"):
        beta_cond_mean = np.nanmean(masked, axis=0)
        beta_cond_std = np.nanstd(masked, axis=0)
    beta_cond_mean[n_included == 0] = np.nan
    beta_cond_std[n_included <= 1] = np.nan

    beta_true = np.load(f"{input_folder}/beta.npy")
    beta_stats_df = pd.DataFrame({
        "var_idx": np.arange(d),
        "pip": pip,
        "true_sig": (beta_true != 0).astype(bool),
        "true_size": beta_true,
        "beta_marg_mean": beta_mean,
        "beta_marg_std": beta_std,
        "beta_cond_mean": beta_cond_mean,
        "beta_cond_std": beta_cond_std,
        "beta_cond_hdi_low": beta_cond_hdi[:, 0],
        "beta_cond_hdi_high": beta_cond_hdi[:, 1],
        "ess": beta_ess,
    })
    
    beta_stats_df.to_csv(f"{output_folder}/{model_name}_{args.seed}_beta.csv", index=False)


    sigma_df = pd.DataFrame([{
        "parameter": "sigma",
        "true_value": sigma_y_true,
        "mean": sigma_mean,
        "std": sigma_std,
        "hdi_low": sigma_hdi[0],
        "hdi_high": sigma_hdi[1],
        "ess": sigma_ess,
    }])
    
    sigma_df.to_csv(f"{output_folder}/{model_name}_{args.seed}_sigma.csv", index=False)


    # --- stats -------------------------------------------------------------------
    pip_tr = [0.1, 0.5, 0.9, 0.95]

    true_sig = (beta_true !=0).astype(int)
    df = aux_f.stats(true_sig, pip, pip_tr,
               beta_true, beta_cond_mean,
               X_train, y_latent_train,
               X_test, y_latent_test)
    df.to_csv(f"{output_folder}/{model_name}_stats.csv")
    
    # ---- time --------------------------------------------------------------------
    row = {
        "n": n,
        "d": d,
        "mean_sparsity": pip.mean(),
        "total_time": time["total"],
        "total_fit_time": time["fit"],
        "n_iters_with_burn_in": n_iter + burn_in,
        "gamma_fit_time": time["gamma"],
        "gamma_batch_size": args.gamma_batch,
        "seed": args.seed,
    }
    
    pd.DataFrame([row]).to_csv(f"{output_folder}/{model_name}_{args.seed}_time.csv", index=False)
    