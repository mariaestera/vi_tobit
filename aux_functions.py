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


def save_args_command(args, parser, script_name):
    """Save all arguments (including defaults) as a copy-paste-ready command."""
    args_dict = vars(args)
    command_parts = [f"python {script_name}"]

    for action in parser._actions:
        if action.dest == "help" or action.dest not in args_dict:
            continue
        flag = action.option_strings[0]
        value = args_dict[action.dest]
        if value is None:
            continue
        command_parts.append(f"{flag} {value}")

    command_str = " \\\n    ".join(command_parts)
    with open(f"{args.output_folder}/{script_name}-args.txt", "w") as f:
        f.write(command_str + "\n")


def scale_y(y, l, u, sigma_y_true):
    mu_y, sd_y = y.mean(), y.std()
    
    y_scaled = (y - mu_y) / sd_y
    l_scaled = (l - mu_y) / sd_y
    u_scaled = (u - mu_y) / sd_y
    
    sigma_y_scaled = sigma_y_true / sd_y

    return y_scaled, sigma_y_scaled, l_scaled, u_scaled, mu_y, sd_y

def intercept_idx(X):
    ones_cols = np.where(np.all(X == 1, axis=0))[0]

    if len(ones_cols) == 0:
        intercept_idx = None
    elif len(ones_cols) == 1:
        intercept_idx = ones_cols[0]
    else:
        raise ValueError(f"More than one intercept cols found: {ones_cols}")
    
    return intercept_idx



def stats(true_sig, pip, pip_tr,
              beta_true, beta_est,
              X_train, y_latent_train,
              X_test, y_latent_test):
    rows = []

    for tr in pip_tr:
        pred_sig = pip >= tr

        TP = int(np.sum(pred_sig & true_sig))
        TN = int(np.sum(~pred_sig & ~true_sig))
        FP = int(np.sum(pred_sig & ~true_sig))
        FN = int(np.sum(~pred_sig & true_sig))

        precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        accuracy = (TP + TN) / len(true_sig)

        if len(np.unique(pred_sig)) > 1 and len(np.unique(true_sig)) > 1:
            mcc = matthews_corrcoef(true_sig, pred_sig)
        else:
            mcc = np.nan

        active_idx = np.where(pred_sig)[0]
        if len(active_idx) >= 2:
            bt = beta_true[active_idx]
            bc = beta_est[active_idx]

            rmse_beta = np.sqrt(np.mean((bt - bc) ** 2))
            r2_beta = r2_score(bt, bc)
            slope_beta = linregress(bt, bc).slope
        elif len(active_idx) == 1:
            bt = beta_true[active_idx]
            bc = beta_est[active_idx]
            rmse_beta = np.sqrt(np.mean((bt - bc) ** 2))
            r2_beta = np.nan
            slope_beta = np.nan
        else:
            rmse_beta = np.nan
            r2_beta = np.nan
            slope_beta = np.nan

        beta_masked = np.where(pred_sig, beta_est, 0.0)

        y_pred_train = X_train @ beta_masked
        y_pred_test = X_test @ beta_masked if X_test is not None else np.nan

        r2_train = r2_score(y_latent_train, y_pred_train) 
        r2_test = r2_score(y_latent_test, y_pred_test) if X_test is not None else np.nan

        rows.append({
            "pip_tr": tr,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "MCC": mcc,
            "RMSE_beta_active": rmse_beta,
            "R2_beta_active": r2_beta,
            "slope_beta_active": slope_beta,
            "R2_train": r2_train,
            "R2_test": r2_test,
        })

    return pd.DataFrame(rows)