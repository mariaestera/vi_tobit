def main():

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

   
    burn_in = 2000
    n_iters = 4000
    
    folder = "Jupyter/vi_tobit/simulations/diagn_scenario_4"
    data_folder = "Jupyter/vi_tobit/simulations/X_design/diagn_scenario_4"
    scenario_nr = folder[-1]
    
    subfolders = [f.name for f in os.scandir(folder) if f.is_dir()]
    subfolders = [f for f in subfolders if f not in  ['.ipynb_checkpoints']]
    
    print("Number of available simulations:", len(subfolders))
    
    beta_chains = [np.load(f"{folder}/{s}/beta_chain.npy")[:burn_in + n_iters] for s in subfolders]
    print("beta")
    gamma_chains = [np.load(f"{folder}/{s}/gamma_chain.npy")[:burn_in + n_iters] for s in subfolders]
    sigma_chains = [np.load(f"{folder}/{s}/sigma_chain.npy")[:burn_in + n_iters] for s in subfolders]
    g_chains = [np.load(f"{folder}/{s}/log_posterior_chain.npy")[:burn_in + n_iters] for s in subfolders]
    beta_true = [np.load(f"{data_folder}_{s}/beta.npy") for s in subfolders][0]
    sigma_true = [ list(np.load(f"{data_folder}_{s}/l_u_sigma.npy"))[2] for s in subfolders][0]
    
    post_burn_gamma = [chain[burn_in:burn_in + n_iters] for chain in gamma_chains]
    
    if burn_in + n_iters > min([len(chain) for chain in gamma_chains]):
        print("Chains to short!")
    
    pip_list = [chain.mean(axis=0) for chain in post_burn_gamma]
    
    def pick_index(mask, target):
        
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            return int(np.argmin(np.abs(pip - target)))
        
        return int(candidates[np.argmin(np.abs(pip[candidates] - target))])
        
    
    for i in range(min(len(subfolders),4)):
        
        pip = pip_list[i]
        
        idx_low = pick_index(pip < 0.1, target=0.05)
        idx_mid = pick_index((pip >= 0.2) & (pip <= 0.8), target=0.5)
        idx_high = pick_index(pip > 0.9, target=0.95)
    
        beta_idx = [idx_low, idx_mid, idx_high]
    
        fig, axes = plt.subplots(1,3, figsize=(15,3))
        
        for ax, idx in zip(axes, beta_idx):
    
            trace = beta_chains[i][:, idx][0:n_iters+burn_in]
            
            post_mean = trace[burn_in:].mean()
            
            ax.plot(trace, lw=0.6, color="#FFC400", label="sample trace")
            ax.axhline(beta_true[idx], color="black", ls="--", lw=1.5, label="ground truth")
            ax.axhline(post_mean, color="red", ls="-", lw=1.5, label="posterior mean")
            ax.axvline(burn_in, color="grey", ls=":", lw=1.5, label="burn-in cutoff")
            ax.set_title(rf"$\beta_{{{idx}}}$ = {beta_true[idx]:.2f}, $\hat \beta_{{{idx}}}$ = {post_mean:.2f}, PIP = {pip[idx]:.2f}", fontsize=10)
            ax.set_xlabel("iteration")
            ax.set_ylabel(r"$\beta$")
            
        axes[2].legend(fontsize=7, loc='upper right')
        
        fig.suptitle(rf"Trace plots: $\beta$, scenario {scenario_nr}", fontsize=13)
        fig.tight_layout()
        fig.savefig(f"Jupyter/vi_tobit/viz/trace_plots/scenario_{scenario_nr}/trace_plot_beta_{subfolders [i]}.png", dpi=150)
        
        if False:
            plt.show()
        else:
            plt.close()
        
    
    
        trace_sigma = sigma_chains[i][0:n_iters+burn_in]
        post_mean_sigma = trace_sigma[burn_in:].mean()
    
        fig_sigma, ax_sigma = plt.subplots(figsize=(6, 4))
        ax_sigma.plot(trace_sigma, lw=0.6, color="#1AC938", label="sample trace")
        ax_sigma.axhline(sigma_true, color="black", ls="--", lw=1.5, label="ground truth")
        ax_sigma.axhline(post_mean_sigma, color="crimson", ls="-", lw=1.5, label="posterior mean")
        ax_sigma.axvline(burn_in, color="grey", ls=":", lw=1.5, label="burn-in cutoff")
        ax_sigma.set_title(rf"$\sigma_{{y^*}}$ = {sigma_true:.2f}, $\hat \sigma_{{y^*}}$ = {post_mean_sigma:.2f}", fontsize=10)
        ax_sigma.set_xlabel("iteration")
        ax_sigma.set_ylabel(r"$\sigma_{y^*}$")
        ax_sigma.legend(fontsize=7, loc='upper right')
    
        fig_sigma.suptitle(rf"Trace plot: $\sigma_{{y^*}}$, scenario {scenario_nr}", fontsize=13)
        fig_sigma.tight_layout()
        fig_sigma.savefig(f"Jupyter/vi_tobit/viz/trace_plots/scenario_{scenario_nr}/trace_plot_sigma_{subfolders[i]}.png", dpi=150)
    
        if False:
            plt.show()
        else:
            plt.close()
    
        trace_log_posterior = g_chains[i][0:n_iters+burn_in] /(-2)
        post_mean_log_posterior = trace_log_posterior[burn_in:].mean()
    
        fig_g, ax_g = plt.subplots(figsize=(6, 4))
        ax_g.plot(trace_log_posterior, lw=0.6, color="steelblue", label="sample trace")
        ax_g.axhline(post_mean_log_posterior, color="crimson", ls="-", lw=1.5, label="posterior mean")
        ax_g.axvline(burn_in, color="grey", ls=":", lw=1.5, label="burn-in cutoff")
        ax_g.set_ylim(trace_log_posterior[burn_in:].min() - 20, trace_log_posterior[burn_in:].max() + 20)
        ax_g.set_title(rf"$\hat{{\log p}}$ = {post_mean_log_posterior:.2f}", fontsize=10)
        ax_g.set_xlabel("iteration")
        ax_g.set_ylabel("log posterior")
        ax_g.legend(fontsize=7, loc='lower right')
    
        fig_g.suptitle(f"Trace plot: logartitm of the posterior density, scenario {scenario_nr}", fontsize=13)
        fig_g.tight_layout()
    
        fig_g.savefig(f"Jupyter/vi_tobit/viz/trace_plots/scenario_{scenario_nr}/trace_plot_log_posterior_{subfolders[i]}.png", dpi=150)
    
        if False:
            plt.show()
        else:
            plt.close()
    
    import numpy as np
    from scipy.stats import norm
    
    def running_mean_ci(x, confidence=0.95):
        """Running mean with a normal-approximation confidence interval half-width."""
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
    
        # Two-sided interval: e.g. 0.95 -> 97.5th percentile of N(0, 1)
        z = norm.ppf(0.5 + confidence / 2)
    
        x = np.asarray(x, dtype=float)
        n = np.arange(1, len(x) + 1)
        running_mean = np.cumsum(x) / n
        running_var = np.cumsum(x ** 2) / n - running_mean ** 2
        running_var = np.clip(running_var, 0, None)  # guard against tiny negative values from rounding
        se = np.sqrt(running_var / n)
        return running_mean, z * se
    
    iters = range(n_iters + burn_in)
    for i in range(min(len(subfolders),4)):
    
        pip = pip_list[i]
    
        idx_low = pick_index(pip < 0.1, target=0.05)
        idx_mid = pick_index((pip >= 0.2) & (pip <= 0.8), target=0.5)
        idx_high = pick_index(pip > 0.9, target=0.95)
    
        beta_idx = [idx_low, idx_mid, idx_high]
    
        # constrained layout automatically reserves space for suptitle, titles and labels
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    
        for ax, idx in zip(axes, beta_idx):
    
            trace = beta_chains[i][:, idx][0:n_iters + burn_in]
            rmean, ci = running_mean_ci(trace)
    
            ax.plot(iters, rmean, color="steelblue", lw=1.2, label="running mean")
            ax.fill_between(iters, rmean - ci, rmean + ci, color="steelblue",
                            alpha=0.25, label="95% CI")
            ax.axhline(beta_true[idx], color="black", ls="--", lw=1.5, label="ground truth")
            ax.axvline(burn_in, color="grey", ls=":", lw=1.5, label="burn-in cutoff")
            ax.set_title(
                rf"$\beta_{{{idx}}}$ = {beta_true[idx]:.2f}, "
                rf"$\hat \beta_{{{idx}}}$ = {post_mean:.2f}, PIP = {pip[idx]:.2f}",
                fontsize=10,
            )
            ax.set_xlabel("iteration")
            ax.set_ylabel(r"$\beta$")
        axes[2].legend(fontsize=6)
    
        fig.suptitle(r"Running mean of $\beta$ $\pm$ 95% CI")
        fig.savefig(f"Jupyter/vi_tobit/viz/trace_plots/scenario_{scenario_nr}/running_mean_plot_beta_{subfolders [i]}.png", dpi=150)
        
        if i == 0:
            plt.show()
        else:
            plt.close()
    
        trace_sigma = sigma_chains[i][0:n_iters+burn_in]
        rmean_s, ci_s = running_mean_ci(trace_sigma)
        
        plt.plot(iters, rmean_s, color="darkorange", lw=1.2, label="running mean")
        plt.fill_between(iters, rmean_s - ci_s, rmean_s + ci_s, color="darkorange",
                         alpha=0.25, label="95% CI")
        plt.axhline(sigma_true, color="black", ls="--", lw=1.5, label="ground truth")
        plt.axvline(burn_in, color="grey", ls=":", lw=1.5, label="burn-in cutoff")
        plt.title(r"$\sigma_y$", fontsize=9)
        plt.xlabel("iteration")
        plt.ylabel(r"$\sigma_y$")
        plt.legend(fontsize=6, loc='upper right')
        plt.savefig(f"Jupyter/vi_tobit/viz/trace_plots/scenario_{scenario_nr}/running_mean_plot_sigma_{subfolders [i]}.png", dpi=150)
        if i == 0:
            plt.show()
        else:
            plt.close()
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    def plot_split_running_mean(ax, trace, burn_in, color, ci_label="95% CI"):
        """Plot running mean +- CI separately for burn-in (grey) and post burn-in (colored).
    
        Returns the final running mean of the post burn-in part (posterior mean estimate).
        """
        trace = np.asarray(trace, dtype=float)
        iters = np.arange(len(trace))
    
        # Split the chain and compute running statistics independently for each part
        rmean_b, ci_b = running_mean_ci(trace[:burn_in])
        rmean_p, ci_p = running_mean_ci(trace[burn_in:])
        iters_b, iters_p = iters[:burn_in], iters[burn_in:]
    
        # Burn-in phase: grey
        ax.plot(iters_b, rmean_b, color="grey", lw=1.2, label="running mean (burn-in)")
        ax.fill_between(iters_b, rmean_b - ci_b, rmean_b + ci_b, color="grey",
                        alpha=0.25, label=f"{ci_label} (burn-in)")
    
        # Post burn-in phase: colored
        ax.plot(iters_p, rmean_p, color=color, lw=1.2, label="running mean (post burn-in)")
        ax.fill_between(iters_p, rmean_p - ci_p, rmean_p + ci_p, color=color,
                        alpha=0.25, label=f"{ci_label} (post burn-in)")
    
        ax.axvline(burn_in, color="black", ls=":", lw=1.5, label="burn-in cutoff")
        return rmean_p[-1]
    
    
    for i in range(min(len(subfolders),4)):
    
        pip = pip_list[i]
    
        idx_low = pick_index(pip < 0.1, target=0.05)
        idx_mid = pick_index((pip >= 0.2) & (pip <= 0.8), target=0.5)
        idx_high = pick_index(pip > 0.9, target=0.95)
    
        beta_idx = [idx_low, idx_mid, idx_high]
    
        # constrained layout automatically reserves space for suptitle, titles and labels
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    
        for ax, idx in zip(axes, beta_idx):
    
            trace = beta_chains[i][:, idx][0:n_iters + burn_in]
            post_mean = plot_split_running_mean(ax, trace, burn_in, color="steelblue")
    
            ax.axhline(beta_true[idx], color="black", ls="--", lw=1.5, label="ground truth")
            ax.set_title(
                rf"$\beta_{{{idx}}}$ = {beta_true[idx]:.2f}, "
                rf"$\hat \beta_{{{idx}}}$ = {post_mean:.2f}, PIP = {pip[idx]:.2f}",
                fontsize=10,
            )
            ax.set_xlabel("iteration")
            ax.set_ylabel(r"$\beta$")
        axes[2].legend(fontsize=6)
    
        fig.suptitle(r"Running mean of $\beta$ $\pm$ 95% CI (burn-in vs. post burn-in)")
        fig.savefig(f"Jupyter/vi_tobit/viz/trace_plots/scenario_{scenario_nr}/running_mean_splitted_plot_beta_{subfolders[i]}.png", dpi=150)
    
        if i == 0:
            plt.show()
        else:
            plt.close(fig)
    
        # Sigma trace: same split, single panel
        trace_sigma = sigma_chains[i][0:n_iters + burn_in]
    
        fig_s, ax_s = plt.subplots(figsize=(6, 4), layout="constrained")
        plot_split_running_mean(ax_s, trace_sigma, burn_in, color="darkorange")
        ax_s.axhline(sigma_true, color="black", ls="--", lw=1.5, label="ground truth")
        ax_s.set_title(r"$\sigma_y$", fontsize=9)
        ax_s.set_xlabel("iteration")
        ax_s.set_ylabel(r"$\sigma_y$")
        ax_s.legend(fontsize=6)
        fig_s.savefig(f"Jupyter/vi_tobit/viz/trace_plots/scenario_{scenario_nr}/running_mean_splitted_plot_sigma_{subfolders [i]}.png", dpi=150)
        
        if i == 0:
            fig_s.show()
        else:
            plt.close()
    
    # from scipy.linalg import eig
    
    # """
    # Compute the multivariate PSRF (MPSRF, Brooks & Gelman 1998) for the beta
    # chain produced by SparseTobitGibbs, based on several parallel chains
    # started from different seeds.
    
    # R_p = (n-1)/n + (1 + 1/m) * lambda1
    
    # where lambda1 is the largest eigenvalue of the generalized eigenvalue
    # problem  B* v = lambda * W* v
    # (W* = within-chain covariance matrix, B*/n = between-chain-mean covariance
    # matrix, as defined in Brooks & Gelman 1998 / Roy 2020 review, Eq. 8).
    # """
    
    # def compute_mpsrf(chains):
    #     """
    #     chains: list of m arrays, each of shape (n, p) -- same n and p for all chains.
    #     Returns the scalar MPSRF (R_p).
    #     """
    #     m = len(chains)
    #     n_len = chains[0].shape[0]
    #     p = chains[0].shape[1]
    
    #     # within-chain covariance W* : average of per-chain sample covariances
    #     chain_covs = [np.cov(chain, rowvar=False, ddof=1) for chain in chains]
    #     W_star = np.mean(chain_covs, axis=0)  # (p, p)
    
    #     # between-chain-mean covariance B*/n : covariance of the m chain means
    #     chain_means = np.array([chain.mean(axis=0) for chain in chains])  # (m, p)
    #     B_over_n = np.cov(chain_means, rowvar=False, ddof=1)  # (p, p)
    
    #     # generalized eigenvalue problem: B*/n * v = lambda * W* * v
    #     eigvals, _ = eig(B_over_n, W_star)
    #     lambda1 = np.max(eigvals.real)
    
    #     R_p = (n_len - 1) / n_len + (1 + 1 / m) * lambda1
    #     return R_p, lambda1
    
    
    beta_chains_post = [chain[burn_in:burn_in+n_iters] for chain in beta_chains]
    
    n_chains = len(beta_chains_post)
    d = beta_chains_post[0].shape[1]
    
    # mpsrf, lambda1 = compute_mpsrf(beta_chains_post)
    
    # converged = mpsrf < 1.1
    
    # print("\n--- MPSRF results ---")
    # print(f"Number of chains (m): {n_chains}")
    # print(f"Samples per chain after burn-in: {beta_chains_post[0].shape[0]}")
    # print(f"Dimension of beta (d): {d}")
    # print(f"Largest eigenvalue (lambda1): {lambda1:.4f}")
    # print(f"MPSRF (R_p): {mpsrf:.4f}")
    
    # if converged:
    #     print("MPSRF < 1.1: no strong evidence against convergence (joint check).")
    # else:
    #     print("MPSRF >= 1.1: chains have likely not converged jointly yet.")
    
    # # Save results as one row in MPSRF.csv (create the file with a header if missing)
    # results = {
    #     "scenario": scenario_nr,
    #     "n_burn_in": burn_in,
    #     "n_post_burn_in": n_iters,
    #     "n_chains": n_chains,
    #     "samples_per_chain": beta_chains_post[0].shape[0],
    #     "dim_beta": d,
    #     "lambda1": lambda1,
    #     "mpsrf": mpsrf,
    #     "converged": converged,
    # }
    
    # csv_path = "Jupyter/vi_tobit/viz/MPSRF.csv"
    # row = pd.DataFrame([results])
    # file_exists = os.path.isfile(csv_path)
    # row.to_csv(csv_path, mode="a", header=not file_exists, index=False)
    
    
    def per_variable_psrf(chains):
        """
        chains: list of m arrays, each (n, d).
        Returns array of shape (d,) with the univariate PSRF of each column,
        computed across the m chains.
        - constant in all chains with identical value -> nan (PSRF undefined)
        - zero within-chain variance but different chain means -> inf
        """
        n_len = chains[0].shape[0]
    
        chain_means = np.array([c.mean(axis=0) for c in chains])        # (m, d)
        chain_vars = np.array([c.var(axis=0, ddof=1) for c in chains])  # (m, d)
    
        W = chain_vars.mean(axis=0)                  # within-chain variance
        B_over_n = chain_means.var(axis=0, ddof=1)   # between-chain-mean variance
        V_hat = ((n_len - 1) / n_len) * W + B_over_n
    
        with np.errstate(divide="ignore", invalid="ignore"):
            r_hat = np.sqrt(V_hat / W)
    
        # Handle degenerate coordinates explicitly
        degenerate = W == 0
        r_hat[degenerate & (B_over_n == 0)] = np.nan  # constant everywhere, undefined
        r_hat[degenerate & (B_over_n > 0)] = np.inf   # chains stuck at different values
        return r_hat
    
    
    def summarize(psrf, name):
        """Min / max / median over coordinates, ignoring undefined (nan) ones."""
        finite_or_inf = psrf[~np.isnan(psrf)]
        return {
            f"{name}_psrf_min": np.min(finite_or_inf),
            f"{name}_psrf_max": np.max(finite_or_inf),
            f"{name}_psrf_median": np.median(finite_or_inf),
            f"{name}_n_undefined": int(np.isnan(psrf).sum()),
        }
    
    
    # Post burn-in chains (same window for every chain)
    beta_post = [c[burn_in:burn_in + n_iters] for c in beta_chains]
    gamma_post = [c[burn_in:burn_in + n_iters].astype(float) for c in gamma_chains]
    sigma_post = [np.asarray(c[burn_in:burn_in + n_iters], dtype=float).reshape(-1, 1)
                  for c in sigma_chains]
    
    # Marginal PSRF across chains (one value per coordinate)
    psrf_beta = per_variable_psrf(beta_post)
    psrf_gamma = per_variable_psrf(gamma_post)
    psrf_sigma = per_variable_psrf(sigma_post)[0]
    
    summary = {}
    summary.update(summarize(psrf_beta, "beta"))
    summary.update(summarize(psrf_gamma, "gamma"))
    
    print("\n--- Marginal PSRF results ---")
    print(f"Number of chains (m): {len(beta_post)}")
    print(f"Samples per chain after burn-in: {beta_post[0].shape[0]}")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"sigma_psrf: {psrf_sigma:.4f}")
    
    # One row per seed; sigma posterior mean differs per chain, PSRF stats are shared
    rows = []
    for i, s in enumerate(subfolders):
        row = {
            "scenario": scenario_nr,
            "n_burn_in": burn_in,
            "n_post_burn_in": n_iters,
            "seed": s,
            "samples_per_chain": beta_post[0].shape[0],
            "sigma_post_mean": sigma_post[i].mean(),
            "sigma_psrf": psrf_sigma,
        }
        row.update(summary)
        rows.append(row)
    
    csv_path = f"Jupyter/vi_tobit/viz/PSRF_scenario_{scenario_nr}.csv"
    file_exists = os.path.isfile(csv_path)
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not file_exists, index=False)



    def count_above_threshold(psrf, threshold=1.1):
        """Count of coordinates with PSRF above threshold, ignoring undefined (nan) ones."""
        finite = psrf[~np.isnan(psrf)]
        return int(np.sum(finite > threshold))

    n_above_beta = count_above_threshold(psrf_beta, threshold=1.1)
    n_above_gamma = count_above_threshold(psrf_gamma, threshold=1.1)
    
    # Single run -> min/max/median of "count above 1.1" collapse to the same value,
    # kept as separate columns for consistency with later multi-run aggregation
    overall_row = {
        "scenario": scenario_nr,
        "n_simulations": len(beta_post),
        "burn_in": burn_in,
        "n_iters": n_iters,
        "max_psrf_beta": summary["beta_psrf_max"],
        "max_psrf_gamma": summary["gamma_psrf_max"],
        "n_above_1.1_beta_min": n_above_beta,
        "n_above_1.1_beta_max": n_above_beta,
        "n_above_1.1_beta_median": n_above_beta,
        "n_above_1.1_gamma_min": n_above_gamma,
        "n_above_1.1_gamma_max": n_above_gamma,
        "n_above_1.1_gamma_median": n_above_gamma,
    }
    
    overall_csv_path = "Jupyter/vi_tobit/viz/PSRF.csv"
    overall_file_exists = os.path.isfile(overall_csv_path)
    pd.DataFrame([overall_row]).to_csv(
        overall_csv_path, mode="a", header=not overall_file_exists, index=False
    )
        
    from itertools import combinations
    
    selection_threshold = 0.5
    m =len(subfolders)
    selected_sets = [set(np.where(pip_list[i] > selection_threshold)[0]) for i in range(m)]
    
    print(f"\n--- Jaccard similarity of active-variable sets (PIP > {selection_threshold}) ---")
    
    rows = []
    for i, j in combinations(range(m), 2):
        inter = len(selected_sets[i] & selected_sets[j])
        union = len(selected_sets[i] | selected_sets[j])
        # Two empty sets are treated as identical (Jaccard = 1)
        jaccard = inter / union if union > 0 else 1.0
        print(f"Chain {subfolders[i]} vs {subfolders[j]}: Jaccard = {jaccard:.3f} "
              f"({inter} shared / {union} total active)")
    
        rows.append({
            "scenario": scenario_nr,
            "n_burn_in": burn_in,
            "n_post_burn_in": n_iters,
            "seed_1": subfolders[i],
            "seed_2": subfolders[j],
            "selection_threshold": selection_threshold,
            "n_selected_1": len(selected_sets[i]),
            "n_selected_2": len(selected_sets[j]),
            "n_shared": inter,
            "n_union": union,
            "jaccard": jaccard,
        })
    
    # Save one row per pair of chains (create the file with a header if missing)
    csv_path = f"Jupyter/vi_tobit/viz/jaccard_similarity_scenario_{scenario_nr}.csv"
    file_exists = os.path.isfile(csv_path)
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not file_exists, index=False)
    
    
    # Aggregate summary across all chain pairs (one row per scenario)
    pairs_df = pd.DataFrame(rows)
    n_selected_all = [len(s) for s in selected_sets]  # one value per chain
    d = beta_chains_post[0].shape[1]
    
    summary_row = {
        "scenario": scenario_nr,
        "n_burn_in": burn_in,
        "n_post_burn_in": n_iters,
        "n_chains": m,
        "n_variables": d,
        "n_pairs": len(pairs_df),
        "selection_threshold": selection_threshold,
        "jaccard_min": pairs_df["jaccard"].min(),
        "jaccard_max": pairs_df["jaccard"].max(),
        "jaccard_median": pairs_df["jaccard"].median(),
        "n_selected_min": int(np.min(n_selected_all)),
        "n_selected_max": int(np.max(n_selected_all)),
        "n_selected_median": float(np.median(n_selected_all)),
    }
    
    # Append to the summary file (create with header if missing)
    summary_path = "Jupyter/vi_tobit/viz/jaccard_similarity.csv"
    summary_exists = os.path.isfile(summary_path)
    pd.DataFrame([summary_row]).to_csv(summary_path, mode="a", header=not summary_exists, index=False)
    
    import numpy as np
    
    # Uses beta_chains / gamma_chains already loaded from disk, plus burn_in and n_iters.
    # No model refit is needed.
    
    def autocorr_fft(x):
        """Full autocorrelation function of a 1D series via FFT (nan if the series is constant)."""
        n_len = len(x)
        x = x - x.mean()
        var0 = np.dot(x, x)
        if var0 == 0:
            return None  # constant series, ACF undefined
        fsize = 2 ** int(np.ceil(np.log2(2 * n_len - 1)))
        xf = np.fft.fft(x, fsize)
        acf = np.fft.ifft(xf * np.conjugate(xf)).real[:n_len]
        return acf / acf[0]
    
    
    def ess_1d(x):
        """ESS via Geyer's initial positive sequence (pairs rho_{2t} + rho_{2t+1})."""
        n_len = len(x)
        acf = autocorr_fft(np.asarray(x, dtype=float))
        if acf is None:
            return np.nan
    
        pair_total = 0.0
        k = 0
        while k + 1 < n_len:
            pair_sum = acf[k] + acf[k + 1]
            if pair_sum < 0:
                break
            pair_total += pair_sum
            k += 2
    
        tau = -1 + 2 * pair_total  # integrated autocorrelation time
        return n_len / tau if tau > 0 else np.nan
    
    
    def ess_matrix(chains):
        """ESS for every chain and every coordinate. Returns array of shape (m, d)."""
        m, d = len(chains), chains[0].shape[1]
        out = np.full((m, d), np.nan)
        for i, chain in enumerate(chains):
            for j in range(d):
                out[i, j] = ess_1d(chain[:, j])
        return out
    
    
    def report_ess(all_ess, name, seeds_list, top=10):
        """Print a summary of the ESS matrix, ignoring constant (nan) coordinates."""
        n_undefined = int(np.isnan(all_ess).all(axis=0).sum())
        min_ess_per_var = np.nanmin(all_ess, axis=0)  # worst chain, per variable
        valid = np.where(~np.isnan(min_ess_per_var))[0]
        worst_vars = valid[np.argsort(min_ess_per_var[valid])[:top]]
    
        print(f"\n--- Univariate ESS per {name} component ---")
        print(f"Dimension = {all_ess.shape[1]}, constant (undefined) coordinates: {n_undefined}")
        print(f"{top} variables with lowest ESS (min across chains):")
        for idx in worst_vars:
            print(f"  variable {idx}: min ESS = {min_ess_per_var[idx]:,.1f} "
                  f"(per-chain: {np.round(all_ess[:, idx], 1)})")
        print(f"Overall minimum ESS: {np.nanmin(all_ess):,.1f}")
        print(f"Overall median ESS: {np.nanmedian(all_ess):,.1f}")
        print(f"Overall mean ESS: {np.nanmean(all_ess):,.1f}")
    
    
    # Post burn-in window, same for all chains
    beta_post = [c[burn_in:burn_in + n_iters] for c in beta_chains]
    gamma_post = [c[burn_in:burn_in + n_iters].astype(float) for c in gamma_chains]
    
    print(f"Samples per chain after burn-in: {beta_post[0].shape[0]}")
    
    ess_beta = ess_matrix(beta_post)
    ess_gamma = ess_matrix(gamma_post)
    
    report_ess(ess_beta, "beta", subfolders)
    report_ess(ess_gamma, "gamma", subfolders)
    
    
    import os
    import numpy as np
    import pandas as pd
    
    ess_threshold = 400
    n_samples = beta_post[0].shape[0]
    
    
    def ess_per_seed_table(all_ess, name, seeds_list, n_samples, threshold=ess_threshold):
        """One row per seed: summary over coordinates of an ESS matrix of shape (m, d)."""
        rows = []
        for i, s in enumerate(seeds_list):
            e = all_ess[i][~np.isnan(all_ess[i])]  # drop constant (undefined) coordinates
            rows.append({
                "scenario": scenario_nr,
                "n_burn_in": burn_in,
                "n_post_burn_in": n_iters,
                "seed": s,
                "parameter": name,
                "n_samples": n_samples,
                "n_coords": len(e),
                "ess_min": e.min(),
                "ess_q05": np.quantile(e, 0.05),
                "ess_median": np.median(e),
                "ess_mean": e.mean(),
                "ess_rel_median": np.median(e) / n_samples,
                "n_below_threshold": int((e < threshold).sum()),
            })
        return pd.DataFrame(rows)
    
    
    def ess_general_row(per_seed_df, all_ess, name, threshold=ess_threshold):
        """One row per parameter: aggregate of the per-seed table across seeds."""
        return {
            "scenario": scenario_nr,
            "n_burn_in": burn_in,
            "n_post_burn_in": n_iters,
            "parameter": name,
            "n_seeds": len(per_seed_df),
            "ess_threshold": threshold,
            "overall_min_ess": np.nanmin(all_ess),                 # worst coordinate in worst seed
            "median_of_min_ess": per_seed_df["ess_min"].median(),  # median over seeds of per-seed min
            "mean_q05_ess": per_seed_df["ess_q05"].mean(),         # mean over seeds of per-seed 5th percentile
            "mean_ess": np.nanmean(all_ess),                       # mean over all seeds and coordinates
            "n_below_min": int(per_seed_df["n_below_threshold"].min()),
            "n_below_max": int(per_seed_df["n_below_threshold"].max()),
            "n_below_median": float(per_seed_df["n_below_threshold"].median()),
        }
    
    
    per_seed_beta = ess_per_seed_table(ess_beta, "beta", subfolders, n_samples)
    per_seed_gamma = ess_per_seed_table(ess_gamma, "gamma", subfolders, n_samples)
    per_seed_all = pd.concat([per_seed_beta, per_seed_gamma], ignore_index=True)
    
    general_rows = [
        ess_general_row(per_seed_beta, ess_beta, "beta"),
        ess_general_row(per_seed_gamma, ess_gamma, "gamma"),
    ]
    
    os.makedirs("viz", exist_ok=True)
    
    # Per-seed file (one per scenario, overwritten on rerun)
    per_seed_path = f"Jupyter/vi_tobit/viz/ESS_per_seed_scenario_{scenario_nr}.csv"
    per_seed_all.to_csv(per_seed_path, index=False)
    
    # General file (shared across scenarios, appended; header created if missing)
    general_path = "Jupyter/vi_tobit/viz/ESS_summary.csv"
    general_exists = os.path.isfile(general_path)
    pd.DataFrame(general_rows).to_csv(general_path, mode="a", header=not general_exists, index=False)
    
    print(f"Saved per-seed ESS table to {per_seed_path}")
    print(f"Appended general ESS summary to {general_path}")
    print(pd.DataFrame(general_rows).to_string(index=False))
    
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    def batch_means_var_of_mean(x):
        """Batch-means estimate of the variance of the sample mean (accounts for autocorrelation)."""
        n_len = len(x)
        b = int(np.floor(np.sqrt(n_len)))   # number of batches
        l_ = n_len // b                     # batch length
        n_used = b * l_
        x = x[:n_used]
        batch_means = x.reshape(b, l_).mean(axis=1)
        sigma_hat2 = (l_ / (b - 1)) * np.sum((batch_means - x.mean()) ** 2)
        return sigma_hat2 / n_used
    
    
    def geweke_test(g_t, frac_a=0.1, frac_b=0.5):
        """Geweke Z-score comparing the mean of the first frac_a with the last frac_b of the chain."""
        n_len = len(g_t)
        n_a = int(frac_a * n_len)
        n_b = int(frac_b * n_len)
        g_a = g_t[:n_a]
        g_b = g_t[-n_b:]
        var_a = batch_means_var_of_mean(g_a)
        var_b = batch_means_var_of_mean(g_b)
        z = (g_a.mean() - g_b.mean()) / np.sqrt(var_a + var_b)
        return z, n_a, n_b
    
    
    alpha = 0.05  # significance level (two-sided test)
    
    g_post = [np.asarray(c[burn_in:burn_in + n_iters], dtype=float) for c in g_chains]
    
    # ---------------------------------------------------------------
    # Geweke test per seed (report p-values only)
    # ---------------------------------------------------------------
    print("\n--- Geweke test on log-posterior chain (post burn-in) ---")
    rows = []
    for s, g_t in zip(subfolders, g_post):
        z, n_a, n_b = geweke_test(g_t)          # Z is used only to obtain the p-value
        p_value = 2 * (1 - norm.cdf(abs(z)))
        passed = p_value >= alpha
        print(f"Seed {s}: p = {p_value:.3f}  [{'pass' if passed else f'FAIL (p<{alpha})'}]  "
              f"(nA={n_a}, nB={n_b})")
        rows.append({
            "scenario": scenario_nr,
            "seed": s,
            "n_burn_in": burn_in,
            "n_post_burn_in": n_iters,
            "n_a": n_a,
            "n_b": n_b,
            "p_value": p_value,
            "passed": passed,
        })
    
    geweke_df = pd.DataFrame(rows)
    print(f"\nPassed: {geweke_df['passed'].sum()} / {len(geweke_df)} chains")
    
    os.makedirs("viz", exist_ok=True)
    geweke_df.to_csv(f"Jupyter/vi_tobit/viz/geweke_scenario_{scenario_nr}.csv", index=False)
    
    # ---------------------------------------------------------------
    # Trace plot: only 3 example chains
    # ---------------------------------------------------------------
    n_plot = 3
    fig, axes = plt.subplots(n_plot, 1, figsize=(9, 2.2 * n_plot), sharex=True,
                             layout="constrained")
    for ax, s, g_t, p in zip(axes, subfolders[:n_plot], g_post[:n_plot],
                             geweke_df["p_value"][:n_plot]):
        n_a = int(0.1 * len(g_t))
        n_b = int(0.5 * len(g_t))
        ax.plot(g_t, lw=0.7, color="steelblue")
        # Highlight the two Geweke windows
        ax.axvspan(0, n_a, color="orange", alpha=0.2, label="window A (first 10%)")
        ax.axvspan(len(g_t) - n_b, len(g_t), color="green", alpha=0.15, label="window B (last 50%)")
        ax.set_ylabel(f"seed={s}")
        ax.set_title(f"p = {p:.3f}", fontsize=9, loc="right")
    axes[0].legend(fontsize=6, loc="upper right")
    axes[-1].set_xlabel("iteration (post burn-in)")
    fig.suptitle("Trace of log-posterior with Geweke windows")
    fig.savefig(f"Jupyter/vi_tobit/viz/geweke_trace_scenario_{scenario_nr}.png", dpi=150)
    plt.show()
    
    
    import os
    import pandas as pd
    
    alpha = 0.05
    m_tests = len(geweke_df)
    
    # Bonferroni: each test is compared with alpha / m; the scenario is "ok" if no chain is rejected
    alpha_bonf = alpha / m_tests
    bonferroni_ok = bool((geweke_df["p_value"] >= alpha_bonf).all())
    
    summary_row = {
        "scenario": scenario_nr,
        "burn_in": burn_in,
        "n_iters": n_iters,
        "n_tests": m_tests,
        "pct_passed_alpha_0.05": 100 * (geweke_df["p_value"] >= alpha).mean(),
        "bonferroni_alpha": alpha_bonf,
        "bonferroni_ok": bonferroni_ok,
    }
    
    # Append to the shared file (create with header if missing)
    csv_path = "Jupyter/vi_tobit/viz/geweke.csv"
    os.makedirs("viz", exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    pd.DataFrame([summary_row]).to_csv(csv_path, mode="a", header=not file_exists, index=False)
    print(pd.DataFrame([summary_row]).to_string(index=False))


if __name__ == "__main__":
    main()