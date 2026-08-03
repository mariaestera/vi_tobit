#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import simulate_data as sim
from mfvb import q_tobit
import evaluation as ev
import gibbs
from gibbs import gibbs_disc_spike_slab


def parse_args():
    parser = argparse.ArgumentParser(description="Script for running Tobit simulation")

    parser.add_argument("--config", type=str, default=None,
                         help="JSON file with parameters")

    # --- basic simulation parameters ---
    parser.add_argument("--seed", type=int, default=42)    
    parser.add_argument("--pip_tr", type=float, default=0.9)
    parser.add_argument("--folder", type=str, default= "Jupyter/vi_tobit")

    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            parser.error(f"Config file not found: {args.config}")
        with open(config_path, "r", encoding="utf-8") as f:
            file_params = json.load(f)

    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("Run parameters:")
    print(vars(args))

    """

    # --- 1. simulate data ---
    # TODO: adjust to match the actual simulate_data API
    data = sim.simulate_data(
        n=args.n,
        d=args.d,
        r2=args.r2,
        perc_sig=args.perc_sig,
        X_design=args.x_design,
        l_perc=args.l_perc,
        u_perc=args.u_perc,
        seed=args.seed,
    )

    # --- 2. inference: MFVB ---
    # TODO: adjust to match the actual q_tobit API
    mfvb_result = q_tobit(
        data,
        tau2_init=args.tau2_init,
        pi0_init=args.pi0_init,
        n_iter=args.n_iter,
    )

    # --- 3. inference: Gibbs ---
    # TODO: adjust to match the actual gibbs_disc_spike_slab API
    gibbs_result = gibbs_disc_spike_slab(
        data,
        tau2_init=args.tau2_init,
        pi0_init=args.pi0_init,
        n_iter=args.n_iter,
        seed=args.seed,
    )

    # --- 4. evaluation ---
    # TODO: adjust to match the actual evaluation API
    metrics = ev.evaluate(
        data=data,
        mfvb_result=mfvb_result,
        gibbs_result=gibbs_result,
        pip_tr=args.pip_tr,
    )
    print("Metrics:")
    print(metrics)

    # --- 5. visualization / saving results (optional) ---
    # fig, ax = plt.subplots()
    # ...
    # plt.savefig("results.png")
    """

if __name__ == "__main__":
    main()