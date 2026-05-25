#!/usr/bin/env python3
"""
Exact Python translation of run_first_DDA.jl
DDA (Delay Differential Analysis) demonstration script
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import platform
from itertools import combinations
import seaborn as sns
from pathlib import Path
from numpy import genfromtxt
import pandas as pd

# Import local functions
from dda_functions import (
    deriv_all,
)
    
# Add command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='DDA analysis with configurable TAU')
    parser.add_argument('--tau1', type=int, default=4, help='First TAU value')
    parser.add_argument('--tau2', type=int, default=6, help='Second TAU value')
    parser.add_argument('--wl', type=int, default=512, help='Window length')
    parser.add_argument('--ws', type=int, default=256, help='Window step')
    parser.add_argument('--dm', type=int, default=4, help='Derivative method')
    return parser.parse_args()

def main():
    print("DEBUG: main() function called")
    args = parse_args()
    print(f"  tau1 = {args.tau1}")
    print(f"  tau2 = {args.tau2}")

    FN_DATA = "channel_182_alignment_activity.csv"
    print(f"data path loaded")

    # Use existing Julia-generated data for exact comparison
    # If you want to generate new data, delete ROS_4.ascii file first
    if not os.path.isfile(FN_DATA):
        print(
            "Error: Input file not found."
        )
        exit(1)

    # Load data
    Y = genfromtxt(FN_DATA, delimiter=',')
    Y = Y[:, 1:]
    Y = Y[Y.shape[0]//2:, :]  # trim to second half (post-alignment), matching MATLAB: x = x(2049:end,:)

    # DDA parameters
    WL = 512
    WS = 256
    TAU = [35, 34]
    dm = 4
    order = 3
    nr_delays = 2
    TM = max(TAU)
    WN = int(1 + np.floor((Y.shape[0] - (WL + TM + 2 * dm - 1)) / WS))

    print(f"Data shape: {Y.shape}")
    print(f"Number of windows: {WN}")

    ########  SingleTimeseries DDA  ########
    Y_single = Y[:, 0]  # First column only

    ST = np.full((WN, 4), np.nan)
    for wn in range(WN):
        anf = wn * WS
        ende = anf + WL + TM + 2 * dm - 1

        data = Y_single[anf : ende + 1]
        ddata = deriv_all(data, dm)
        data = data[dm:-dm]

        STD = np.std(data, ddof=1)
        DATA = (data - np.mean(data)) / STD
        dDATA = ddata / STD

        # CRITICAL FIX: Julia constructs M first, then slices dDATA
        # Julia: M = hcat(DATA[(TM+1:end).-TAU[1]], DATA[(TM+1:end).-TAU[2]], DATA[(TM+1:end).-TAU[1]] .^ 3)
        # Julia indexing: (TM+1:end).-TAU[1] = TM+1-TAU[1]:end-TAU[1]
        # Python 0-based: TM-TAU[0]:len(DATA)-TAU[0]
        M = np.column_stack(
            [
                DATA[TM - TAU[0] : len(DATA) - TAU[0]],  # First delay coordinate
                DATA[TM - TAU[1] : len(DATA) - TAU[1]],  # Second delay coordinate
                (DATA[TM - TAU[0] : len(DATA) - TAU[0]]) ** 2,  # Nonlinear term
            ]
        )

        # Julia: dDATA = dDATA[TM+1:end] (AFTER M construction)
        dDATA_sliced = dDATA[TM:]

        # Use solve instead of lstsq to match Julia's \ operator more closely
        try:
            ST[wn, :3] = np.linalg.solve(
                M.T @ M, M.T @ dDATA_sliced
            )  # Normal equation approach
        except np.linalg.LinAlgError:
            ST[wn, :3] = np.linalg.lstsq(M, dDATA_sliced, rcond=None)[0]
        ST[wn, 3] = np.sqrt(np.mean((dDATA_sliced - M @ ST[wn, :3]) ** 2))


    ###  for all time series
    # Y = np.loadtxt(FN_DATA)  # Reload full data
    # trim data to only the second half
    ST = np.full((WN, 4, Y.shape[1]), np.nan)

    for n_Y in range(Y.shape[1]):
        for wn in range(WN):
            anf = wn * WS
            ende = anf + WL + TM + 2 * dm - 1

            data = Y[anf : ende + 1, n_Y]
            ddata = deriv_all(data, dm)
            data = data[dm:-dm]

            STD = np.std(data, ddof=1)
            DATA = (data - np.mean(data)) / STD
            dDATA = ddata / STD

            # Fixed matrix construction (same as single time series)
            M = np.column_stack(
                [
                    DATA[TM - TAU[0] : len(DATA) - TAU[0]],
                    DATA[TM - TAU[1] : len(DATA) - TAU[1]],
                    (DATA[TM - TAU[0] : len(DATA) - TAU[0]]) ** 2,
                ]
            )

            # Julia: dDATA = dDATA[TM+1:end] (AFTER M construction)
            dDATA_sliced = dDATA[TM:]

            # Use solve instead of lstsq to match Julia's \ operator more closely
            try:
                ST[wn, :3, n_Y] = np.linalg.solve(M.T @ M, M.T @ dDATA_sliced)
            except np.linalg.LinAlgError:
                ST[wn, :3, n_Y] = np.linalg.lstsq(M, dDATA_sliced, rcond=None)[0]
            ST[wn, 3, n_Y] = np.sqrt(np.mean((dDATA_sliced - M @ ST[wn, :3, n_Y]) ** 2))
            
    print(ST[:, 0, 1])

    ## Save the result
    output_dir = Path("channel_182_parameters")
    output_dir.mkdir(exist_ok=True)

    for n in range(Y.shape[1]):
        para_data = {
            'window': range(WN),
            'a1': ST[:, 0, n],
            'a2': ST[:, 1, n], 
            'a3': ST[:, 2, n],
            'RMSE': ST[:, 3, n]
        }

        df = pd.DataFrame(para_data)
        df.to_csv(output_dir / f"dda_coefficients_timeseries_{n+1}.csv", index=False)
        
if __name__ == "__main__":
    main()
