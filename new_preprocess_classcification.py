import pandas as pd
import numpy as np
from pathlib import Path

# Load and combine all DDA coefficient CSVs
csv_folder = Path("channel_182_parameters")
csv_files = sorted(csv_folder.glob("dda_coefficients_timeseries_*.csv"),
                   key=lambda p: int(p.stem.split('_')[-1]))

rows = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    avg = df[['a1', 'a2', 'a3', 'RMSE']].mean()
    rows.append(avg)

features_df = pd.DataFrame(rows)

# Attach labels
full_single_opt_times = pd.read_csv("full_single_opt_info_align_times.csv", header=None)[0].values
single_opt_first_times = pd.read_csv("single_opt_first_align_times.csv", header=None)[0].values

n_full   = len(full_single_opt_times)
n_single = len(single_opt_first_times)

features_df['event_type'] = ['full-single-opt'] * n_full + ['single-opt-first'] * n_single

# Save
output_dir = Path("classifier_ready_data")
output_dir.mkdir(exist_ok=True)
features_df.to_csv(output_dir / "merged_dda_features.csv", index=False)

print(f"Saved {len(features_df)} events: {n_full} full-single-opt, {n_single} single-opt-first")