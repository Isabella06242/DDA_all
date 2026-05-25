import scipy.io
import pandas as pd
import numpy as np

def get_brain_regions(mat_file):
    """Extract brain region information from setup .mat file"""
    mat = scipy.io.loadmat(mat_file)
    regions = mat['channel_region']

    # Convert to a more usable format
    region_dict = {}
    for i in range(regions.shape[1]):
        # MATLAB uses 1-based indexing, so channel i+1 corresponds to index i
        channel_num = i + 1
        region_name = str(regions[0, i][0])  # Extract string from numpy array
        region_dict[channel_num] = region_name

    return region_dict

def main():
    # Load AUC data
    df025 = pd.read_csv('DDA_Intro/py/tau_sweep_results/tau35_34_patientEMU025_results.csv')
    df038 = pd.read_csv('DDA_Intro/py/tau_sweep_results/tau35_34_patientEMU038_results.csv')

    # Get top 5 channels by test_auc for each patient
    top5_025 = df025.nlargest(5, 'test_auc')[['channel', 'test_auc']]
    top5_038 = df038.nlargest(5, 'test_auc')[['channel', 'test_auc']]

    # Load brain region data
    regions_025 = get_brain_regions('maybritt/brain_region/EMU025_MAD_SES1_Setup_bipolar.mat')
    regions_038 = get_brain_regions('maybritt/brain_region/EMU038_MAD_SES1_Setup_bipolar.mat')

    # Match channels to regions
    results = []

    print("=== EMU025 Top 5 Channels ===")
    for _, row in top5_025.iterrows():
        channel = int(row['channel'])
        auc = row['test_auc']
        region = regions_025.get(channel, 'Unknown')
        results.append({
            'patient': 'EMU025',
            'channel': channel,
            'test_auc': auc,
            'brain_region': region
        })
        print(f"Channel {channel}: AUC={auc:.4f}, Region={region}")

    print("\n=== EMU038 Top 5 Channels ===")
    for _, row in top5_038.iterrows():
        channel = int(row['channel'])
        auc = row['test_auc']
        region = regions_038.get(channel, 'Unknown')
        results.append({
            'patient': 'EMU038',
            'channel': channel,
            'test_auc': auc,
            'brain_region': region
        })
        print(f"Channel {channel}: AUC={auc:.4f}, Region={region}")

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('top5_auc_brain_regions.csv', index=False)
    print(f"\nResults saved to top5_auc_brain_regions.csv")

if __name__ == '__main__':
    main()