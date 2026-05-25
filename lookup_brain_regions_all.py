"""
Extend top5_auc_brain_regions.csv with brain regions for all remaining patients
in top5_channels_per_patient.csv that haven't been done yet (EMU024, EMU030,
EMU036, EMU037, EMU039, EMU051).

Setup_bipolar.mat files are read from /media/Data/Human_Intracranial_MAD/1_formatted/.
"""

import scipy.io
import pandas as pd
import numpy as np

FORMATTED_DIR = "/media/Data/Human_Intracranial_MAD/1_formatted"
TOP_N = 3  # top N channels per patient (user asked for top 3)


def get_brain_regions(mat_file):
    """
    Extract channel_region mapping {channel_number (1-based) -> region_string}
    from a Setup_bipolar.mat file.
    Works with both squeeze_me=True (1D object array) and raw cell format.
    """
    mat = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    regions_raw = mat['channel_region']

    region_dict = {}
    arr = np.atleast_1d(regions_raw)

    if arr.ndim == 1:
        # squeeze_me gave us a 1D array of strings/objects
        for i, val in enumerate(arr):
            channel_num = i + 1
            if isinstance(val, (bytes, np.bytes_)):
                val = val.decode('utf-8', errors='ignore')
            elif isinstance(val, np.ndarray):
                val = str(val.flat[0]) if val.size > 0 else 'Unknown'
            region_dict[channel_num] = str(val)
    else:
        # Raw cell array: shape (1, N)
        for i in range(arr.shape[1]):
            channel_num = i + 1
            val = arr[0, i]
            if isinstance(val, np.ndarray) and val.size > 0:
                val = val.flat[0]
            if isinstance(val, (bytes, np.bytes_)):
                val = val.decode('utf-8', errors='ignore')
            region_dict[channel_num] = str(val)

    return region_dict


def get_channel_name(mat_file, channel_num):
    """
    Also extract the channel_name (bipolar label like "F'2 - F'1") for context.
    Returns empty string if not available.
    """
    try:
        mat = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
        names_raw = mat.get('channel_name', None)
        if names_raw is None:
            return ''
        arr = np.atleast_1d(names_raw)
        idx = channel_num - 1
        if idx < 0 or idx >= len(arr):
            return ''
        val = arr[idx]
        if isinstance(val, (bytes, np.bytes_)):
            val = val.decode('utf-8', errors='ignore')
        elif isinstance(val, np.ndarray):
            val = str(val.flat[0]) if val.size > 0 else ''
        return str(val)
    except Exception:
        return ''


def main():
    # Load the channel AUC rankings
    top5_df = pd.read_csv('top5_channels_per_patient.csv')

    # Load existing brain-region results so we don't redo EMU025 and EMU038
    existing_df = pd.read_csv('top5_auc_brain_regions.csv')
    done_patients = set(existing_df['patient'].unique())
    print(f"Already done: {sorted(done_patients)}")

    # Patients still to process
    remaining = [p for p in top5_df['patient'].unique() if p not in done_patients]
    print(f"Remaining: {sorted(remaining)}\n")

    new_rows = []

    for patient in sorted(remaining):
        pat_num = patient.replace('EMU', '')  # e.g. '024'
        setup_path = f"{FORMATTED_DIR}/{patient}/{patient}_MAD_SES1_Setup_bipolar.mat"

        # Load setup file
        try:
            region_dict = get_brain_regions(setup_path)
        except FileNotFoundError:
            print(f"[SKIP] {patient}: Setup_bipolar.mat not found at {setup_path}")
            continue
        except Exception as e:
            print(f"[ERROR] {patient}: could not read setup file — {e}")
            continue

        # Get top-3 channels for this patient
        pat_rows = top5_df[top5_df['patient'] == patient].head(TOP_N)

        print(f"=== {patient} Top {TOP_N} Channels ===")
        for _, row in pat_rows.iterrows():
            ch = int(row['channel'])
            auc = float(row['test_auc'])
            region = region_dict.get(ch, 'Unknown')
            ch_name = get_channel_name(setup_path, ch)

            print(f"  Channel {ch:3d} ({ch_name}): AUC={auc:.4f}, Region={region}")
            new_rows.append({
                'patient':      patient,
                'channel':      ch,
                'test_auc':     auc,
                'brain_region': region,
                'channel_name': ch_name,
            })
        print()

    if not new_rows:
        print("No new results to add.")
        return

    # Build and save the combined output
    new_df = pd.DataFrame(new_rows)

    # Append to the existing CSV (keep all 5 for EMU025/EMU038, add 3 for the rest)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_csv('top5_auc_brain_regions.csv', index=False)
    print(f"Saved {len(new_rows)} new rows → top5_auc_brain_regions.csv")
    print("\nFull updated table:")
    print(combined.to_string(index=False))


if __name__ == '__main__':
    main()
