import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output PNG file")
    parser.add_argument("--title", type=str, default="Yes - No Logit Difference (16x16)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["logit_diff"] = df["Yes_logit"] - df["No_logit"]
    df["p_yes"] = 1.0 / (1.0 + np.exp(-df["logit_diff"]))

    # Extract phA and phB from image filename like '0000_phA00_phB01.png'
    parsed = df["image"].str.extract(r"phA(\d+)_phB(\d+)\.png$")
    df["phA"] = parsed[0].astype(int)
    df["phB"] = parsed[1].astype(int)

    # Average across scenes for each (phA, phB) cell
    grid_df = df.groupby(["phA", "phB"])["p_yes"].mean().reset_index()
    grid = np.full((16, 16), np.nan)
    for _, row in grid_df.iterrows():
        grid[int(row["phA"]), int(row["phB"])] = row["p_yes"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap="seismic", aspect="equal", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="P(Yes)")

    # Compute RMSE against ground truth
    gt_binary = (df["ground_truth"] == "Yes").astype(float)
    rmse = np.sqrt(np.mean((df["p_yes"] - gt_binary) ** 2))

    # Compute RMSE against average p_yes (not 1.0)
    avg_p_yes = df["p_yes"].mean()
    rmse_avg = np.sqrt(np.mean((df["p_yes"] - avg_p_yes) ** 2))

    # Vertical consistency: compare columns 2-6 vs 10-14 (obj2 phase effect)
    obj2_lo = np.nanmean(grid[:, 2:7])   # cols 2~6
    obj2_hi = np.nanmean(grid[:, 10:15]) # cols 10~14
    vc_obj2 = obj2_hi - obj2_lo

    # Horizontal consistency: compare rows 2-6 vs 10-14 (obj1 phase effect)
    obj1_lo = np.nanmean(grid[2:7, :])   # rows 2~6
    obj1_hi = np.nanmean(grid[10:15, :]) # rows 10~14
    vc_obj1 = obj1_hi - obj1_lo

    # Horizontal difference (obj1/rows): rows 6-10 vs rows (14,15,0,1,2)
    wrap_idx = np.array([14, 15, 0, 1, 2])
    hz_obj1_mid = np.nanmean(grid[6:11, :])
    hz_obj1_wrap = np.nanmean(grid[wrap_idx, :])
    hz_obj1 = hz_obj1_mid - hz_obj1_wrap

    # Horizontal difference (obj2/cols): cols 6-10 vs cols (14,15,0,1,2)
    hz_obj2_mid = np.nanmean(grid[:, 6:11])
    hz_obj2_wrap = np.nanmean(grid[:, wrap_idx])
    hz_obj2 = hz_obj2_mid - hz_obj2_wrap

    metrics_text = (
        f"RMSE = {rmse:.4f}  RMSE(avg) = {rmse_avg:.4f}\n"
        f"Obj1 Δ(rows 10-14 vs 2-6)  = {vc_obj1:+.4f}\n"
        f"Obj2 Δ(cols 10-14 vs 2-6)  = {vc_obj2:+.4f}\n"
        f"Obj1 Δ(rows 6-10 vs 14-2)  = {hz_obj1:+.4f}\n"
        f"Obj2 Δ(cols 6-10 vs 14-2)  = {hz_obj2:+.4f}"
    )
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Obj2 Phase (phB)")
    ax.set_ylabel("Obj1 Phase (phA)")
    ax.set_title(args.title)
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
