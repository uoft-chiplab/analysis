#!/usr/bin/env python3
"""
Compute and plot optical density from a .mat file that contains four images.

Frame meanings:
  - img2: object + imaging light
  - img3: imaging light only (reference)
  - img4: dark frame (no light; for dark-count subtraction)

OD = -ln( (img2 - img4) / (img3 - img4) )

How to use:
  1) Set MAT_PATH below.
  2) If you know the variable names for frames 2–4, set IMG2_NAME/IMG3_NAME/IMG4_NAME.
     Otherwise, leave them as "" and the script will auto-pick the first four 2D arrays.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

files = ["G_bandmappingFMimages_0001.mat",
         "H_bandmappingAMimages_0001.mat",
         "F_LAT1_40er_fm_rfspectra_199p4G_0024.mat",
         ]

# -------------------- USER SETTINGS --------------------
IMG_PATH = "C:\\Users\\colin\\OneDrive\\Physics\\Thesis\\figures\\images\\matlab_files\\"
MAT_FILE = files[2]   # <-- set your .mat file name
IMG2_NAME = ""                # e.g., "I_obj"; leave "" to auto-detect
IMG3_NAME = ""                # e.g., "I_ref"; leave "" to auto-detect
IMG4_NAME = ""                # e.g., "I_dark"; leave "" to auto-detect
SHOW_FRAMES = True             # also display I2, I3, I4
VMAX = None                    # e.g., 3.0 to cap OD color scale; None = auto 99th percentile
CMAP = "magma"               # e.g., "viridis", "plasma", "inferno", "magma", "cividis"
SAVE_OD_PDF = f"ssi_{CMAP}.pdf"    # e.g., "od.pdf" to save, or "" to skip
# SAVE_OD_PDF = ""
EPS = 1e-6                     # small number for safe log/divide
# -------------------------------------------------------

# --- ROI SETTINGS (set None for full image) ---
# Use pixel indices (inclusive start, exclusive end) — like Python slicing

## Bandmapping
X_MIN, X_MAX = 430, 600   # columns
Y_MIN, Y_MAX = 50, 220   # rows

## SSI
X_MIN, X_MAX = 490, 555   # columns
Y_MIN, Y_MAX = 90, 200   # rows
# Set all four to None to disable cropping
# -------------------------------------------------------

MAT_FILE_PATH = IMG_PATH + MAT_FILE

def is_2d_numeric(a):
    return isinstance(a, np.ndarray) and a.ndim == 2 and np.issubdtype(a.dtype, np.number)


def auto_select_images(vars_dict):
    """Pick four 2D arrays, preferring names that look like img1..img4/I1..I4."""
    candidates = [(k, v) for k, v in vars_dict.items() if is_2d_numeric(v)]
    if len(candidates) < 4:
        raise ValueError(f"Need at least 4 2D arrays; found {len(candidates)}.")

    def key_rank(k):
        tag = k.lower()
        rank = 999
        for i in range(1, 5):
            for prefix in ("img", "im", "i"):
                if tag == f"{prefix}{i}" or tag.endswith(f"_{i}") or tag.startswith(f"{prefix}{i}"):
                    rank = min(rank, i)
        return rank

    candidates.sort(key=lambda kv: (key_rank(kv[0]), kv[0]))
    # Map assumed order [img1, img2, img3, img4] -> take 1,2,3 as I2,I3,I4
    _, I2 = candidates[1]
    _, I3 = candidates[2]
    _, I4 = candidates[3]
    used_names = [candidates[1][0], candidates[2][0], candidates[3][0]]
    return I2, I3, I4, used_names


def get_images(vars_dict, img2, img3, img4):
    """Return I2, I3, I4 and the names used (for plotting labels)."""
    if img2 and img3 and img4:
        try:
            I2 = np.array(vars_dict[img2])
            I3 = np.array(vars_dict[img3])
            I4 = np.array(vars_dict[img4])
        except KeyError as e:
            raise KeyError(f"Variable {e} not found. Available keys: {list(vars_dict.keys())}")
        for nm, arr in (("img2", I2), ("img3", I3), ("img4", I4)):
            if not is_2d_numeric(arr):
                raise ValueError(f"{nm} ('{locals()[nm.upper() + '_NAME']}' ) is not a 2D numeric array.")
        return I2, I3, I4, [img2, img3, img4]
    else:
        return auto_select_images(vars_dict)

def crop_roi(img, x_min, x_max, y_min, y_max):
    """Crop to region of interest if limits are defined."""
    if None in (x_min, x_max, y_min, y_max):
        return img
    return img[y_min:y_max, x_min:x_max]

def compute_optical_density(I2, I3, I4, eps=1e-6):
    """OD = -ln( (I2 - I4) / (I3 - I4) ), with clipping to avoid nonpositive values."""
    I2 = I2.astype(np.float64, copy=False)
    I3 = I3.astype(np.float64, copy=False)
    I4 = I4.astype(np.float64, copy=False)

    num = np.maximum(I2 - I4, eps)
    den = np.maximum(I3 - I4, eps)
    ratio = np.maximum(num / den, eps)

    return -np.log(ratio)


def main():
    data = scipy.io.loadmat(MAT_FILE_PATH)
    # Strip MATLAB meta keys
    vars_dict = {k: v for k, v in data.items() if not k.startswith("__")}

    # Fetch images
    I2, I3, I4, used = get_images(vars_dict, IMG2_NAME, IMG3_NAME, IMG4_NAME)

    if I2.shape != I3.shape or I2.shape != I4.shape:
        raise ValueError(f"Image shapes must match. Got I2={I2.shape}, I3={I3.shape}, I4={I4.shape}")

    # Crop to ROI
    I2 = crop_roi(I2, X_MIN, X_MAX, Y_MIN, Y_MAX)
    I3 = crop_roi(I3, X_MIN, X_MAX, Y_MIN, Y_MAX)
    I4 = crop_roi(I4, X_MIN, X_MAX, Y_MIN, Y_MAX)

    # Compute OD
    OD = compute_optical_density(I2, I3, I4, eps=EPS)

    # Plot OD
    plt.figure(figsize=(6, 5))
    vmin = 0.0
    finite_mask = np.isfinite(OD)
    vmax = VMAX if VMAX is not None else np.percentile(OD[finite_mask], 99) if np.any(finite_mask) else None
    im = plt.imshow(OD, vmin=vmin, vmax=vmax, cmap=CMAP)
    # plt.title("Optical Density")
    # cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    # cbar.set_label("OD")
    plt.axis("off")
    plt.tight_layout()

    if SAVE_OD_PDF:
        plt.savefig(IMG_PATH + SAVE_OD_PDF)
        print(f"Saved OD image to {IMG_PATH + SAVE_OD_PDF}")

    # Optionally show input frames
    if SHOW_FRAMES:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(I2); axs[0].set_title(f"I2 (obj+light)\n{used[0]}")
        axs[1].imshow(I3); axs[1].set_title(f"I3 (light only)\n{used[1]}")
        axs[2].imshow(I4); axs[2].set_title(f"I4 (dark)\n{used[2]}")
        for ax in axs: ax.axis("off")
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
