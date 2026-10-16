import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Save Codebooks
def save_codebook(kmeans, out_dir="cv", fname="kmeans_codebook.joblib"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, out_dir / fname)
    # save each centroids
    np.save(out_dir / "codebook_centers.npy", kmeans.cluster_centers_.astype(np.float32))
    print(f"[SAVE] codebook -> {out_dir/fname}, centers -> {out_dir/'codebook_centers.npy'}")

# Load Codebooks
def load_codebook(path_joblib="cv/kmeans_codebook.joblib"):
    kmeans = joblib.load(path_joblib)
    print(f"[LOAD] codebook <- {path_joblib}")
    return kmeans

#  Visulaize codewords
def visualize_codebook_grid(centers, patch=5, cols=16, out_path="cv/codewords_grid.png"):
    K = centers.shape[0]
    rows = int(np.ceil(K / cols))
    
    imgs = []
    for k in range(K):
        c = centers[k].reshape(patch, patch)
        c_min, c_max = c.min(), c.max()
        if c_max - c_min < 1e-6:
            v = np.zeros((patch, patch), dtype=np.uint8)
        else:
            v = ((c - c_min) / (c_max - c_min) * 255.0).astype(np.uint8)
        imgs.append(v)

    Hm = rows * patch + (rows-1) * 1
    Wm = cols * patch + (cols-1) * 1
    grid = np.full((Hm, Wm), 255, np.uint8)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= K: break
            y = r*(patch+1); x = c*(patch+1)
            grid[y:y+patch, x:x+patch] = imgs[idx]
            idx += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(min(16, cols), max(4, rows)))
    plt.imshow(grid, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Codewords (K={K}, patch={patch}x{patch})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVE] codewords grid -> {out_path}")
