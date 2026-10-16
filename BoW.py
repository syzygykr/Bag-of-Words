import numpy as np
from torchvision.datasets import MNIST
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
#from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from pathlib import Path

from vis import save_codebook, load_codebook, visualize_codebook_grid

def load_data():
    train_data = MNIST(root='./data', train=True, download=True)
    test_data  = MNIST(root='./data', train=False, download=True)

    # image: (N, 28, 28) -> (N, 784)
    X_train = train_data.data.numpy().reshape(-1, 28*28).astype(np.float32)
    X_test  = test_data.data.numpy().reshape(-1, 28*28).astype(np.float32)

    # normalize [0, 255] -> [0, 1]
    X_train /= 255.0
    X_test  /= 255.0

    # labels
    Y_train = train_data.targets.numpy()
    Y_test  = test_data.targets.numpy()

    return X_train, Y_train, X_test, Y_test


# 1) Patch Extraction
def extract_dense_patches(images_flat, patch=5, stride=2):

    patches_list = []
    img_patch_slices = []
    start = 0
    H, W = 28, 28
    p = patch
    images = images_flat.reshape(-1, H, W)
    P_per_img = ((H - p) // stride + 1) * ((W - p) // stride + 1)  # Number of patches per image
    s = stride

    N = len(images)

    for i, img in enumerate(images):
        cur = []
        for y in range(0, H - p + 1, s):
            for x in range(0, W - p + 1, s):
                patch_ = img[y:y+p, x:x+p].astype(np.float32)
                # Data Normalization
                mu, sd = patch_.mean(), patch_.std() + 1e-6
                cur.append(((patch_ - mu) / sd).reshape(-1))
        cur = np.stack(cur, axis=0)  # (P_per_img, p*p)
        patches_list.append(cur)
        img_patch_slices.append(slice(start, start + P_per_img)) # Index (Start - End) for each image
        start += P_per_img

        if (i + 1) % 1000 == 0 or (i + 1) == N:
            print(f"[Patch] {i + 1}/{N}", end="\r")
    
    patches_all = np.concatenate(patches_list, axis=0).astype(np.float32)
    return patches_all, img_patch_slices, P_per_img


# 2) Codebook Formation using KMeans
def build_codebook(patches_all, K=128, sample_for_kmeans=200000, random_state=0):
    # For high-speed, we samples data (sample_for_kmeans) from all observations
    rng = np.random.default_rng(random_state)
    idx = rng.choice(patches_all.shape[0], size=min(sample_for_kmeans, patches_all.shape[0]), replace=False)
    sample = patches_all[idx]
    km = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init=5, random_state=random_state, verbose=0)
    km.fit(sample)
    return km


# 3) Image Representation
def images_to_bow_hist(kmeans, patches_all, img_patch_slices, K):
    # Find the closest codeword index
    codes = kmeans.predict(patches_all)  # (TotalP,)
    N = len(img_patch_slices)
    H_bow = np.zeros((N, K), dtype=np.float32)

    for i, sl in enumerate(img_patch_slices):
        hist, _ = np.histogram(codes[sl], bins=np.arange(K+1))
        hist = hist.astype(np.float32)
        n = np.linalg.norm(hist) + 1e-12
        hist /= n
        H_bow[i] = hist
    return H_bow


# 4) Train ML Classifier
def train_and_eval(X_train, Y_train, X_test, Y_test):
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, solver="lbfgs")
    # clf = LinearSVC()  # You can use SVM as well
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(Y_test, pred)
    print(f"Test acc = {acc:.4f}")
    print(classification_report(Y_test, pred))
    return clf, pred

# Background codewords = (almost) zero-vector centers.
# Normalized real patches have std 1, so only uniform patches map to a center with norm ~ 0.
def find_background_codewords(centers, thr=0.5):
    norms = np.linalg.norm(centers, axis=1)   # (K,)
    bg_idx = np.where(norms < thr)[0]
    return bg_idx, norms

# Primary codeword = argmax of class-mean histogram, background codewords excluded
# Returns: (hists to plot, primary indices)
def find_primary_codewords(Xbow, Y, n_classes=10, bg_idx=None):
    # Xbow: (N, K), Y: (N,)
    if bg_idx is None:
        bg_idx = []
    mean_hists = np.stack([Xbow[Y == c].mean(axis=0) for c in range(n_classes)])  # (10, K)

    hists = mean_hists.copy()

    masked = hists.copy()
    if len(bg_idx) > 0:
        masked[:, bg_idx] = -np.inf   # excluded from argmax
        hists[:, bg_idx] = 0.0        # shown as 0 in the plot
    primary = masked.argmax(axis=1)                                              # (10,)
    return hists, primary

def visualize_primary_codewords(X_img, Y, mean_hists, primary, centers,
                                patch=5, out_path="cv/primary_codewords.png", title=""):
    n, K = mean_hists.shape
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), layout="constrained")
    for c, ax in enumerate(axes.flat):
        k = primary[c]
        h = mean_hists[c]

        # mean histogram, highlight primary bin
        bars = ax.bar(np.arange(K), h, color="tab:blue")
        bars[k].set_color("tab:red")
        ax.set_xlim(0, K)
        # leave the top ~40% of the axes free for the inset images
        lo, hi = min(0.0, h.min()), h.max()
        ax.set_ylim(lo, lo + (hi - lo) * 1.6 + 1e-9)
        ax.set_title(f"class {c} | primary {k} ({h[k]:.3f})", fontsize=9)

        # sample image of class c (inset, top-left)
        sample = X_img[np.where(Y == c)[0][0]].reshape(28, 28)
        ax_img = ax.inset_axes([0.05, 0.66, 0.25, 0.32])
        ax_img.imshow(sample, cmap="gray")
        ax_img.axis("off")

        # primary codeword patch (inset, top-right; imshow auto-scales min-max)
        ax_cw = ax.inset_axes([0.70, 0.66, 0.25, 0.32])
        ax_cw.imshow(centers[k].reshape(patch, patch), cmap="gray")
        ax_cw.set_title(f"codeword {k}", fontsize=7)
        ax_cw.axis("off")

    fig.suptitle(title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[SAVE] primary codewords -> {out_path}")
    # plt.show() is called once in __main__ so that all figures are shown together


# Run Bag of Words
if __name__ == "__main__":
    # Hyper-parameters
    N_TRAIN = 20000   # Training samples
    N_TEST  = 5000    # Test samples
    PATCH   = 5       # Local feature size
    STRIDE  = 2       # Dense sampling (low value) -> Coarse sampling (high value)
    K       = 128     # Dictionary size
    RS      = 0

    # Data Loader which is provided for previous assignments.
    X_train, Y_train, X_test, Y_test = load_data()

    # Step 2
    #kmeans = build_codebook(patches_tr, K=K, sample_for_kmeans=200000, random_state=RS)
    #save_codebook(kmeans)
    kmeans = load_codebook()
    centers = kmeans.cluster_centers_         # (K, p*p)
    #visualize_codebook_grid(centers, patch=PATCH, cols=16, out_path="cv/codewords_grid.png")

    # Step 1 + Step 3 (cached: delete cv/Xtrain_bow.npy if the codebook is rebuilt)
    bow_path = Path("cv/Xtrain_bow.npy")
    if bow_path.exists():
        Xtrain_bow = np.load(bow_path)
        print(f"[LOAD] Xtrain_bow <- {bow_path} {Xtrain_bow.shape}")
    else:
        # Step 1
        patches_tr, slices_tr, P_per = extract_dense_patches(X_train, patch=PATCH, stride=STRIDE)
        #patches_te, slices_te, _     = extract_dense_patches(X_test, patch=PATCH, stride=STRIDE)

        print(f"train patches: {patches_tr.shape}, per image: {P_per}")

        # Step 3
        Xtrain_bow = images_to_bow_hist(kmeans, patches_tr, slices_tr, K)
        #Xtest_bow  = images_to_bow_hist(kmeans, patches_te, slices_te, K)
        bow_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(bow_path, Xtrain_bow)
        print(f"[SAVE] Xtrain_bow -> {bow_path}")

    # Step 4
    #clf, pred = train_and_eval(Xtrain_bow, Y_train, Xtest_bow, Y_test)
    
    # Background codewords
    bg_idx, norms = find_background_codewords(centers)
    print(f"center norms (smallest 5): { {int(i): round(float(norms[i]), 3) for i in np.argsort(norms)[:5]} }")
    print(f"background codewords: {bg_idx.tolist()}")

    #Print primary codewords for each class
    hists, primary = find_primary_codewords(Xtrain_bow, Y_train, bg_idx=bg_idx)
    print(f"distinct primary codewords: {len(set(primary.tolist()))}/10")
    for c in range(10):
        print(f"class {c}: primary codeword = {primary[c]}")
    visualize_primary_codewords(X_train, Y_train, hists, primary, centers, patch=PATCH,
                                out_path="cv/primary_exclude_bg.png",
                                title="Primary codeword (background codewords excluded)")
    plt.show()


    # Vis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].imshow(X_train[0].reshape(28, 28), cmap="gray"); axes[0].set_title(f"y={Y_train[0]}"); axes[0].axis("off")
    axes[1].bar(np.arange(K), Xtrain_bow[0]); axes[1].set_title("BoW histogram"); axes[1].set_xlim(0, K)
    plt.tight_layout(); plt.show()
    """