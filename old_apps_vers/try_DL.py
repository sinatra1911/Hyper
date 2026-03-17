import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.ndimage
import cv2
from tqdm import tqdm
import scipy.io as sio
import h5py
import os
from sklearn.decomposition import PCA

# ==========================================
# 0. Hardware Check
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n" + "=" * 50)
if device.type == 'cuda':
    print(f"✅ SOTA HARDWARE ACCELERATION: Using {torch.cuda.get_device_name(0)}")
else:
    print("❌ WARNING: CUDA not detected! Running on CPU.")
print("=" * 50 + "\n")


# ==========================================
# 1. Multi-Sensor Data Prep & Auto-Alignment
# ==========================================
def auto_align_swir_to_vnir(vnir_raw, swir_raw):
    print("  ➤ Auto-Aligning SWIR to VNIR using Fourier Phase Correlation...")
    swir_resized = cv2.resize(swir_raw, (vnir_raw.shape[1], vnir_raw.shape[0]), interpolation=cv2.INTER_LINEAR)

    vnir_gray = cv2.normalize(np.mean(vnir_raw, axis=2).astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    swir_gray = cv2.normalize(np.mean(swir_resized, axis=2).astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

    transformations = [
        ("None", lambda x: x),
        ("Flip Up/Down", lambda x: np.flipud(x)),
        ("Flip Left/Right", lambda x: np.fliplr(x)),
        ("Flip Both", lambda x: np.flipud(np.fliplr(x)))
    ]

    best_response, best_shift, best_name, best_func = -1, (0, 0), "", None

    for name, func in transformations:
        shift, response = cv2.phaseCorrelate(func(swir_gray).astype(np.float32), vnir_gray)
        if response > best_response:
            best_response, best_shift, best_func, best_name = response, shift, func, name

    shift_x, shift_y = best_shift
    print(
        f"  ➤ Alignment Found! [{best_name}], Shift (X, Y): ({shift_x:.2f}, {shift_y:.2f}), Confidence: {best_response:.4f}")

    swir_aligned = best_func(swir_resized)
    return scipy.ndimage.shift(swir_aligned, shift=[shift_y, shift_x, 0], mode='nearest')


def load_all_modalities(vnir_path, swir_path=None):
    print(f"Loading Primary VNIR Cube: {vnir_path}...")
    ext = os.path.splitext(vnir_path)[1].lower()

    red_idx, nir_idx = 29, 50
    vnir_raw = None

    if ext == '.mat':
        mat_data = sio.loadmat(vnir_path)
        max_size = 0
        for key, val in mat_data.items():
            if isinstance(val, np.ndarray) and val.ndim == 3:
                if val.size > max_size:
                    max_size = val.size
                    vnir_raw = val.astype(np.float32)
        if vnir_raw is None:
            raise ValueError("Could not find a valid 3D hyperspectral array in the .mat file.")
    else:
        img_meta = spy.open_image(vnir_path)
        vnir_raw = np.asarray(img_meta.load(), dtype=np.float32)

        if img_meta.bands.centers:
            wl = np.array(img_meta.bands.centers)
            if np.max(wl) < 100: wl = wl * 1000
            red_idx = np.argmin(np.abs(wl - 650))
            nir_idx = np.argmin(np.abs(wl - 850))
        elif vnir_raw.shape[2] == 271:
            red_idx, nir_idx = 113, 203

    total_bands = vnir_raw.shape[2]
    red_idx = min(red_idx, total_bands - 1)
    nir_idx = min(nir_idx, total_bands - 1)

    vnir_clean = np.clip(vnir_raw, 0, None)
    vnir_clean /= (np.linalg.norm(vnir_clean, axis=-1, keepdims=True) + 1e-8)

    if not swir_path:
        return vnir_clean, None, vnir_clean, vnir_raw, red_idx, nir_idx

    print(f"Loading Secondary SWIR Cube: {swir_path}...")
    swir_raw = np.asarray(spy.open_image(swir_path).load(), dtype=np.float32)
    swir_aligned = auto_align_swir_to_vnir(vnir_raw, swir_raw)

    swir_clean = np.clip(swir_aligned, 0, None)
    swir_clean /= (np.linalg.norm(swir_clean, axis=-1, keepdims=True) + 1e-8)

    print(f"Fusing VNIR ({vnir_raw.shape[2]} bands) + SWIR ({swir_aligned.shape[2]} bands)...")
    fused_raw = np.concatenate((vnir_raw, swir_aligned), axis=-1)
    fused_clean = np.clip(fused_raw, 0, None)
    fused_clean /= (np.linalg.norm(fused_clean, axis=-1, keepdims=True) + 1e-8)

    return vnir_clean, swir_clean, fused_clean, vnir_raw, red_idx, nir_idx


# ==========================================
# 2. Soft Semantic Weighting
# ==========================================
def compute_semantic_weights(cube, vnir_raw, red_idx, nir_idx):
    print("  ➤ Computing Soft Semantic Background Weights...")
    H, W, B = cube.shape

    # 1. Soft Vegetation Score
    red_band = vnir_raw[:, :, red_idx]
    nir_band = vnir_raw[:, :, nir_idx]
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
    veg_score = np.clip((ndvi - 0.2) / 0.4, 0, 1)

    # 2. Soft Soil/Rock Score
    X = cube.reshape(-1, B)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_pca)

    residual = np.linalg.norm(X - X_recon, axis=1).reshape(H, W)
    res_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
    soil_score = np.clip(1.0 - (res_norm * 3.0), 0, 1)

    # 3. Spectral Weirdness (Preserves metals and planes)
    grad = np.mean(np.abs(np.diff(cube, axis=2)), axis=2)
    weirdness = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

    multiplier = (1.0 - veg_score) * (1.0 - soil_score) * (weirdness + 0.1)
    multiplier = (multiplier - multiplier.min()) / (multiplier.max() - multiplier.min() + 1e-8)

    return multiplier


# ==========================================
# 3. NEW: Self-Supervised Deep Autoencoder
# ==========================================
class SpectralAutoencoder(nn.Module):
    """
    1D Deep Autoencoder. Compresses the spectral chemistry into an 8-neuron bottleneck.
    It memorizes dirt/roads easily, but fails on planes.
    """

    def __init__(self, input_dim, bottleneck=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_deep_autoencoder(cube, epochs=100, batch_size=2048):
    print("  ➤ Running Self-Supervised Deep Autoencoder (Background Learner)...")
    H, W, B = cube.shape
    X_flat = cube.reshape(-1, B)

    # Train only on a random 10% sample of the image for massive speedup
    num_samples = int(H * W * 0.1)
    indices = np.random.choice(H * W, num_samples, replace=False)
    train_data = torch.tensor(X_flat[indices], dtype=torch.float32, device=device)

    model = SpectralAutoencoder(input_dim=B).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.train()
    dataset = torch.utils.data.TensorDataset(train_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in dataloader:
            x_batch = batch[0]
            optimizer.zero_grad()
            reconstructed = model(x_batch)
            loss = F.mse_loss(reconstructed, x_batch)
            loss.backward()
            optimizer.step()

    # Inference on the entire image
    model.eval()
    X_tensor = torch.tensor(X_flat, dtype=torch.float32, device=device)
    with torch.no_grad():
        reconstructed_full = model(X_tensor)
        # Anomaly score is the Reconstruction Error (MSE + Cosine Distance)
        mse_error = torch.mean((reconstructed_full - X_tensor) ** 2, dim=1)
        cos_error = 1.0 - F.cosine_similarity(reconstructed_full, X_tensor, dim=1)
        total_error = mse_error + (0.5 * cos_error)

    ae_map = total_error.view(H, W).cpu().numpy()
    return (ae_map - np.min(ae_map)) / (np.max(ae_map) - np.min(ae_map) + 1e-8)


# ==========================================
# 4. Classical Detectors
# ==========================================
def run_subpixel_unmixing_osp(cube, k_endmembers=15):
    print(f"  ➤ Running OSP Sub-Pixel Unmixing...")
    H, W, B = cube.shape
    X = torch.tensor(cube.reshape(-1, B).T, dtype=torch.float32, device=device)
    R_mat = torch.matmul(X, X.T) / X.shape[1]
    eigenvalues, eigenvectors = torch.linalg.eigh(R_mat)
    idx = torch.argsort(eigenvalues, descending=True)
    U_bg = eigenvectors[:, idx][:, :k_endmembers]
    I = torch.eye(B, device=device)
    P_ortho = I - torch.matmul(U_bg, U_bg.T)
    unmixed_residual = torch.matmul(P_ortho, X)
    osp_scores = torch.norm(unmixed_residual, dim=0).view(H, W).cpu().numpy()
    return (osp_scores - np.min(osp_scores)) / (np.max(osp_scores) - np.min(osp_scores) + 1e-8)


def run_local_rx_gpu(cube, inner_win=5, outer_win=15, n_components=10):
    print(f"  ➤ Running Local RX (Memory-Safe Mode)...")
    H, W, B = cube.shape
    pca_cube = PCA(n_components=n_components).fit_transform(cube.reshape(-1, B)).reshape(H, W, n_components)
    cube_tensor = torch.tensor(pca_cube, dtype=torch.float32, device=device).permute(2, 0, 1)
    C = n_components

    pad = outer_win // 2
    padded = F.pad(cube_tensor.unsqueeze(0), (pad, pad, pad, pad), mode='reflect')

    mask = torch.ones((outer_win, outer_win), dtype=torch.bool, device=device)
    center = outer_win // 2
    in_r = inner_win // 2
    mask[center - in_r:center + in_r + 1, center - in_r:center + in_r + 1] = False
    mask_flat = mask.view(-1)

    rx_scores = torch.zeros(H * W, device=device)

    chunk_lines = 100
    pbar = tqdm(total=H, desc="    Scanning Map")
    for row_start in range(0, H, chunk_lines):
        row_end = min(row_start + chunk_lines, H)
        slice_padded = padded[:, :, row_start: row_end + 2 * pad, :]

        unfold = nn.Unfold(kernel_size=outer_win)
        patches = unfold(slice_padded).view(C, outer_win ** 2, -1).permute(2, 0, 1)
        bg_patches = patches[:, :, mask_flat]
        cp_c = cube_tensor[:, row_start:row_end, :].reshape(C, -1).T.unsqueeze(2)

        L_chunk = bg_patches.shape[0]
        sub_chunk_size = 10000

        for i in range(0, L_chunk, sub_chunk_size):
            sub_end = min(i + sub_chunk_size, L_chunk)
            bg_sub = bg_patches[i:sub_end]
            cp_sub = cp_c[i:sub_end]

            mu = torch.mean(bg_sub, dim=2, keepdim=True)
            bg_centered = bg_sub - mu

            cov = torch.bmm(bg_centered, bg_centered.transpose(1, 2)) / (bg_centered.shape[2] - 1)
            cov = cov + torch.eye(C, device=device).unsqueeze(0) * 1e-4
            cov_inv = torch.linalg.inv(cov)

            x_c = cp_sub - mu
            md = torch.bmm(torch.bmm(x_c.transpose(1, 2), cov_inv), x_c)

            rx_scores[(row_start * W) + i: (row_start * W) + sub_end] = md.squeeze(-1).squeeze(-1)

        pbar.update(row_end - row_start)
    pbar.close()

    local_map = rx_scores.view(H, W).cpu().numpy()
    return (local_map - local_map.min()) / (local_map.max() - local_map.min() + 1e-8)


# ==========================================
# 5. Interactive Visualization Engine
# ==========================================
def get_display_image(raw_vnir_cube):
    H, W, B = raw_vnir_cube.shape
    if B == 224:
        r_idx, g_idx, b_idx = 29, 19, 9
    else:
        r_idx, g_idx, b_idx = min(int(B * 0.15), B - 1), min(int(B * 0.1), B - 1), min(int(B * 0.04), B - 1)

    rgb = np.stack([raw_vnir_cube[:, :, r_idx], raw_vnir_cube[:, :, g_idx], raw_vnir_cube[:, :, b_idx]], axis=-1)
    for i in range(3):
        c_min, c_max = np.percentile(rgb[:, :, i], 1), np.percentile(rgb[:, :, i], 99.5)
        rgb[:, :, i] = np.clip((rgb[:, :, i] - c_min) / (c_max - c_min + 1e-8), 0, 1)
    return rgb


class InteractiveDashboard:
    def __init__(self, scene_img, heatmaps_dict, modality_name):
        self.num_algos = len(heatmaps_dict)
        total_panels = 1 + self.num_algos + 1

        self.fig = plt.figure(figsize=(4 * total_panels, 7))
        self.fig.canvas.manager.set_window_title(f"Results: {modality_name}")
        self.fig.subplots_adjust(bottom=0.25, top=0.9, wspace=0.15)

        gs = self.fig.add_gridspec(1, total_panels)
        self.axes = []
        self.sliders = []
        self.heatmaps_data = list(heatmaps_dict.values())
        self.heatmap_imgs = []

        ax_orig = self.fig.add_subplot(gs[0])
        ax_orig.imshow(scene_img)
        ax_orig.set_title("Original Scene", fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        self.axes.append(ax_orig)

        idx = 1
        for name, heatmap in heatmaps_dict.items():
            ax = self.fig.add_subplot(gs[idx], sharex=self.axes[0], sharey=self.axes[0])
            img = ax.imshow(np.clip(heatmap, 0, np.percentile(heatmap, 95.0)), cmap='turbo')
            ax.set_title(f"{name}", fontsize=12, fontweight='bold')
            ax.axis('off')

            self.axes.append(ax)
            self.heatmap_imgs.append(img)

            ax_slider = self.fig.add_axes([0.1 + (idx * 0.8 / total_panels), 0.1, 0.6 / total_panels, 0.03])
            slider = Slider(ax_slider, 'Thresh %', 80.0, 99.99, valinit=97.0, valstep=0.1)

            def make_update(img_obj, data):
                def update(val):
                    vmax = np.percentile(data, val)
                    img_obj.set_data(np.clip(data, 0, vmax))
                    img_obj.set_clim(vmin=0, vmax=vmax)
                    self.update_fusion()

                return update

            slider.on_changed(make_update(img, heatmap))
            self.sliders.append(slider)
            idx += 1

        self.ax_det = self.fig.add_subplot(gs[idx], sharex=self.axes[0], sharey=self.axes[0])
        self.ax_det.imshow(scene_img)
        self.det_img = self.ax_det.imshow(np.zeros((scene_img.shape[0], scene_img.shape[1], 4)))
        self.ax_det.set_title("Voting Detections (Cyan)", fontsize=12, fontweight='bold')
        self.ax_det.axis('off')
        self.axes.append(self.ax_det)

        ax_vote_slider = self.fig.add_axes([0.1 + (idx * 0.8 / total_panels), 0.1, 0.6 / total_panels, 0.03])
        self.slider_vote = Slider(ax_vote_slider, 'Required Votes', 1, self.num_algos, valinit=2, valstep=1)
        self.slider_vote.on_changed(lambda val: self.update_fusion())
        self.sliders.append(self.slider_vote)

        self.update_fusion()
        plt.show(block=False)

    def update_fusion(self):
        H, W = self.heatmaps_data[0].shape
        vote_map = np.zeros((H, W), dtype=np.float32)

        for i, data in enumerate(self.heatmaps_data):
            thresh_val = np.percentile(data, self.sliders[i].val)
            binary_mask = (data > thresh_val).astype(np.float32)
            vote_map += binary_mask

        required_votes = int(self.slider_vote.val)
        final_mask = vote_map >= required_votes

        solid_mask = scipy.ndimage.binary_opening(final_mask, structure=np.ones((2, 2)))
        solid_mask = scipy.ndimage.binary_dilation(solid_mask, structure=np.ones((2, 2)))

        overlay = np.zeros((H, W, 4))
        overlay[solid_mask] = [0, 1, 1, 1]
        self.det_img.set_data(overlay)
        self.fig.canvas.draw_idle()


# ==========================================
# 6. Execution Engine
# ==========================================
def execute_modality(modality_name, cube, rgb_background, semantic_weights, config):
    print(f"\n[{modality_name}] Initiating Evaluation...")
    active_heatmaps = {}

    # 1. Evaluate unmasked statistics
    if config['use_deep_ae']: active_heatmaps['Deep AE'] = run_deep_autoencoder(cube)
    if config['use_osp']: active_heatmaps['OSP Unmixing'] = run_subpixel_unmixing_osp(cube)
    if config['use_rx']: active_heatmaps['Local RX'] = run_local_rx_gpu(cube)

    if not active_heatmaps: return None

    # 2. Soft Post-Process Multiplication
    for name in active_heatmaps:
        active_heatmaps[name] = active_heatmaps[name] * semantic_weights

    return InteractiveDashboard(rgb_background, active_heatmaps, modality_name)


def run_pipeline(vnir_path, swir_path, config):
    print("\n" + "*" * 60)
    print("INITIALIZING DEEP-AE HYPERSPECTRAL PIPELINE")
    print("*" * 60)

    vnir_clean, swir_clean, fused_clean, raw_vnir_cube, red_idx, nir_idx = load_all_modalities(vnir_path, swir_path)
    rgb_bg = get_display_image(raw_vnir_cube)

    cube_for_semantics = fused_clean if fused_clean is not None else vnir_clean
    semantic_weights = compute_semantic_weights(cube_for_semantics, raw_vnir_cube, red_idx, nir_idx)

    dashboards = []
    dashboards.append(execute_modality("VNIR Only", vnir_clean, rgb_bg, semantic_weights, config))

    if swir_clean is not None:
        dashboards.append(execute_modality("SWIR Only", swir_clean, rgb_bg, semantic_weights, config))
        dashboards.append(execute_modality("FUSED (VNIR + SWIR)", fused_clean, rgb_bg, semantic_weights, config))

    print("\n✅ Execution Complete! Interactive windows are active.")
    plt.show(block=True)


if __name__ == "__main__":
    vnir_filepath = r"C:\Users\Public\HyperData\ADERET\100m\raw_7024_rd_rf.hdr"
    swir_filepath = None

    pipeline_config = {
        'use_deep_ae': True,  # Replaces IForest
        'use_osp': True,
        'use_rx': True
    }

    run_pipeline(vnir_filepath, swir_filepath, pipeline_config)