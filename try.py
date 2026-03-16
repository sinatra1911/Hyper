import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
from tqdm import tqdm

# ==========================================
# 0. Hardware Check
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n" + "=" * 50)
if device.type == 'cuda':
    print(f"✅ HARDWARE ACCELERATION ACTIVE: Using {torch.cuda.get_device_name(0)}")
else:
    print("❌ WARNING: CUDA not detected! Running on slow CPU.")
print("=" * 50 + "\n")


# ==========================================
# 1. Data Loading & Alignment
# ==========================================
def load_reflectance_cube(filepath):
    if filepath is None:
        return None
    print(f"Loading Reflectance Cube: {filepath}...")
    cube = np.asarray(spy.open_image(filepath).load(), dtype=np.float32)
    cube_max = np.max(cube)
    if cube_max > 1.0:
        cube = cube / cube_max
    return np.clip(cube, 0, 1)


def register_and_mask_cubes(vnir_np, swir_np):
    H_v, W_v, B_v = vnir_np.shape
    H_s, W_s, B_s = swir_np.shape

    ref_vnir = cv2.normalize(vnir_np[:, :, B_v // 2], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    ref_swir = cv2.normalize(swir_np[:, :, B_s // 2], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    sift = cv2.SIFT_create()
    kp_s, des_s = sift.detectAndCompute(ref_swir, None)
    best_matches, best_kp_v, best_flip = [], None, None

    print("Aligning VNIR to SWIR canvas...")
    for flip_code in [None, 0, 1, -1]:
        test_vnir = ref_vnir.copy() if flip_code is None else cv2.flip(ref_vnir, flip_code)
        kp_v, des_v = sift.detectAndCompute(test_vnir, None)
        if des_s is not None and des_v is not None:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des_v, des_s, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) > len(best_matches):
                best_matches, best_kp_v, best_flip = good, kp_v, flip_code

    if len(best_matches) < 10:
        raise ValueError("CRITICAL: Cameras do not overlap enough for feature matching.")

    src_pts = np.float32([best_kp_v[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_s[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    aligned_vnir = np.zeros((H_s, W_s, B_v), dtype=np.float32)
    for b in tqdm(range(B_v), desc="Warping VNIR"):
        band_data = vnir_np[:, :, b]
        if best_flip is not None:
            band_data = cv2.flip(band_data, best_flip)
        aligned_vnir[:, :, b] = cv2.warpPerspective(band_data, M, (W_s, H_s))

    overlap_mask = (np.sum(aligned_vnir, axis=2) > 0.01) & (np.sum(swir_np, axis=2) > 0.01)
    return aligned_vnir, overlap_mask


# ==========================================
# 2. Dynamic Multi-Norm Multi-Metric Loss
# ==========================================
class DynamicMultiMetricLoss(nn.Module):
    def __init__(self, top_k_ratio=0.05):
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.loss_weights = nn.Parameter(torch.ones(9))

    def _apply_norms(self, x):
        x_min = x.amin(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
        norm_minmax = (x - x_min) / (x_max - x_min + 1e-8)

        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True)
        norm_z = (x - x_mean) / (x_std + 1e-8)

        b_mean = x.mean(dim=(0, 2, 3), keepdim=True)
        b_std = x.std(dim=(0, 2, 3), keepdim=True)
        norm_bn = (x - b_mean) / (b_std + 1e-8)

        return [norm_minmax, norm_z, norm_bn]

    def _compute_metrics(self, pred, targ):
        B, C, H, W = pred.shape
        p = pred.view(B, C, -1)
        t = targ.view(B, C, -1)

        sam = 1.0 - F.cosine_similarity(p, t, dim=1)

        p_mu = p.mean(dim=1, keepdim=True)
        t_mu = t.mean(dim=1, keepdim=True)
        p_ctr, t_ctr = p - p_mu, t - t_mu
        cov = (p_ctr * t_ctr).sum(dim=1)
        std_p = torch.sqrt((p_ctr ** 2).sum(dim=1) + 1e-8)
        std_t = torch.sqrt((t_ctr ** 2).sum(dim=1) + 1e-8)
        pearson = 1.0 - (cov / (std_p * std_t))

        p_pos = p - p.amin(dim=1, keepdim=True)
        t_pos = t - t.amin(dim=1, keepdim=True)
        p_prob = torch.clamp(p_pos / (p_pos.sum(dim=1, keepdim=True) + 1e-8), min=1e-8)
        t_prob = torch.clamp(t_pos / (t_pos.sum(dim=1, keepdim=True) + 1e-8), min=1e-8)

        d_pt = (p_prob * torch.log(p_prob / t_prob)).sum(dim=1)
        d_tp = (t_prob * torch.log(t_prob / p_prob)).sum(dim=1)
        sid = d_pt + d_tp

        return sam, pearson, sid

    def forward(self, reconstructed, target):
        preds_normed = self._apply_norms(reconstructed)
        targs_normed = self._apply_norms(target)

        all_losses = []
        for p_norm, t_norm in zip(preds_normed, targs_normed):
            sam, pearson, sid = self._compute_metrics(p_norm, t_norm)
            all_losses.extend([sam, pearson, sid])

        stacked_losses = torch.stack(all_losses, dim=0)
        weights = F.softmax(self.loss_weights, dim=0).view(9, 1, 1)
        weighted_error_map = (stacked_losses * weights).sum(dim=0)

        err_flat = weighted_error_map.reshape(-1)
        k = max(1, int(err_flat.size(0) * self.top_k_ratio))
        top_k_errors, _ = torch.topk(err_flat, k)

        return torch.mean(top_k_errors), weighted_error_map.view(reconstructed.shape[0], reconstructed.shape[2],
                                                                 reconstructed.shape[3])


# ==========================================
# 3. Network Architecture (Multi-Scale ASPP)
# ==========================================
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling for Dynamic Multi-Scale Target Detection"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.b1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.b2 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=2, dilation=2)
        self.b3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=4, dilation=4)
        self.b4 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=6, dilation=6)

        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.relu(self.bn(self.out_conv(out)))


class MemoryBank(nn.Module):
    def __init__(self, num_slots, dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_slots, dim))

    def forward(self, z):
        B, C, H, W = z.shape
        z_flat = z.view(B, C, -1).permute(0, 2, 1)
        attn = F.linear(F.normalize(z_flat, dim=-1), F.normalize(self.memory, dim=-1))
        attn = F.softmax(attn * 10, dim=-1)
        z_recon = torch.matmul(attn, self.memory)
        return z_recon.permute(0, 2, 1).view(B, C, H, W), attn


class MemAE_MultiScale(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(bands, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            SpatialSelfAttention(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            ASPPBlock(128, 128)
        )
        self.mem_bank = MemoryBank(num_slots=100, dim=128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            SpatialSelfAttention(64),
            nn.ConvTranspose2d(64, bands, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z_mem, attn_weights = self.mem_bank(z)
        out = self.decoder(z_mem)
        return out, attn_weights


# ==========================================
# 4. Training & Inference
# ==========================================
def train_and_infer_memae(features, mask=None, name="Model", epochs=25, crop_size=64):
    H, W, B = features.shape
    model = MemAE_MultiScale(bands=B).to(device)
    criterion_recon = DynamicMultiMetricLoss(top_k_ratio=0.05).to(device)

    optimizer = optim.Adam(list(model.parameters()) + list(criterion_recon.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    if mask is None:
        mask = np.ones((H, W), dtype=bool)

    valid_y, valid_x = np.where(mask)
    batches_per_epoch = 75
    loss_history = []

    print(f"\n--- Training {name} ---")
    model.train()
    with tqdm(total=epochs, desc=f"Training {name}") as pbar:
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(batches_per_epoch):
                batch_crops = []
                for _ in range(16):
                    idx = np.random.randint(0, len(valid_y))
                    y, x = min(valid_y[idx], H - crop_size), min(valid_x[idx], W - crop_size)
                    batch_crops.append(features[y:y + crop_size, x:x + crop_size, :].transpose(2, 0, 1))

                bx = torch.tensor(np.stack(batch_crops), dtype=torch.float32).to(device)
                optimizer.zero_grad()

                reconstructed, attn = model(bx)
                entropy_loss = torch.mean(-attn * torch.log(attn + 1e-8))

                recon_loss, _ = criterion_recon(reconstructed, bx)
                loss = recon_loss + (0.0002 * entropy_loss)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / batches_per_epoch
            loss_history.append(avg_loss)

            pbar.set_postfix(loss=f"{avg_loss:.4f}")
            pbar.update(1)

    print(f"Inferring {name} Map...")
    model.eval()
    criterion_recon.eval()

    learned_weights = F.softmax(criterion_recon.loss_weights, dim=0).detach().cpu().numpy()

    window_1d = np.hanning(crop_size)
    window_2d = np.outer(window_1d, window_1d)
    pad_h = (crop_size - H % crop_size) % crop_size
    pad_w = (crop_size - W % crop_size) % crop_size
    padded_data = np.pad(features, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    pH, pW, _ = padded_data.shape

    anomaly_map_sum = np.zeros((pH, pW))
    weight_map = np.zeros((pH, pW))

    stride = crop_size // 2
    r_coords = sorted(list(set(list(range(0, pH - crop_size + 1, stride)) + [pH - crop_size])))
    c_coords = sorted(list(set(list(range(0, pW - crop_size + 1, stride)) + [pW - crop_size])))
    patches = [(r, r + crop_size, c, c + crop_size) for r in r_coords for c in c_coords]

    with torch.no_grad():
        for i, end_i, j, end_j in tqdm(patches, desc=f"Inference {name}"):
            patch = padded_data[i:end_i, j:end_j, :]
            ptx = torch.tensor(patch.transpose(2, 0, 1)[None, ...], dtype=torch.float32).to(device)
            reconstructed, _ = model(ptx)

            _, error_map = criterion_recon(reconstructed, ptx)
            error_np = error_map.squeeze(0).cpu().numpy()

            anomaly_map_sum[i:end_i, j:end_j] += error_np * window_2d
            weight_map[i:end_i, j:end_j] += window_2d

    final_map = (anomaly_map_sum / (weight_map + 1e-8))[:H, :W]

    final_map[:5, :] = 0
    final_map[-5:, :] = 0
    final_map[:, :5] = 0
    final_map[:, -5:] = 0

    return final_map, learned_weights, loss_history


# ==========================================
# 5. Presentation & Visualization
# ==========================================
def get_display_image(cube, is_swir=False):
    H, W, B = cube.shape
    if is_swir:
        rgb = np.stack([
            cube[:, :, min(int(B * 0.8), B - 1)],
            cube[:, :, min(int(B * 0.5), B - 1)],
            cube[:, :, min(int(B * 0.2), B - 1)]
        ], axis=-1)
    else:
        rgb = np.stack([
            cube[:, :, min(100, B - 1)],
            cube[:, :, min(50, B - 1)],
            cube[:, :, min(10, B - 1)]
        ], axis=-1)

    for i in range(3):
        c_min, c_max = np.percentile(rgb[:, :, i], 1), np.percentile(rgb[:, :, i], 99)
        rgb[:, :, i] = np.clip((rgb[:, :, i] - c_min) / (c_max - c_min + 1e-8), 0, 1)
    return rgb


def generate_solid_overlay(anomaly_map, mask_area=None, top_percent=1.5):
    if mask_area is None:
        mask_area = np.ones_like(anomaly_map, dtype=bool)

    thresh = np.percentile(anomaly_map[mask_area], 100 - top_percent)
    binary_mask = anomaly_map > thresh

    structuring_element = np.ones((3, 3), dtype=bool)
    solid_mask = scipy.ndimage.binary_closing(binary_mask, structure=structuring_element)
    solid_mask = scipy.ndimage.binary_dilation(solid_mask, structure=np.ones((2, 2)))

    H, W = anomaly_map.shape
    overlay = np.zeros((H, W, 4))
    overlay[solid_mask] = [1, 0, 0, 1]
    return overlay


def format_weights_text(weights):
    w = weights.reshape(3, 3) * 100
    text = (
        "Learned Metric Importance:\n"
        "           SAM | Pearson | SID  \n"
        "------------------------------------\n"
        f"Min-Max: {w[0, 0]:>4.1f}% | {w[0, 1]:>5.1f}% | {w[0, 2]:>4.1f}%\n"
        f"Z-Score: {w[1, 0]:>4.1f}% | {w[1, 1]:>5.1f}% | {w[1, 2]:>4.1f}%\n"
        f"BtchNrm: {w[2, 0]:>4.1f}% | {w[2, 1]:>5.1f}% | {w[2, 2]:>4.1f}%"
    )
    return text


def plot_single_cube_results(scene_img, anomaly_map, learned_weights, loss_history):
    overlay = generate_solid_overlay(anomaly_map, top_percent=1.0)

    fig1, axes = plt.subplots(1, 3, figsize=(18, 8))
    axes[0].imshow(scene_img)
    axes[0].set_title("Original Scene")
    axes[0].axis('off')

    im = axes[1].imshow(np.clip(anomaly_map, 0, np.percentile(anomaly_map, 95.0)), cmap='turbo')
    axes[1].set_title("Multi-Metric Heatmap")
    axes[1].axis('off')

    axes[2].imshow(scene_img)
    axes[2].imshow(overlay)
    axes[2].set_title("Severe Anomalies")
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()

    fig2 = plt.figure(figsize=(14, 12))
    gs = fig2.add_gridspec(2, 2, width_ratios=[1, 1.2])

    ax_img = fig2.add_subplot(gs[:, 0])
    ax_img.imshow(scene_img)
    ax_img.imshow(overlay)
    ax_img.set_title("Detected Targets", fontsize=16, fontweight='bold')
    ax_img.axis('off')

    ax_loss = fig2.add_subplot(gs[0, 1])
    ax_loss.plot(range(1, len(loss_history) + 1), loss_history, 'b-', marker='o', linewidth=2)
    ax_loss.set_title("Training Loss Curve", fontsize=14, fontweight='bold')
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Multi-Metric Top-K Loss", fontsize=12)
    ax_loss.grid(True, linestyle='--', alpha=0.7)

    ax_text = fig2.add_subplot(gs[1, 1])
    ax_text.axis('off')
    props = dict(boxstyle='round,pad=0.5', facecolor='#1e1e1e', edgecolor='yellow', linewidth=2)
    ax_text.text(0.5, 0.5, format_weights_text(learned_weights),
                 transform=ax_text.transAxes, fontsize=15, color='white',
                 ha='center', va='center', bbox=props, family='monospace')

    plt.tight_layout()
    plt.show()


def plot_dual_cube_results(vnir_rgb, swir_false, vnir_map, swir_map, fused_map, overlap_mask, fused_weights,
                           loss_history):
    overlay = generate_solid_overlay(fused_map, overlap_mask, top_percent=1.0)

    fig1, axes = plt.subplots(1, 5, figsize=(25, 12))
    axes[0].imshow(swir_false)
    axes[0].set_title("Aligned SWIR Scene")
    axes[0].axis('off')

    axes[1].imshow(np.clip(vnir_map, 0, np.percentile(vnir_map, 95.0)), cmap='turbo')
    axes[1].set_title("VNIR-Only")
    axes[1].axis('off')

    axes[2].imshow(np.clip(swir_map, 0, np.percentile(swir_map, 95.0)), cmap='turbo')
    axes[2].set_title("SWIR-Only")
    axes[2].axis('off')

    axes[3].imshow(np.clip(fused_map, 0, np.percentile(fused_map, 95.0)), cmap='turbo')
    axes[3].set_title("Fused VNIR+SWIR Detection")
    axes[3].axis('off')

    axes[4].imshow(swir_false)
    axes[4].imshow(overlay)
    axes[4].set_title("Grouped Anomalies (Fused)")
    axes[4].axis('off')
    plt.tight_layout()
    plt.show()

    fig2 = plt.figure(figsize=(14, 12))
    gs = fig2.add_gridspec(2, 2, width_ratios=[1, 1.2])

    ax_img = fig2.add_subplot(gs[:, 0])
    ax_img.imshow(swir_false)
    ax_img.imshow(overlay)
    ax_img.set_title("High-Confidence Targets", fontsize=16, fontweight='bold')
    ax_img.axis('off')

    ax_loss = fig2.add_subplot(gs[0, 1])
    ax_loss.plot(range(1, len(loss_history) + 1), loss_history, 'b-', marker='o', linewidth=2)
    ax_loss.set_title("Fused Model Training Loss Curve", fontsize=14, fontweight='bold')
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Multi-Metric Top-K Loss", fontsize=12)
    ax_loss.grid(True, linestyle='--', alpha=0.7)

    ax_text = fig2.add_subplot(gs[1, 1])
    ax_text.axis('off')
    props = dict(boxstyle='round,pad=0.5', facecolor='#1e1e1e', edgecolor='yellow', linewidth=2)
    ax_text.text(0.5, 0.5, format_weights_text(fused_weights),
                 transform=ax_text.transAxes, fontsize=15, color='white',
                 ha='center', va='center', bbox=props, family='monospace')

    plt.tight_layout()
    plt.show()


# ==========================================
# 6. Dynamic Execution Pipeline
# ==========================================
def run_pipeline(vnir_path=None, swir_path=None):
    if vnir_path is None and swir_path is None:
        raise ValueError("You must provide at least one valid file path (VNIR or SWIR).")

    if vnir_path and swir_path:
        print("\n--- Initializing Dual-Cube Fusion Pipeline ---")
        vnir_np = load_reflectance_cube(vnir_path)
        swir_np = load_reflectance_cube(swir_path)

        aligned_vnir, overlap_mask = register_and_mask_cubes(vnir_np, swir_np)
        fused_cube = np.concatenate((aligned_vnir, swir_np), axis=2)

        # Note: crop_size shifted to 64 to allow the ASPP layer to scan large areas
        vnir_map, _, _ = train_and_infer_memae(aligned_vnir, overlap_mask, name="VNIR Model", crop_size=64)
        swir_map, _, _ = train_and_infer_memae(swir_np, overlap_mask, name="SWIR Model", crop_size=64)
        fused_map, fused_weights, fused_loss_history = train_and_infer_memae(fused_cube, overlap_mask,
                                                                             name="Fused VNIR+SWIR", crop_size=64)

        vnir_map[~overlap_mask] = 0
        fused_map[~overlap_mask] = 0

        vnir_rgb = get_display_image(aligned_vnir, is_swir=False)
        swir_false = get_display_image(swir_np, is_swir=True)
        plot_dual_cube_results(vnir_rgb, swir_false, vnir_map, swir_map, fused_map, overlap_mask, fused_weights,
                               fused_loss_history)

    elif vnir_path:
        print("\n--- Initializing Single-Cube Pipeline (VNIR) ---")
        vnir_np = load_reflectance_cube(vnir_path)
        vnir_map, vnir_weights, vnir_loss = train_and_infer_memae(vnir_np, mask=None, name="VNIR Model", crop_size=64)
        vnir_rgb = get_display_image(vnir_np, is_swir=False)
        plot_single_cube_results(vnir_rgb, vnir_map, vnir_weights, vnir_loss)

    elif swir_path:
        print("\n--- Initializing Single-Cube Pipeline (SWIR) ---")
        swir_np = load_reflectance_cube(swir_path)
        swir_map, swir_weights, swir_loss = train_and_infer_memae(swir_np, mask=None, name="SWIR Model", crop_size=64)
        swir_false = get_display_image(swir_np, is_swir=True)
        plot_single_cube_results(swir_false, swir_map, swir_weights, swir_loss)


if __name__ == "__main__":
    vnir_filepath = r"C:\Users\Public\HyperData\ADERET\100m\raw_7024_rd_rf.hdr"
    swir_filepath = r"C:\Users\Public\HyperData\ADERET\100m\swir\spectralview\100134_78910_2024_04_02_09_52_05\raw_7049_rd_rf.hdr"
    run_pipeline(vnir_path=vnir_filepath, swir_path=swir_filepath)