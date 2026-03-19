import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
from .base import BaseDetector


class IForestDetector(BaseDetector):
    def __init__(self, n_components=10):
        super().__init__("Isolation Forest")
        self.n_components = n_components

    def detect(self, cube: np.ndarray) -> np.ndarray:
        print(f"  ➤ Running {self.name}...")
        H, W, B = cube.shape
        pca_features = PCA(n_components=self.n_components).fit_transform(cube.reshape(-1, B))
        clf = IsolationForest(n_estimators=100, contamination=0.01, n_jobs=-1, random_state=42)
        clf.fit(pca_features)
        scores = -clf.score_samples(pca_features)

        heatmap = scores.reshape(H, W)
        return (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)


class OSPDetector(BaseDetector):
    def __init__(self, k_endmembers=15, device='cuda'):
        super().__init__("OSP Unmixing")
        self.k_endmembers = k_endmembers
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def detect(self, cube: np.ndarray) -> np.ndarray:
        print(f"  ➤ Running {self.name}...")
        H, W, B = cube.shape
        X = torch.tensor(cube.reshape(-1, B).T, dtype=torch.float32, device=self.device)
        R_mat = torch.matmul(X, X.T) / X.shape[1]
        eigenvalues, eigenvectors = torch.linalg.eigh(R_mat)
        idx = torch.argsort(eigenvalues, descending=True)
        U_bg = eigenvectors[:, idx][:, :self.k_endmembers]

        I = torch.eye(B, device=self.device)
        P_ortho = I - torch.matmul(U_bg, U_bg.T)
        unmixed_residual = torch.matmul(P_ortho, X)

        osp_scores = torch.norm(unmixed_residual, dim=0).view(H, W).cpu().numpy()
        return (osp_scores - np.min(osp_scores)) / (np.max(osp_scores) - np.min(osp_scores) + 1e-8)


class LocalRXDetector(BaseDetector):
    def __init__(self, inner_win=5, outer_win=15, n_components=10, device='cuda'):
        super().__init__("Local RX")
        self.inner_win = inner_win
        self.outer_win = outer_win
        self.n_components = n_components
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def detect(self, cube: np.ndarray) -> np.ndarray:
        print(f"  ➤ Running {self.name} (Memory-Safe)...")
        H, W, B = cube.shape
        pca_cube = PCA(n_components=self.n_components).fit_transform(cube.reshape(-1, B)).reshape(H, W,
                                                                                                  self.n_components)
        cube_tensor = torch.tensor(pca_cube, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        C = self.n_components

        pad = self.outer_win // 2
        padded = F.pad(cube_tensor.unsqueeze(0), (pad, pad, pad, pad), mode='reflect')

        mask = torch.ones((self.outer_win, self.outer_win), dtype=torch.bool, device=self.device)
        c = self.outer_win // 2
        in_r = self.inner_win // 2
        mask[c - in_r:c + in_r + 1, c - in_r:c + in_r + 1] = False
        mask_flat = mask.view(-1)

        rx_scores = torch.zeros(H * W, device=self.device)
        chunk_lines = 100

        pbar = tqdm(total=H, desc="    Scanning Map", leave=False)
        for row_start in range(0, H, chunk_lines):
            row_end = min(row_start + chunk_lines, H)
            slice_padded = padded[:, :, row_start: row_end + 2 * pad, :]

            unfold = nn.Unfold(kernel_size=self.outer_win)
            patches = unfold(slice_padded).view(C, self.outer_win ** 2, -1).permute(2, 0, 1)
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
                cov = cov + torch.eye(C, device=self.device).unsqueeze(0) * 1e-4
                cov_inv = torch.linalg.inv(cov)

                x_c = cp_sub - mu
                md = torch.bmm(torch.bmm(x_c.transpose(1, 2), cov_inv), x_c)
                rx_scores[(row_start * W) + i: (row_start * W) + sub_end] = md.squeeze(-1).squeeze(-1)

            pbar.update(row_end - row_start)
        pbar.close()

        local_map = rx_scores.view(H, W).cpu().numpy()
        return (local_map - local_map.min()) / (local_map.max() - local_map.min() + 1e-8)

class CBADDetector(BaseDetector):
    """
    Cluster-Based Anomaly Detection (CBAD).
    Groups the image into semantic natural clusters first, completely eliminating
    false positives from natural background variations like scattered rocks.
    """

    def __init__(self, n_clusters=5, n_components=10, device='cuda'):
        super().__init__("Cluster-Based RX (CBAD)")
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def detect(self, cube: np.ndarray) -> np.ndarray:
        print(f"  ➤ Running {self.name} (Suppressing Natural Clusters)...")
        H, W, B = cube.shape
        X_flat = cube.reshape(-1, B)

        # 1. Dimensionality Reduction for Stability
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(X_flat)

        # 2. Unsupervised Semantic Clustering (Groups all rocks together, all dirt together)
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=2048, n_init='auto')
        labels = kmeans.fit_predict(X_pca)

        # 3. Calculate Anomaly Score relative to the pixel's OWN cluster
        X_tensor = torch.tensor(X_pca, dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(labels, device=self.device)
        anomaly_scores = torch.zeros(H * W, device=self.device)

        for k in range(self.n_clusters):
            cluster_mask = (labels_tensor == k)
            cluster_pixels = X_tensor[cluster_mask]

            if len(cluster_pixels) > self.n_components + 1:
                mu = torch.mean(cluster_pixels, dim=0, keepdim=True)
                centered = cluster_pixels - mu

                # Covariance of just this specific background type (e.g., just the rocks)
                cov = torch.matmul(centered.T, centered) / (len(cluster_pixels) - 1)
                cov += torch.eye(self.n_components, device=self.device) * 1e-4
                cov_inv = torch.linalg.inv(cov)

                # Distance from the cluster average
                dist = torch.sum(torch.matmul(centered, cov_inv) * centered, dim=1)
                anomaly_scores[cluster_mask] = dist

        map_2d = anomaly_scores.view(H, W).cpu().numpy()
        return (map_2d - np.min(map_2d)) / (np.max(map_2d) - np.min(map_2d) + 1e-8)