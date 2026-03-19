# models/deep.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from .base import BaseDetector


class _SpectralAutoencoderNN(nn.Module):
    def __init__(self, input_dim, bottleneck=16):
        super().__init__()
        # Added Batch Normalization to prevent Mode Collapse
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # REMOVED SIGMOID: L2 Normalized targets are tiny (~0.05).
            # A linear output prevents the gradients from dying.
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderDetector(BaseDetector):
    def __init__(self, epochs=80, batch_size=2048, device='cuda'):
        super().__init__("Deep Autoencoder")
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def detect(self, cube: np.ndarray) -> np.ndarray:
        print(f"  ➤ Running {self.name} (Self-Supervised Learner)...")
        H, W, B = cube.shape
        X_flat = cube.reshape(-1, B)

        # 1. Safely filter out pure-zero padded pixels from the alignment process
        # so they don't corrupt the training gradients
        pixel_sums = np.sum(np.abs(X_flat), axis=1)
        valid_pixel_mask = pixel_sums > 1e-5
        valid_indices = np.where(valid_pixel_mask)[0]

        # Fallback just in case the image is completely clean
        if len(valid_indices) == 0:
            valid_indices = np.arange(len(X_flat))

        # Train on a random 10% sample of valid background
        num_samples = int(len(valid_indices) * 0.1)
        train_indices = np.random.choice(valid_indices, num_samples, replace=False)
        train_data = torch.tensor(X_flat[train_indices], dtype=torch.float32, device=self.device)

        model = _SpectralAutoencoderNN(input_dim=B).to(self.device)
        # Lowered learning rate slightly for stability
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        model.train()
        dataset = torch.utils.data.TensorDataset(train_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        pbar = tqdm(range(self.epochs), desc="    Training Epochs", leave=False, bar_format="{l_bar}{bar:30}{r_bar}")
        for epoch in pbar:
            epoch_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                reconstructed = model(batch[0])
                loss = F.mse_loss(reconstructed, batch[0])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            pbar.set_postfix({'MSE Loss': f"{epoch_loss / len(dataloader):.6f}"})
        pbar.close()

        # 2. VRAM-Safe Inference Chunking
        # Doing the whole image at once blows up GPU memory and causes silent corruption
        model.eval()
        total_error = torch.zeros(H * W, device=self.device)
        chunk_size = 50000

        print("  ➤ Evaluating Image...")
        with torch.no_grad():
            for i in range(0, H * W, chunk_size):
                end = min(i + chunk_size, H * W)
                x_chunk = torch.tensor(X_flat[i:end], dtype=torch.float32, device=self.device)

                recon_chunk = model(x_chunk)

                mse_error = torch.mean((recon_chunk - x_chunk) ** 2, dim=1)
                cos_error = 1.0 - F.cosine_similarity(recon_chunk, x_chunk, dim=1)

                total_error[i:end] = mse_error + (0.5 * cos_error)

        # 3. Explicitly apply the mask using a native Torch Tensor to prevent silent Numpy indexing bugs
        mask_tensor = torch.tensor(~valid_pixel_mask, device=self.device)
        total_error[mask_tensor] = 0.0

        ae_map = total_error.view(H, W).cpu().numpy()

        # Normalize the final heatmap
        map_min, map_max = np.min(ae_map), np.max(ae_map)
        return (ae_map - map_min) / (map_max - map_min + 1e-8)