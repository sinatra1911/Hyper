import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from .base import BaseDetector

class _SpectralAutoencoderNN(nn.Module):
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

class AutoencoderDetector(BaseDetector):
    def __init__(self, epochs=100, batch_size=2048, device='cuda'):
        super().__init__("Deep Autoencoder")
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def detect(self, cube: np.ndarray) -> np.ndarray:
        print(f"  ➤ Running {self.name} (Self-Supervised Learner)...")
        H, W, B = cube.shape
        X_flat = cube.reshape(-1, B)

        # Sample 10% of the image for fast background memorization
        num_samples = int(H * W * 0.1)
        indices = np.random.choice(H * W, num_samples, replace=False)
        train_data = torch.tensor(X_flat[indices], dtype=torch.float32, device=self.device)

        model = _SpectralAutoencoderNN(input_dim=B).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        model.train()
        dataset = torch.utils.data.TensorDataset(train_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        pbar = tqdm(range(self.epochs), desc="Training Epochs", leave=False, bar_format="{l_bar}{bar:30}{r_bar}")

        for epoch in pbar:
            epoch_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                reconstructed = model(batch[0])
                loss = F.mse_loss(reconstructed, batch[0])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Update the progress bar suffix to show the falling loss
            avg_loss = epoch_loss / len(dataloader)
            pbar.set_postfix({'MSE Loss': f"{avg_loss:.5f}"})

        pbar.close()
        # ---------------------------------------------------------

        model.eval()
        X_tensor = torch.tensor(X_flat, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            reconstructed_full = model(X_tensor)
            mse_error = torch.mean((reconstructed_full - X_tensor) ** 2, dim=1)
            cos_error = 1.0 - F.cosine_similarity(reconstructed_full, X_tensor, dim=1)
            total_error = mse_error + (0.5 * cos_error)

        ae_map = total_error.view(H, W).cpu().numpy()
        return (ae_map - np.min(ae_map)) / (np.max(ae_map) - np.min(ae_map) + 1e-8)