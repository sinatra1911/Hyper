import numpy as np
from sklearn.decomposition import PCA

class SemanticSuppressor:
    @staticmethod
    def compute_weights(cube, vnir_raw, red_idx, nir_idx):
        print("  ➤ Computing Soft Semantic Background Weights...")
        H, W, B = cube.shape

        ndvi = (vnir_raw[:, :, nir_idx] - vnir_raw[:, :, red_idx]) / (vnir_raw[:, :, nir_idx] + vnir_raw[:, :, red_idx] + 1e-8)
        veg_score = np.clip((ndvi - 0.15) / 0.4, 0, 1)

        X = cube.reshape(-1, B)
        X_pca = PCA(n_components=5).fit_transform(X)
        X_recon = PCA(n_components=5).fit(X).inverse_transform(X_pca)

        res = np.linalg.norm(X - X_recon, axis=1).reshape(H, W)
        res_norm = (res - res.min()) / (res.max() - res.min() + 1e-8)
        soil_score = np.clip(1.0 - (res_norm * 3.0), 0, 1)

        grad = np.mean(np.abs(np.diff(cube, axis=2)), axis=2)
        weirdness = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

        multiplier = (1.0 - veg_score) * (1.0 - soil_score) * (weirdness + 0.1)
        return (multiplier - multiplier.min()) / (multiplier.max() - multiplier.min() + 1e-8)