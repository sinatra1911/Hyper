import numpy as np
from sklearn.decomposition import PCA

class SemanticSuppressor:
    @staticmethod
    def compute_weights(cube, raw_primary, red_idx, nir_idx):
        print("  ➤ Computing Soft Semantic Background Weights...")
        H, W, B = cube.shape

        # Bypass vegetation masking safely if valid VNIR indices aren't present
        if red_idx != nir_idx and raw_primary.shape[2] > max(red_idx, nir_idx):
            nir_band = raw_primary[:, :, nir_idx]
            red_band = raw_primary[:, :, red_idx]
            ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
            veg_score = np.clip((ndvi - 0.15) / 0.4, 0, 1)
        else:
            veg_score = np.zeros((H, W), dtype=np.float32)

        X = cube.reshape(-1, B)
        n_comp = min(5, B) # Safegaurd for cubes with very few bands
        X_pca = PCA(n_components=n_comp).fit_transform(X)
        X_recon = PCA(n_components=n_comp).fit(X).inverse_transform(X_pca)

        res = np.linalg.norm(X - X_recon, axis=1).reshape(H, W)
        res_norm = (res - res.min()) / (res.max() - res.min() + 1e-8)
        soil_score = np.clip(1.0 - (res_norm * 3.0), 0, 1)

        grad = np.mean(np.abs(np.diff(cube, axis=2)), axis=2)
        weirdness = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

        multiplier = (1.0 - veg_score) * (1.0 - soil_score) * (weirdness + 0.1)
        return (multiplier - multiplier.min()) / (multiplier.max() - multiplier.min() + 1e-8)