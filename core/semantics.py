# core/semantics.py
import numpy as np


class SemanticSuppressor:
    @staticmethod
    def compute_weights(cube, raw_primary, red_idx, nir_idx):
        print("  ➤ Computing Safe Semantic Background Weights...")
        H, W, B = cube.shape

        # 1. Confident Vegetation Suppression (NDVI)
        if red_idx != nir_idx and raw_primary.shape[2] > max(red_idx, nir_idx):
            nir_band = raw_primary[:, :, nir_idx]
            red_band = raw_primary[:, :, red_idx]
            ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
            # Soft suppression starts at 0.15 NDVI
            veg_score = np.clip((ndvi - 0.15) / 0.4, 0, 1)
        else:
            veg_score = np.zeros((H, W), dtype=np.float32)

        # 2. Confident Shadow/Water Suppression (Albedo/Brightness)
        # Metals are highly reflective. Dark pixels are usually shadows or water.
        albedo = np.mean(cube, axis=2)
        albedo_norm = (albedo - albedo.min()) / (albedo.max() - albedo.min() + 1e-8)

        # Aggressively suppress the darkest 5-10% of the image
        shadow_score = np.clip(1.0 - (albedo_norm * 15.0), 0, 1)

        # 3. Safe Multiplier
        # We NO LONGER penalize smooth spectra. Metals will easily survive this.
        # We rely entirely on the Autoencoder and RX algorithms to separate the rocks from the metals.
        multiplier = (1.0 - veg_score) * (1.0 - shadow_score)

        return multiplier