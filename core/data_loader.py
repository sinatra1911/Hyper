# core/data_loader.py
import os
import cv2
import numpy as np
import scipy.ndimage
import scipy.io as sio
import spectral as spy


class HyperspectralLoader:

    @staticmethod
    def get_display_image(cube, wavelengths):
        H, W, B = cube.shape
        rgb_targets = [650, 550, 450]
        if wavelengths is not None and len(wavelengths) == B:
            bands = [int(np.abs(wavelengths - t).argmin()) for t in rgb_targets]
        else:
            bands = [min(int(B * .15), B - 1), min(int(B * .1), B - 1), min(int(B * .04), B - 1)]

        rgb = np.dstack([cube[:, :, b] for b in bands]).astype(float)
        low, high = np.percentile(rgb, (1, 99))
        rgb = np.clip((rgb - low) / (high - low + 1e-8), 0, 1)
        return np.power(rgb, 0.8)

    @classmethod
    def load_modalities(cls, vnir_path, swir_path=None):
        print(f"Loading Data: {vnir_path}...")
        img_meta = spy.open_image(vnir_path)
        vnir_raw = np.asarray(img_meta.load(), dtype=np.float32)

        wavelengths = None
        red_idx, nir_idx = 29, 50
        if img_meta.bands.centers:
            wl = np.array(img_meta.bands.centers)
            wavelengths = wl * 1000 if np.max(wl) < 100 else wl
            red_idx = np.argmin(np.abs(wavelengths - 650))
            nir_idx = np.argmin(np.abs(wavelengths - 850))

        vnir_clean = np.clip(vnir_raw, 0, None)
        vnir_clean /= (np.linalg.norm(vnir_clean, axis=-1, keepdims=True) + 1e-8)

        # Build payload
        payload = {
            'vnir': vnir_clean, 'fused': vnir_clean, 'raw_vnir': vnir_raw,
            'wavelengths': wavelengths, 'r_idx': red_idx, 'n_idx': nir_idx
        }
        return payload  # (Truncated SWIR loading for brevity, same as previous logic)