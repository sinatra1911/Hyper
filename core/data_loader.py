# core/data_loader.py
import os
import cv2
import numpy as np
import scipy.ndimage
import scipy.io as sio
import h5py
import spectral as spy


class HyperspectralLoader:
    """Handles parsing, resizing, and Fourier Phase Alignment of HS Data."""

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

    @staticmethod
    def _auto_align(vnir_raw, swir_raw):
        print("  ➤ Auto-Aligning SWIR to VNIR (Fourier Phase Correlation)...")
        swir_resized = cv2.resize(swir_raw, (vnir_raw.shape[1], vnir_raw.shape[0]), interpolation=cv2.INTER_LINEAR)
        vnir_gray = cv2.normalize(np.mean(vnir_raw, axis=2).astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        swir_gray = cv2.normalize(np.mean(swir_resized, axis=2).astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

        transformations = [
            ("None", lambda x: x),
            ("Flip Up/Down", lambda x: np.flipud(x)),
            ("Flip Left/Right", lambda x: np.fliplr(x)),
            ("Flip Both", lambda x: np.flipud(np.fliplr(x)))
        ]

        best_resp, best_shift, best_func, best_name = -1, (0, 0), None, ""
        for name, func in transformations:
            shift, resp = cv2.phaseCorrelate(func(swir_gray).astype(np.float32), vnir_gray)
            if resp > best_resp:
                best_resp, best_shift, best_func, best_name = resp, shift, func, name

        print(f"    ↳ Alignment: [{best_name}], Shift: ({best_shift[0]:.2f}, {best_shift[1]:.2f})")
        aligned = best_func(swir_resized)
        return scipy.ndimage.shift(aligned, shift=[best_shift[1], best_shift[0], 0], mode='nearest')

    @classmethod
    def load_modalities(cls, vnir_path, swir_path=None):
        print(f"Loading Primary VNIR: {vnir_path}...")
        ext = os.path.splitext(vnir_path)[1].lower()

        wavelengths = None
        red_idx, nir_idx = 29, 50
        vnir_raw = None

        if ext == '.mat':
            try:
                mat_data = sio.loadmat(vnir_path)
                max_size = 0
                for key, val in mat_data.items():
                    if isinstance(val, np.ndarray) and val.ndim == 3:
                        if val.size > max_size:
                            max_size = val.size
                            vnir_raw = val.astype(np.float32)
            except NotImplementedError:
                # Fallback for massive v7.3 .mat files
                with h5py.File(vnir_path, 'r') as f:
                    max_size = 0
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 3:
                            if f[key].size > max_size:
                                max_size = f[key].size
                                vnir_raw = np.array(f[key]).transpose(2, 1, 0).astype(np.float32)

            if vnir_raw is None:
                raise ValueError("Could not find a valid 3D hyperspectral array in the .mat file.")

        else:
            # hdr file
            img_meta = spy.open_image(vnir_path)
            vnir_raw = np.asarray(img_meta.load(), dtype=np.float32)

            if img_meta.bands.centers:
                wl = np.array(img_meta.bands.centers)
                wavelengths = wl * 1000 if np.max(wl) < 100 else wl
                red_idx = np.argmin(np.abs(wavelengths - 650))
                nir_idx = np.argmin(np.abs(wavelengths - 850))
            elif vnir_raw.shape[2] == 271:
                red_idx, nir_idx = 113, 203
        # ---------------------------------------------------------

        # Safety bounds in case the user loads a tiny cube
        red_idx = min(red_idx, vnir_raw.shape[2] - 1)
        nir_idx = min(nir_idx, vnir_raw.shape[2] - 1)

        vnir_clean = np.clip(vnir_raw, 0, None)
        vnir_clean /= (np.linalg.norm(vnir_clean, axis=-1, keepdims=True) + 1e-8)

        if not swir_path:
            return {
                'vnir': vnir_clean, 'swir': None, 'fused': vnir_clean,
                'raw_vnir': vnir_raw, 'wavelengths': wavelengths,
                'r_idx': red_idx, 'n_idx': nir_idx
            }

        print(f"Loading Secondary SWIR: {swir_path}...")
        swir_ext = os.path.splitext(swir_path)[1].lower()
        if swir_ext == '.mat':
            swir_raw = sio.loadmat(swir_path)[list(sio.loadmat(swir_path).keys())[-1]].astype(np.float32)
        else:
            swir_raw = np.asarray(spy.open_image(swir_path).load(), dtype=np.float32)

        swir_aligned = cls._auto_align(vnir_raw, swir_raw)

        swir_clean = np.clip(swir_aligned, 0, None)
        swir_clean /= (np.linalg.norm(swir_clean, axis=-1, keepdims=True) + 1e-8)

        fused_raw = np.concatenate((vnir_raw, swir_aligned), axis=-1)
        fused_clean = np.clip(fused_raw, 0, None)
        fused_clean /= (np.linalg.norm(fused_clean, axis=-1, keepdims=True) + 1e-8)

        return {
            'vnir': vnir_clean, 'swir': swir_clean, 'fused': fused_clean,
            'raw_vnir': vnir_raw, 'wavelengths': wavelengths,
            'r_idx': red_idx, 'n_idx': nir_idx
        }