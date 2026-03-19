import os
import cv2
import numpy as np
import scipy.ndimage
import scipy.io as sio
import h5py
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

    @staticmethod
    def _auto_align(pri_raw, sec_raw):
        print("  ➤ Auto-Aligning Secondary to Primary (Fourier Phase Correlation)...")
        sec_resized = cv2.resize(sec_raw, (pri_raw.shape[1], pri_raw.shape[0]), interpolation=cv2.INTER_LINEAR)
        pri_gray = cv2.normalize(np.mean(pri_raw, axis=2).astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        sec_gray = cv2.normalize(np.mean(sec_resized, axis=2).astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

        transformations = [
            ("None", lambda x: x),
            ("Flip Up/Down", lambda x: np.flipud(x)),
            ("Flip Left/Right", lambda x: np.fliplr(x)),
            ("Flip Both", lambda x: np.flipud(np.fliplr(x)))
        ]

        best_resp, best_shift, best_func, best_name = -1, (0, 0), None, ""
        for name, func in transformations:
            shift, resp = cv2.phaseCorrelate(func(sec_gray).astype(np.float32), pri_gray)
            if resp > best_resp:
                best_resp, best_shift, best_func, best_name = resp, shift, func, name

        print(f"    ↳ Alignment: [{best_name}], Shift: ({best_shift[0]:.2f}, {best_shift[1]:.2f})")
        aligned = best_func(sec_resized)
        return scipy.ndimage.shift(aligned, shift=[best_shift[1], best_shift[0], 0], mode='nearest')

    @classmethod
    def _load_cube(cls, path):
        ext = os.path.splitext(path)[1].lower()
        raw, wl = None, None
        if ext == '.mat':
            try:
                mat_data = sio.loadmat(path)
                max_size = 0
                for key, val in mat_data.items():
                    if isinstance(val, np.ndarray) and val.ndim == 3 and val.size > max_size:
                        max_size, raw = val.size, val.astype(np.float32)
            except NotImplementedError:
                with h5py.File(path, 'r') as f:
                    max_size = 0
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 3 and f[key].size > max_size:
                            max_size, raw = f[key].size, np.array(f[key]).transpose(2, 1, 0).astype(np.float32)
        else:
            img_meta = spy.open_image(path)
            raw = np.asarray(img_meta.load(), dtype=np.float32)
            if img_meta.bands.centers:
                wl_arr = np.array(img_meta.bands.centers)
                wl = wl_arr * 1000 if np.max(wl_arr) < 100 else wl_arr

        if raw is None: raise ValueError(f"Could not load valid 3D array from {path}")
        if wl is None: wl = np.arange(raw.shape[2], dtype=np.float32)
        return raw, wl

    @classmethod
    def load_modalities(cls, pri_path, sec_path=None):
        print(f"Loading Primary Data: {pri_path}...")
        pri_raw, pri_wl = cls._load_cube(pri_path)

        red_idx = np.argmin(np.abs(pri_wl - 650)) if np.max(pri_wl) > 600 else 0
        nir_idx = np.argmin(np.abs(pri_wl - 850)) if np.max(pri_wl) > 600 else 0

        pri_clean = np.clip(pri_raw, 0, None)
        pri_clean /= (np.linalg.norm(pri_clean, axis=-1, keepdims=True) + 1e-8)

        # NEW: Log the exact point where Primary ends
        split_idx = pri_raw.shape[2]

        payload = {
            'primary': pri_clean, 'secondary': None, 'fused': pri_clean,
            'raw_primary': pri_raw, 'raw_fused': pri_raw,
            'wl_primary': pri_wl, 'wl_secondary': None, 'wl_fused': pri_wl,
            'r_idx': red_idx, 'n_idx': nir_idx, 'split_idx': split_idx
        }

        if not sec_path:
            return payload

        print(f"Loading Secondary Data: {sec_path}...")
        sec_raw, sec_wl = cls._load_cube(sec_path)
        sec_aligned = cls._auto_align(pri_raw, sec_raw)

        sec_clean = np.clip(sec_aligned, 0, None)
        sec_clean /= (np.linalg.norm(sec_clean, axis=-1, keepdims=True) + 1e-8)

        fused_raw = np.concatenate((pri_raw, sec_aligned), axis=-1)
        fused_clean = np.clip(fused_raw, 0, None)
        fused_clean /= (np.linalg.norm(fused_clean, axis=-1, keepdims=True) + 1e-8)
        fused_wl = np.concatenate((pri_wl, sec_wl))

        payload.update({
            'secondary': sec_clean, 'fused': fused_clean,
            'raw_fused': fused_raw, 'wl_secondary': sec_wl, 'wl_fused': fused_wl
        })
        return payload