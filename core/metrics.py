import numpy as np


class SpectralMetrics:
    """Handles all spectral mathematical comparisons and normalizations."""

    @staticmethod
    def normalize_array(data: np.ndarray, mode: str) -> np.ndarray:
        if mode == "None": return data
        if mode == "Min–Max":
            mins = np.min(data, axis=1, keepdims=True)
            p2p = np.ptp(data, axis=1, keepdims=True) + 1e-12
            return (data - mins) / p2p
        if mode == "L2":
            norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-12
            return data / norms
        if mode == "Z-Score":
            m = np.mean(data, axis=1, keepdims=True)
            s = np.std(data, axis=1, keepdims=True) + 1e-12
            return (data - m) / s
        return data

    @staticmethod
    def compute_metrics(mean1: np.ndarray, mean2: np.ndarray) -> dict:
        valid = np.isfinite(mean1) & np.isfinite(mean2)
        a, b = mean1[valid], mean2[valid]
        if a.size == 0: return {"error": "No valid bands."}

        eps = 1e-12
        # SAM
        cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)
        sam = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

        # SID
        a_l1, b_l1 = a / (np.sum(a) + eps), b / (np.sum(b) + eps)
        sid = np.sum(a_l1 * np.log((a_l1 + eps) / (b_l1 + eps))) + np.sum(b_l1 * np.log((b_l1 + eps) / (a_l1 + eps)))

        # Correlation & Euclidean
        corr = np.nan if np.std(a) < eps or np.std(b) < eps else np.corrcoef(a, b)[0, 1]
        euc = np.linalg.norm(a - b)

        return dict(SAM=sam, SID=sid, Corr=corr, Euclid=euc)