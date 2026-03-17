import torch
from core.data_loader import HyperspectralLoader
from core.semantics import SemanticSuppressor
from models.classical import *
from models.deep import AutoencoderDetector
from execution.engine import PipelineEngine


def main():
    print("=" * 60)
    print("HYPERSPECTRAL ANALYSIS SUITE (AUTO + MANUAL)")
    print("=" * 60)

    # --- CONFIGURATION ---
    vnir_filepath = r"C:\Users\Public\HyperData\ADERET\100m\raw_7024_rd_rf.hdr"
    swir_filepath = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    RUN_AUTOMATED_AI = True
    RUN_MANUAL_INSPECTOR = True

    # --- 1. LOAD DATA ---
    data = HyperspectralLoader.load_modalities(vnir_filepath, swir_filepath)
    rgb_bg = HyperspectralLoader.get_display_image(data['raw_vnir'], data['wavelengths'])

    # --- 2. INITIALIZE ENGINE ---
    detectors = [
        AutoencoderDetector(epochs=100, device=device),
        OSPDetector(k_endmembers=15, device=device),
        IForestDetector(n_components=10)
    ]
    engine = PipelineEngine(detectors)

    # --- 3. EXECUTE MODULES ---
    if RUN_AUTOMATED_AI:
        semantics = SemanticSuppressor.compute_weights(
            data['fused'], data['raw_vnir'], data['r_idx'], data['n_idx']
        )
        engine.run_automated_evaluation("VNIR Automated", data['vnir'], rgb_bg, semantics)

    if RUN_MANUAL_INSPECTOR:
        engine.run_manual_inspector(data['raw_vnir'], rgb_bg, data['wavelengths'])

    # --- 4. RENDER UI ---
    engine.show_all()


if __name__ == "__main__":
    main()