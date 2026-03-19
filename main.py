import torch
from core.data_loader import HyperspectralLoader
from core.semantics import SemanticSuppressor
from models.classical import OSPDetector, LocalRXDetector
from models.deep import *
from execution.engine import PipelineEngine

def main():
    print("=" * 60)
    print("HYPERSPECTRAL ANALYSIS SUITE (AUTO + MANUAL)")
    print("=" * 60)

    # --- CONFIGURATION ---
    primary_filepath = r"C:\Users\Public\HyperData\ADERET\50m-adarat\vnir\spectralview\100119_Aderet_2_4_24_50M_2024_04_02_08_36_30\22303\raw_22303_rd_rf.hdr"
    secondary_filepath = r"C:\Users\Public\HyperData\ADERET\50m-adarat\swir\spectralview\22256\raw_22256_rd_rf.hdr"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    RUN_AUTOMATED_AI = True
    RUN_MANUAL_INSPECTOR = True
    EVALUATE_ALL_MODALITIES = False

    # --- 1. LOAD DATA ---
    data = HyperspectralLoader.load_modalities(primary_filepath, secondary_filepath)
    rgb_bg = HyperspectralLoader.get_display_image(data['raw_primary'], data['wl_primary'])

    # --- 2. INITIALIZE ENGINE ---
    detectors = [
        OSPDetector(k_endmembers=15, device=device),
        LocalRXDetector(device=device),
        AutoencoderDetector(device=device),
    ]
    engine = PipelineEngine(detectors)

    # --- 3. EXECUTE MODULES ---
    target_name = "FUSED (Primary + Secondary)" if data['secondary'] is not None else "Primary Cube"

    if RUN_MANUAL_INSPECTOR:
        # NEW: We pass the split_idx so the inspector knows where the sensors divide
        engine.run_manual_inspector(data['raw_fused'], rgb_bg, data['wl_fused'], data['split_idx'])

    if RUN_AUTOMATED_AI:
        sem_fused = SemanticSuppressor.compute_weights(data['fused'], data['raw_primary'], data['r_idx'], data['n_idx'])
        engine.run_automated_evaluation(target_name, data['fused'], rgb_bg, sem_fused)

        if EVALUATE_ALL_MODALITIES and data['secondary'] is not None:
            sem_pri = SemanticSuppressor.compute_weights(data['primary'], data['raw_primary'], data['r_idx'],
                                                         data['n_idx'])
            engine.run_automated_evaluation("Primary (VNIR) Only", data['primary'], rgb_bg, sem_pri)

            sem_sec = SemanticSuppressor.compute_weights(data['secondary'], data['raw_primary'], data['r_idx'],
                                                         data['n_idx'])
            engine.run_automated_evaluation("Secondary (SWIR) Only", data['secondary'], rgb_bg, sem_sec)

    # --- 4. RENDER UI ---
    engine.show_all()


if __name__ == "__main__":
    main()