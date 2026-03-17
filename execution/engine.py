# execution/engine.py
import matplotlib.pyplot as plt
from typing import List
from models.base import BaseDetector
from visualization.automated_dash import InteractiveDashboard
from visualization.manual_inspector import ManualSpectralInspector


class PipelineEngine:
    """Orchestrates Automated AI Detectors and Manual Inspection Tools."""

    def __init__(self, detectors: List[BaseDetector]):
        self.detectors = detectors
        self.dashboards = []
        self.inspectors = []

    def run_automated_evaluation(self, name: str, cube, rgb_bg, semantic_weights):
        if cube is None or not self.detectors: return
        print(f"\n[{name}] Running Automated AI Detectors...")

        heatmaps = {}
        for detector in self.detectors:
            raw_map = detector.detect(cube)
            heatmaps[detector.name] = raw_map * semantic_weights

        db = InteractiveDashboard(rgb_bg, heatmaps, name)
        self.dashboards.append(db)

    def run_manual_inspector(self, cube, rgb_bg, wavelengths):
        if cube is None: return
        print(f"\n[Manual Tool] Launching Interactive Spectral Inspector...")
        inspector = ManualSpectralInspector(cube, rgb_bg, wavelengths)
        inspector.launch()
        self.inspectors.append(inspector)

    def show_all(self):
        print("\n✅ Pipeline Execution Complete! All interactive windows are active.")
        plt.show(block=True)