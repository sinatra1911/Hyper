# visualization/automated_dash.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.ndimage


def get_display_image(raw_vnir_cube):
    H, W, B = raw_vnir_cube.shape
    r, g, b = (29, 19, 9) if B == 224 else (min(int(B * .15), B - 1), min(int(B * .1), B - 1), min(int(B * .04), B - 1))
    rgb = np.stack([raw_vnir_cube[:, :, r], raw_vnir_cube[:, :, g], raw_vnir_cube[:, :, b]], axis=-1)
    for i in range(3):
        c_min, c_max = np.percentile(rgb[:, :, i], 1), np.percentile(rgb[:, :, i], 99.5)
        rgb[:, :, i] = np.clip((rgb[:, :, i] - c_min) / (c_max - c_min + 1e-8), 0, 1)
    return rgb


class InteractiveDashboard:
    def __init__(self, scene_img, heatmaps_dict, modality_name):
        self.num_algos = len(heatmaps_dict)
        total_panels = self.num_algos + 2

        # 1. OPTIMIZED SIZE: Wider figure, taller height to let images breathe
        self.fig = plt.figure(figsize=(16, 6.5))
        self.fig.canvas.manager.set_window_title(f"Automated AI Dashboard: {modality_name}")

        # 2. OPTIMIZED MARGINS: wspace=0.05 eliminates empty gaps, forcing images to grow larger
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.25, top=0.90, wspace=0.05)
        gs = self.fig.add_gridspec(1, total_panels)

        self.axes, self.sliders, self.heatmap_imgs = [], [], []
        self.heatmaps_data = list(heatmaps_dict.values())

        # Panel 1: Original
        ax_orig = self.fig.add_subplot(gs[0])
        ax_orig.imshow(scene_img)
        ax_orig.set_title("Original Scene", fontsize=11, fontweight='bold')
        ax_orig.axis('off')
        self.axes.append(ax_orig)

        idx = 1
        for name, heatmap in heatmaps_dict.items():
            ax = self.fig.add_subplot(gs[idx], sharex=self.axes[0], sharey=self.axes[0])
            img = ax.imshow(np.clip(heatmap, 0, np.percentile(heatmap, 95.0)), cmap='turbo')
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.axis('off')
            self.axes.append(ax)
            self.heatmap_imgs.append(img)

            # 3. OPTIMIZED SLIDERS: Mathematically center them and shrink the track width
            # This guarantees the text label ("Thr %") has room on the left side to exist without overlapping.
            slider_width = 0.08
            center_x = (idx + 0.5) / total_panels
            left_pos = center_x - (slider_width / 2)

            ax_sld = self.fig.add_axes([left_pos, 0.10, slider_width, 0.03])
            sld = Slider(ax_sld, 'Thr %', 80.0, 99.99, valinit=95.0, valstep=0.1)

            def make_update(i_obj, data):
                def update(val):
                    vmax = np.percentile(data, val)
                    i_obj.set_data(np.clip(data, 0, vmax))
                    i_obj.set_clim(vmin=0, vmax=vmax)
                    self.update_fusion()

                return update

            sld.on_changed(make_update(img, heatmap))
            self.sliders.append(sld)
            idx += 1

        # Panel Final: Detections
        self.ax_det = self.fig.add_subplot(gs[idx], sharex=self.axes[0], sharey=self.axes[0])
        self.ax_det.imshow(scene_img)
        self.det_img = self.ax_det.imshow(np.zeros((scene_img.shape[0], scene_img.shape[1], 4)))
        self.ax_det.set_title("Voting Detections", fontsize=11, fontweight='bold')
        self.ax_det.axis('off')
        self.axes.append(self.ax_det)

        # Voting Slider Alignment
        slider_width = 0.08
        center_x = (idx + 0.5) / total_panels
        left_pos = center_x - (slider_width / 2)

        ax_vote = self.fig.add_axes([left_pos, 0.10, slider_width, 0.03])
        self.slider_vote = Slider(ax_vote, 'Votes', 1, self.num_algos, valinit=max(1, self.num_algos - 1), valstep=1)
        self.slider_vote.on_changed(lambda val: self.update_fusion())
        self.sliders.append(self.slider_vote)

        self.update_fusion()
        plt.show(block=False)

    def update_fusion(self):
        H, W = self.heatmaps_data[0].shape
        vote_map = np.zeros((H, W), dtype=np.float32)

        for i, data in enumerate(self.heatmaps_data):
            thresh = np.percentile(data, self.sliders[i].val)
            vote_map += (data > thresh).astype(np.float32)

        final_mask = vote_map >= int(self.slider_vote.val)
        solid_mask = scipy.ndimage.binary_dilation(scipy.ndimage.binary_opening(final_mask, np.ones((2, 2))),
                                                   np.ones((2, 2)))

        overlay = np.zeros((H, W, 4))
        overlay[solid_mask] = [0, 1, 1, 1]
        self.det_img.set_data(overlay)
        self.fig.canvas.draw_idle()