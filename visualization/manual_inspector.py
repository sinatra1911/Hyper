import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from core.metrics import SpectralMetrics


class ManualSpectralInspector:
    def __init__(self, cube, rgb_img, wavelengths, split_idx=None):
        self.cube = cube
        self.rgb_img = rgb_img
        self.wavelengths = wavelengths
        self.split_idx = split_idx

        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        self.points = []
        self.color_idx = 0
        self.selected_point = None
        self.last_action = None
        self.last_action_time = 0.0
        self.COOLDOWN = 0.4

        self.object1 = []
        self.object2 = []
        self.leg_map = {}
        self.stat_fig = None
        self.stat_axes = {}

        # NEW: Identify the shifted "dead zones" where SWIR is padded with zeros
        if self.split_idx is not None and self.split_idx < self.cube.shape[2]:
            swir_sum = np.sum(np.abs(self.cube[:, :, self.split_idx:]), axis=-1)
            self.invalid_swir_mask = (swir_sum == 0)
        else:
            self.invalid_swir_mask = np.zeros((self.cube.shape[0], self.cube.shape[1]), dtype=bool)

    def launch(self):
        self.fig, (self.ax_img, self.ax_spec) = plt.subplots(1, 2, figsize=(10, 5.5))
        self.fig.canvas.manager.set_window_title("Manual Spectral Inspector")
        self.fig.subplots_adjust(bottom=0.20, wspace=0.25)

        self.ax_img.imshow(self.rgb_img)

        # NEW: Overlay a faint 30% red tint over areas that are missing SWIR data due to alignment
        if np.any(self.invalid_swir_mask):
            overlay = np.zeros((*self.invalid_swir_mask.shape, 4))
            overlay[self.invalid_swir_mask] = [1, 0, 0, 0.3]
            self.ax_img.imshow(overlay)

        self.ax_img.set_title("Shift+Left Click = Add | Right Click = Move", fontsize=10)
        self.ax_img.axis("off")

        self.ax_spec.set_title("Reflectance Spectra", fontsize=11)
        self.ax_spec.set_xlabel("Wavelength (nm)", fontsize=9)
        self.ax_spec.set_ylabel("Reflectance", fontsize=9)
        self.ax_spec.grid(True)

        self.tooltip = self.ax_img.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                                            arrowprops=dict(arrowstyle="->"))
        self.tooltip.set_visible(False)

        self.btn_obj1 = Button(plt.axes([0.05, 0.05, 0.25, 0.06]), "Save Obj 1 & Clear")
        self.btn_obj1.on_clicked(lambda e: self._take_object(e, 1))
        self.btn_stat = Button(plt.axes([0.375, 0.05, 0.25, 0.06]), "Calc Statistics")
        self.btn_stat.on_clicked(self._show_stat_calc)
        self.btn_obj2 = Button(plt.axes([0.70, 0.05, 0.25, 0.06]), "Save Obj 2 & Clear")
        self.btn_obj2.on_clicked(lambda e: self._take_object(e, 2))

        self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        self.fig.canvas.mpl_connect('pick_event', self._onpick)  # NEW: Interactive Legend Listener
        plt.show(block=False)

    def _onpick(self, event):
        """NEW: Toggles line visibility when its legend entry is clicked."""
        legline = event.artist
        origline = self.leg_map.get(legline)
        if origline:
            vis = not origline.get_visible()
            origline.set_visible(vis)
            legline.set_alpha(1.0 if vis else 0.2)
            self.fig.canvas.draw_idle()

    def _get_gui_shift(self, event):
        if getattr(event, "key", None): return 'shift' in event.key
        gui = getattr(event, "guiEvent", None)
        if gui:
            try:
                return 'Shift' in repr(gui.modifiers()) or (hasattr(gui, "state") and (gui.state & 0x0001) != 0)
            except Exception:
                pass
        return False

    def _plot_spectrum(self, x, y):
        if not (0 <= x < self.cube.shape[1] and 0 <= y < self.cube.shape[0]): return
        if any(p['x'] == x and p['y'] == y for p in self.points): return

        color = self.colors[self.color_idx % len(self.colors)]
        self.color_idx += 1

        spectrum = np.squeeze(self.cube[y, x, :])

        # -------------------------------------------------------------
        # NEW: Line Separation & Invalid SWIR Handling (Without modifying raw arrays)
        # -------------------------------------------------------------
        wl_plot = np.copy(self.wavelengths)
        spec_plot = np.copy(spectrum)

        if self.split_idx is not None and self.split_idx < len(self.wavelengths):
            if self.invalid_swir_mask[y, x]:
                # User clicked in the Red Zone -> SWIR is fake padding, truncate it to VNIR only
                wl_plot, spec_plot = wl_plot[:self.split_idx], spec_plot[:self.split_idx]
            else:
                # User clicked Valid Zone -> Insert NaN to visually break the overlapping line
                wl_plot = np.insert(wl_plot, self.split_idx, np.nan)
                spec_plot = np.insert(spec_plot, self.split_idx, np.nan)
        # -------------------------------------------------------------

        line, = self.ax_spec.plot(wl_plot, spec_plot, color=color, label=f"({x},{y})")
        marker, = self.ax_img.plot([x], [y], 'o', color=color, markersize=6, markeredgecolor='white')
        self.points.append({'x': x, 'y': y, 'marker': marker, 'line': line, 'color': color})

        # Refresh Interactive Legend mapping
        leg = self.ax_spec.legend(title="Pixel (x,y)", loc='upper right', fontsize=8)
        self.leg_map = {}
        for legline, origline in zip(leg.get_lines(), self.ax_spec.lines):
            legline.set_picker(True)
            legline.set_pickradius(5)
            self.leg_map[legline] = origline

        self.fig.canvas.draw_idle()

    # ... (Keep _take_object, _clear_points, _onclick, _collect_spectra, _show_stat_calc, _update_stat_plots exactly as they were) ...

    def _take_object(self, event, target):
        current_data = [{'x': p['x'], 'y': p['y']} for p in self.points]
        if target == 1:
            self.object1 = current_data
        elif target == 2:
            self.object2 = current_data
        self._clear_points()

    def _clear_points(self):
        for p in self.points:
            p['marker'].remove();
            p['line'].remove()
        self.points, self.selected_point, self.color_idx = [], None, 0
        if self.ax_spec.get_legend(): self.ax_spec.get_legend().remove()
        self.tooltip.set_visible(False)
        self.fig.canvas.draw_idle()

    def _onclick(self, event):
        if event.inaxes != self.ax_img or event.xdata is None: return
        x, y = int(round(event.xdata)), int(round(event.ydata))

        action = None
        if event.button == 1 and self._get_gui_shift(event):
            action = ("add", x, y)
        elif event.button == 3:
            action = ("select" if not self.selected_point else "move", x, y)

        if not action: return
        now = time.time()
        if self.last_action == action and (now - self.last_action_time) < self.COOLDOWN: return
        self.last_action, self.last_action_time = action, now

        if action[0] == "add":
            self._plot_spectrum(x, y)
        elif action[0] == "select":
            p = next((p for p in self.points if abs(x - p['x']) <= 6 and abs(y - p['y']) <= 6), None)
            if p:
                self.selected_point = p
                self.tooltip.xy, self.tooltip.set_text = (p['x'], p['y']), f"Selected ({p['x']},{p['y']})"
                self.tooltip.set_visible(True)
                self.fig.canvas.draw_idle()
        elif action[0] == "move":
            p = self.selected_point
            p['x'], p['y'] = x, y
            p['marker'].set_data([x], [y])

            wl_plot, spec_plot = np.copy(self.wavelengths), np.squeeze(self.cube[y, x, :])
            if self.split_idx is not None and self.split_idx < len(self.wavelengths):
                if self.invalid_swir_mask[y, x]:
                    wl_plot, spec_plot = wl_plot[:self.split_idx], spec_plot[:self.split_idx]
                else:
                    wl_plot, spec_plot = np.insert(wl_plot, self.split_idx, np.nan), np.insert(spec_plot,
                                                                                               self.split_idx, np.nan)

            p['line'].set_data(wl_plot, spec_plot)
            p['line'].set_label(f"({x},{y})")

            leg = self.ax_spec.legend(title="Pixel (x,y)", loc='upper right', fontsize=8)
            self.leg_map = {}
            for legline, origline in zip(leg.get_lines(), self.ax_spec.lines):
                legline.set_picker(True)
                legline.set_pickradius(5)
                self.leg_map[legline] = origline

            self.tooltip.set_visible(False)
            self.selected_point = None
            self.fig.canvas.draw_idle()

    def _collect_spectra(self, obj):
        valid = [(y, x) for x, y in zip([p['x'] for p in obj], [p['y'] for p in obj])
                 if 0 <= x < self.cube.shape[1] and 0 <= y < self.cube.shape[0]]
        if not valid: return np.empty((0, len(self.wavelengths)))
        ys, xs = zip(*valid)
        arr = np.squeeze(self.cube[ys, xs, :])
        return arr[np.newaxis, :] if arr.ndim == 1 else arr.astype(float)

    def _show_stat_calc(self, event):
        if not self.object1 or not self.object2:
            print("🛑 Error: Save pixels to Object 1 & 2 first.")
            return

        if not self.stat_fig:
            self.stat_fig = plt.figure(figsize=(15, 8))
            self.stat_fig.canvas.manager.set_window_title("Statistical Comparison")
            gs = self.stat_fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 0.8, 0.8], height_ratios=[1, 1, 0])
            self.stat_axes = {
                'ax1': self.stat_fig.add_subplot(gs[0:2, 0]), 'ax2': self.stat_fig.add_subplot(gs[0:2, 1]),
                'ax3': self.stat_fig.add_subplot(gs[0:2, 2]), 'ax_met': self.stat_fig.add_subplot(gs[0, 3:]),
                'ax_norm': self.stat_fig.add_subplot(gs[1, 3:]), 'norm_mode': "Min–Max"
            }
            self.stat_fig.subplots_adjust(left=0.07, right=0.96, wspace=0.35, hspace=0.6)

            self.stat_axes['ax_norm'].set_title("Normalization", fontsize=11)
            self.radio = RadioButtons(self.stat_axes['ax_norm'], ["None", "Min–Max", "L2", "Z-Score"], active=1)
            self.radio.on_clicked(self._update_stat_plots)

        self._update_stat_plots(self.stat_axes['norm_mode'])
        self.stat_fig.show()

    def _update_stat_plots(self, norm_mode):
        self.stat_axes['norm_mode'] = norm_mode
        A = self._collect_spectra(self.object1)
        B = self._collect_spectra(self.object2)
        ax1, ax2, ax3, ax_met = self.stat_axes['ax1'], self.stat_axes['ax2'], self.stat_axes['ax3'], self.stat_axes[
            'ax_met']

        if A.size == 0 or B.size == 0: return

        A_norm = SpectralMetrics.normalize_array(A, norm_mode)
        B_norm = SpectralMetrics.normalize_array(B, norm_mode)
        mA, sA = np.mean(A_norm, axis=0), np.std(A_norm, axis=0)
        mB, sB = np.mean(B_norm, axis=0), np.std(B_norm, axis=0)

        for ax, data, m, s, color, title in zip([ax1, ax2], [A_norm, B_norm], [mA, mB], [sA, sB], ['orange', 'blue'],
                                                ["Object 1", "Object 2"]):
            ax.cla()
            # Insert NaNs into stats view if Fused Cube so it breaks here too!
            wl_p = np.insert(self.wavelengths, self.split_idx, np.nan) if self.split_idx and self.split_idx < len(
                self.wavelengths) else self.wavelengths
            for spec in data:
                sp_p = np.insert(spec, self.split_idx, np.nan) if self.split_idx and self.split_idx < len(
                    self.wavelengths) else spec
                ax.plot(wl_p, sp_p, color=color, alpha=0.4, lw=0.8)

            mp = np.insert(m, self.split_idx, np.nan) if self.split_idx and self.split_idx < len(
                self.wavelengths) else m
            sp = np.insert(s, self.split_idx, np.nan) if self.split_idx and self.split_idx < len(
                self.wavelengths) else s

            ax.plot(wl_p, mp, color=f'dark{color}' if color == 'orange' else 'navy', lw=2.2)
            ax.fill_between(wl_p, mp - sp, mp + sp, color=color, alpha=0.2)
            ax.set_title(f"{title} ({norm_mode})", fontsize=10);
            ax.grid(True)

        ax3.cla()
        ax3.plot(wl_p, np.insert(mA, self.split_idx, np.nan) if self.split_idx and self.split_idx < len(
            self.wavelengths) else mA, color='darkorange', label="Obj 1 Mean")
        ax3.plot(wl_p, np.insert(mB, self.split_idx, np.nan) if self.split_idx and self.split_idx < len(
            self.wavelengths) else mB, color='navy', label="Obj 2 Mean")
        ax3.set_title("Comparison", fontsize=10);
        ax3.legend(fontsize=8);
        ax3.grid(True)

        metrics = SpectralMetrics.compute_metrics(mA, mB)
        ax_met.clear();
        ax_met.axis("off")
        txt = metrics.get("error",
                          f"SAM: {metrics['SAM']:.2f}°\nSID: {metrics['SID']:.4f}\nCorr: {metrics['Corr']:.3f}\nEuc: {metrics['Euclid']:.3f}")
        ax_met.text(0.05, 0.95, txt, va="top", ha="left", fontsize=10,
                    bbox=dict(boxstyle="round", fc="khaki", alpha=0.7))
        self.stat_fig.canvas.draw_idle()