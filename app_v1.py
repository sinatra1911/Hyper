import time
import spectral as spy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

# === Load hyperspectral image ===
# image_path = r"D:\Hyper_Data\Aderet_2_4_24\50m\swir\spectralview\14230\raw_14230_rd_rf.hdr"
image_path = r"C:\Users\Public\HyperData\BEIT_JAMAL\40m_try\vnir\raw_76000_rd_rf.hdr"
img = spy.open_image(image_path).load()

# === Wavelengths and RGB composite ===
wavelengths = np.array([float(w) for w in img.metadata['wavelength']])

def find_band_indices(wavelengths, targets):
    """Return band indices closest to target wavelengths."""
    return [int(np.abs(wavelengths - t).argmin()) for t in targets]

rgb_targets = [650, 550, 450]  # red, green, blue (in nm)
rgb_bands = find_band_indices(wavelengths, rgb_targets)

# Normalize and gamma-correct for display
rgb = np.dstack([img[:, :, b] for b in rgb_bands]).astype(float)

# Clip to percentile range to avoid overexposure
low, high = np.percentile(rgb, (1, 99))
rgb = np.clip((rgb - low) / (high - low), 0, 1)

# Optional: apply mild gamma correction
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
rgb = np.power(rgb, 0.8)

# === Figure setup ===
fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(13, 7))
plt.subplots_adjust(bottom=0.15)
ax_img.imshow(rgb)
ax_img.set_title("Shift+Left Click = Add | Right Click once = Select | Right Click again = Move")
ax_img.axis("off")

ax_spec.set_title("Reflectance Spectra")
ax_spec.set_xlabel("Wavelength (nm)")
ax_spec.set_ylabel("Reflectance")
ax_spec.grid(True)

# === State ===
colors = plt.cm.tab10(np.linspace(0, 1, 10))
points = []                      # list of dicts: {'x','y','marker','line','color'}
color_index = 0
selected_point = None            # when a point is selected for relocation
_last_action = None              # store last processed action id
_last_action_time = 0.0
COOLDOWN = 0.4                   # seconds to ignore repeated identical actions

# === Tooltip ===
tooltip = ax_img.annotate(
    "", xy=(0,0), xytext=(10,10), textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
    arrowprops=dict(arrowstyle="->")
)
tooltip.set_visible(False)


# === Helpers ===
def get_gui_shift(event):
    """Return True if shift is pressed (works across backends)."""
    # Prefer event.key (matplotlib provided)
    if getattr(event, "key", None):
        try:
            return 'shift' in event.key
        except Exception:
            pass
    # Fallback to guiEvent (tk/qt raw event) if available
    gui = getattr(event, "guiEvent", None)
    if gui is not None:
        # Qt: gui.modifiers() returns Qt.ShiftModifier; Tk: gui.state; careful and tolerant
        try:
            # Qt: check attribute 'modifiers' or 'modifiers()'
            if hasattr(gui, "modifiers"):
                mods = gui.modifiers()
                # The numeric value is backend-dependent, but checking repr for 'Shift' is safe:
                return 'Shift' in repr(mods)
            if hasattr(gui, "state"):
                return (gui.state & 0x0001) != 0  # Tk common mask for Shift (best-effort)
        except Exception:
            pass
    return False


def get_point_near(x, y, tol=6):
    """Return a point dict if (x,y) is within tol pixels of a marker."""
    for p in points:
        px_arr, py_arr = p['marker'].get_data()
        # px_arr, py_arr are sequences of length 1 -> extract scalars safely
        px = float(np.ravel(px_arr)[0])
        py = float(np.ravel(py_arr)[0])
        if abs(x - px) <= tol and abs(y - py) <= tol:
            return p
    return None


def plot_spectrum(x, y, color=None):
    global color_index
    if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
        print("⚠️ Pixel out of range.")
        return
    # avoid duplicates
    for p in points:
        if p['x'] == x and p['y'] == y:
            print(f"⚠️ Pixel ({x},{y}) already exists. Skipping.")
            return
    if color is None:
        color = colors[color_index % len(colors)]
        color_index += 1
    spectrum = np.squeeze(img[y, x, :])
    line, = ax_spec.plot(wavelengths, spectrum, color=color, label=f"({x},{y})")
    marker, = ax_img.plot([x], [y], 'o', color=color, markersize=6, markeredgecolor='white')
    points.append({'x': x, 'y': y, 'marker': marker, 'line': line, 'color': color})
    ax_spec.legend(title="Pixel (x,y)", loc='upper right')
    fig.canvas.draw_idle()
    print(f"🟢 Added pixel ({x},{y})")


def update_point(p, new_x, new_y):
    if not (0 <= new_x < img.shape[1] and 0 <= new_y < img.shape[0]):
        print("⚠️ Out of image bounds.")
        return
    p['x'], p['y'] = new_x, new_y
    p['marker'].set_data([new_x], [new_y])
    spectrum = np.squeeze(img[new_y, new_x, :])
    p['line'].set_ydata(spectrum)
    p['line'].set_label(f"({new_x},{new_y})")
    ax_spec.legend(title="Pixel (x,y)", loc='upper right')
    fig.canvas.draw_idle()
    print(f"✅ Moved pixel to ({new_x},{new_y})")

def take_object(event, target):
    global object1, object2, points

    if target == "object1":
        object1 = list(points)
        print("Saved object1:", object1)

    elif target == "object2":
        object2 = list(points)
        print("Saved object2:", object2)

# === Main click handler (button_press_event only) ===
def onclick(event):
    global selected_point, _last_action, _last_action_time

    # only handle clicks on the image axes
    if event.inaxes != ax_img or event.xdata is None or event.ydata is None:
        return

    # normalized integer pixel coords
    x = int(round(event.xdata))
    y = int(round(event.ydata))

    # determine if this is a Shift+Left click (add) or Right-click (select/move)
    is_shift = get_gui_shift(event)
    is_left = (event.button == 1)
    is_right = (event.button == 3)

    # form a deterministic action id tuple
    action = None
    if is_left and is_shift:
        action = ("add", x, y)
    elif is_right:
        # decide whether this right click is "select" or "move" depending on selected_point
        if selected_point is None:
            action = ("select", x, y)
        else:
            action = ("move", x, y)

    # If no recognized action, ignore
    if action is None:
        return

    # Debounce / duplicate protection: ignore identical action within COOLDOWN seconds
    now = time.time()
    if _last_action == action and (now - _last_action_time) < COOLDOWN:
        # duplicate — ignore
        # print("Duplicate action ignored")  # debug if needed
        return
    _last_action = action
    _last_action_time = now

    # Perform action
    if action[0] == "add":
        # Add pixel (plot_spectrum has its own duplicate-check)
        plot_spectrum(x, y)
        return

    if action[0] == "select":
        # find a nearby existing point and select it
        p = get_point_near(x, y)
        if p:
            selected_point = p
            tooltip.set_visible(True)
            tooltip.xy = (p['x'], p['y'])
            tooltip.set_text(f"Selected ({p['x']},{p['y']})  → right-click to place")
            fig.canvas.draw_idle()
            print(f"🟡 Selected pixel ({p['x']},{p['y']}) to move")
        else:
            print("❌ No pixel near click to select.")
        return

    if action[0] == "move":
        if selected_point is None:
            print("❌ No pixel was selected for moving.")
            return
        update_point(selected_point, x, y)
        tooltip.set_visible(False)
        selected_point = None
        return


# === Manual input & clear UI ===
def clear(event):
    global points, color_index, selected_point
    for p in points:
        try:
            p['marker'].remove()
            p['line'].remove()
        except Exception:
            pass
    points = []
    selected_point = None
    color_index = 0
    legend = ax_spec.get_legend()
    if legend:
        legend.remove()
    tooltip.set_visible(False)
    fig.canvas.draw_idle()
    print("✅ Cleared all points.")

def statistical_calc(obj1, obj2):
    fig, axes = plt.subplots(1, 4, figsize=(16, 7))
    for ax in axes[:3]:
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")

    # === Extract spectra for object 1 ===
    obj1_x, obj1_y = [o['x'] for o in obj1], [o['y'] for o in obj1]
    spectra1 = np.squeeze(img[obj1_y, obj1_x, :])
    spectra1 = (spectra1 - np.min(spectra1, axis=1, keepdims=True)) / (
        np.max(spectra1, axis=1, keepdims=True) - np.min(spectra1, axis=1, keepdims=True)
    )
    for s in spectra1:
        axes[0].plot(wavelengths, s, color='skyblue', alpha=0.5, linewidth=1)
    mean1 = np.mean(spectra1, axis=0)
    std1 = np.std(spectra1, axis=0)
    axes[0].plot(wavelengths, mean1, color='navy', linewidth=2.5, label='Mean')
    axes[0].fill_between(wavelengths, mean1 - std1, mean1 + std1, color='blue', alpha=0.15)
    axes[0].set_title("Object 1 Spectra")
    axes[0].legend()

    # === Extract spectra for object 2 ===
    obj2_x, obj2_y = [o['x'] for o in obj2], [o['y'] for o in obj2]
    spectra2 = np.squeeze(img[obj2_y, obj2_x, :])
    spectra2 = (spectra2 - np.min(spectra2, axis=1, keepdims=True)) / (
        np.max(spectra2, axis=1, keepdims=True) - np.min(spectra2, axis=1, keepdims=True)
    )
    for s in spectra2:
        axes[1].plot(wavelengths, s, color='lightcoral', alpha=0.5, linewidth=1)
    mean2 = np.mean(spectra2, axis=0)
    std2 = np.std(spectra2, axis=0)
    axes[1].plot(wavelengths, mean2, color='darkred', linewidth=2.5, label='Mean')
    axes[1].fill_between(wavelengths, mean2 - std2, mean2 + std2, color='red', alpha=0.15)
    axes[1].set_title("Object 2 Spectra")
    axes[1].legend()

    # === Mean comparison ===
    axes[2].plot(wavelengths, mean1, 'orange', linewidth=2, label="Mean Obj1")
    axes[2].plot(wavelengths, mean2, 'limegreen', linewidth=2, label="Mean Obj2")
    axes[2].fill_between(wavelengths, mean1 - std1, mean1 + std1, color='orange', alpha=0.2)
    axes[2].fill_between(wavelengths, mean2 - std2, mean2 + std2, color='green', alpha=0.2)
    axes[2].set_title("Mean Spectra Comparison")
    axes[2].legend()

    # === Compute metrics ===
    valid = np.isfinite(mean1) & np.isfinite(mean2)
    a, b = mean1[valid], mean2[valid]
    eps = 1e-10

    sam = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    pa, pb = a / (np.sum(a) + eps), b / (np.sum(b) + eps)
    sid = np.sum(pa * np.log((pa + eps) / (pb + eps)) + pb * np.log((pb + eps) / (pa + eps)))
    scc = np.corrcoef(a, b)[0, 1]
    euclidean = np.linalg.norm(a - b)


    # === Combine metrics into similarity score ===
    # Normalize and invert where needed (lower is better → higher score)
    sam_score = np.exp(-sam * 5)       # smaller angle = higher score
    sid_score = np.exp(-sid * 2)
    scc_score = (scc + 1) / 2          # -1..1 → 0..1
    euc_score = np.exp(-euclidean * 3)

    # Weighted combination (tweak weights as you like)
    overall_similarity = (
        0.3 * sam_score +
        0.3 * sid_score +
        0.25 * scc_score +
        0.15 * euc_score
    ) * 100

    # === Display metrics panel ===
    metrics = {
        "SAM (rad)": sam,
        "SID": sid,
        "SCC": scc,
        "Euclidean": euclidean,
        "Manhattan": manhattan,
        "MAE": mae,
    }

    def sim_color(val, invert=False):
        val = np.clip(val, 0, 1)
        if invert:
            val = 1 - val
        return (1 - val, val, 0)  # red → green gradient

    axes[3].axis("off")
    axes[3].set_title("Similarity Metrics", fontweight="bold", pad=15)

    # === Display overall similarity score ===
    score_color = sim_color(overall_similarity / 100)
    axes[3].text(
        0.5, 0.92, f"Overall Similarity: {overall_similarity:.1f}%",
        transform=axes[3].transAxes,
        fontsize=14, fontweight="bold",
        ha='center', bbox=dict(boxstyle="round,pad=0.5", fc=score_color, alpha=0.8)
    )

    # === Display individual metrics ===
    y_positions = np.linspace(0.78, 0.1, len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        if name == "SCC":
            color = sim_color(val)
        elif name == "SAM (rad)":
            color = sim_color(np.exp(-val), invert=True)
        else:
            norm_val = np.exp(-val / np.max([euclidean, manhattan, mae, 1e-6]))
            color = sim_color(norm_val)
        axes[3].text(
            0.05, y_positions[i], f"{name}: {val:.4f}",
            transform=axes[3].transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.7)
        )

    plt.tight_layout()
    plt.show()

# === Widgets ===
object1 = []
object2 = []
axbox1 = plt.axes([0.20, 0.03, 0.15, 0.05])
object_button1 = Button(axbox1, "Save object 1")
object_button1.on_clicked(lambda event: take_object(event, "object1"))
object_button1.on_clicked(clear)

axbox2 = plt.axes([0.60, 0.03, 0.15, 0.05])
object_button2 = Button(axbox2, "Save object 2")
object_button2.on_clicked(lambda event: take_object(event, "object2"))
object_button2.on_clicked(clear)

ax_stat = plt.axes([0.40, 0.03, 0.15, 0.05])
statistic_button = Button(ax_stat, "Statistic calculation")
statistic_button.on_clicked(lambda event: statistical_calc(object1, object2))

# === Connect only press events ===
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
