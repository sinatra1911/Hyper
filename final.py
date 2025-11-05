import time
import spectral as spy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

# --- GLOBAL STATE AND CONFIGURATION ---

# === Load hyperspectral image ===
# NOTE: Replace with a valid path to an accessible .hdr file
# image_path = r"C:\Users\Public\HyperData\BEIT_JAMAL\40m_try\vnir\raw_76000_rd_rf.hdr"
image_path = r"D:\Hyper_Data\Aderet_2_4_24\50m\vnir\spectralview\100119_Aderet_2_4_24_50M_2024_04_02_08_36_30\22303\raw_22303_rd_rf.hdr"
try:
    # Use 'spectral' to open the image. If this fails, we use dummy data.
    img = spy.open_image(image_path).load()
    wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
except Exception as e:
    print(f"⚠️ Warning: Could not load actual image at path: {image_path}. Using dummy data instead.")
    # Fallback to dummy data
    H, W, B = 100, 100, 10
    img = np.random.rand(H, W, B)
    wavelengths = np.linspace(400, 700, B)


def find_band_indices(wavelengths, targets):
    """Return band indices closest to target wavelengths."""
    return [int(np.abs(wavelengths - t).argmin()) for t in targets]


rgb_targets = [650, 550, 450]  # red, green, blue (in nm)
rgb_bands = find_band_indices(wavelengths, rgb_targets)

# Normalize and gamma-correct for display
rgb = np.dstack([img[:, :, b] for b in rgb_bands]).astype(float)
low, high = np.percentile(rgb, (1, 99))
rgb = np.clip((rgb - low) / (high - low), 0, 1)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
rgb = np.power(rgb, 0.8)

# === Interactive State ===
colors = plt.cm.tab10(np.linspace(0, 1, 10))
points = []  # list of dicts: {'x','y','marker','line','color'} for current selection
color_index = 0
selected_point = None
_last_action = None
_last_action_time = 0.0
COOLDOWN = 0.4
object1 = []  # Stores 'points' when saved
object2 = []  # Stores 'points' when saved

# === Statistical Plot Widgets (Will store persistent references) ===
# This dictionary holds all necessary state and widget objects for the statistical plot
stat_fig = None
stat_axes = {}


# --- HELPER FUNCTIONS ---

def get_gui_shift(event):
    """Return True if shift is pressed (works across backends)."""
    if getattr(event, "key", None):
        return 'shift' in event.key
    gui = getattr(event, "guiEvent", None)
    if gui is not None:
        try:
            return 'Shift' in repr(gui.modifiers()) or (hasattr(gui, "state") and (gui.state & 0x0001) != 0)
        except Exception:
            pass
    return False


def get_point_near(x, y, tol=6):
    """Return a point dict if (x,y) is within tol pixels of a marker."""
    for p in points:
        px = float(np.ravel(p['marker'].get_xdata())[0])
        py = float(np.ravel(p['marker'].get_ydata())[0])
        if abs(x - px) <= tol and abs(y - py) <= tol:
            return p
    return None


def plot_spectrum(x, y, color=None):
    """Adds a new pixel to the current selection (points) and plots its spectrum."""
    global color_index
    if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
        print("⚠️ Pixel out of range.")
        return
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
    """Updates an existing point's coordinates and spectrum."""
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
    """Saves the current 'points' selection to object1/object2 and clears the current selection."""
    global object1, object2

    current_object_data = [{'x': p['x'], 'y': p['y']} for p in points]

    if target == 1:
        object1 = current_object_data
        print(f"\nSaved Object 1 with {len(object1)} pixels.")
    elif target == 2:
        object2 = current_object_data
        print(f"\nSaved Object 2 with {len(object2)} pixels.")

    clear(event)  # Clear the current selection markers and lines


def clear(event):
    """Clears all markers and lines from the main figure."""
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
    print("✅ Cleared all points on main figure.")


def onclick(event):
    """Main click handler for the image plot."""
    global selected_point, _last_action, _last_action_time

    if event.inaxes != ax_img or event.xdata is None or event.ydata is None:
        return

    x = int(round(event.xdata))
    y = int(round(event.ydata))

    is_shift = get_gui_shift(event)
    is_left = (event.button == 1)
    is_right = (event.button == 3)

    action = None
    if is_left and is_shift:
        action = ("add", x, y)
    elif is_right:
        action = ("select" if selected_point is None else "move", x, y)

    if action is None:
        return

    # Debounce / duplicate protection
    now = time.time()
    if _last_action == action and (now - _last_action_time) < COOLDOWN:
        return
    _last_action = action
    _last_action_time = now

    # Perform action
    if action[0] == "add":
        plot_spectrum(x, y)
        return

    if action[0] == "select":
        p = get_point_near(x, y)
        if p:
            selected_point = p
            tooltip.set_visible(True)
            tooltip.xy = (p['x'], p['y'])
            tooltip.set_text(f"Selected ({p['x']},{p['y']}) → right-click to place")
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


# --- STATISTICAL ANALYSIS FUNCTIONS ---

def normalize_array(data, mode):
    """Performs spectrum normalization based on the mode."""
    if mode == "None":
        return data
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


def compute_metrics(mean1, mean2):
    """Calculates spectral similarity metrics."""
    valid = np.isfinite(mean1) & np.isfinite(mean2)
    a, b = mean1[valid], mean2[valid]
    if a.size == 0:
        return {"error": "No valid bands."}
    eps = 1e-12

    # Spectral Angle Mapper (SAM)
    cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)
    cosang = np.clip(cosang, -1.0, 1.0)
    sam = np.degrees(np.arccos(cosang))

    # Spectral Information Divergence (SID)
    # The SID formula uses relative probabilities (p_i / sum(p)), so normalizing to unit sum (L1 norm) is better
    a_l1 = a / (np.sum(a) + eps)
    b_l1 = b / (np.sum(b) + eps)
    sid = np.sum(a_l1 * np.log((a_l1 + eps) / (b_l1 + eps))) + np.sum(b_l1 * np.log((b_l1 + eps) / (a_l1 + eps)))

    # Correlation
    if np.std(a) < eps or np.std(b) < eps:
        corr = np.nan
    else:
        corr = np.corrcoef(a, b)[0, 1]

    euc = np.linalg.norm(a - b)

    return dict(SAM=sam, SID=sid, Corr=corr, Euclid=euc)


def collect_spectra_for_stat(obj):
    """Collects spectra from coordinates in a simple object list."""
    xs = [p['x'] for p in obj]
    ys = [p['y'] for p in obj]
    if len(xs) == 0:
        return np.empty((0, len(wavelengths)))

    H, W, B = img.shape
    valid_indices = [(y, x) for x, y in zip(xs, ys) if 0 <= x < W and 0 <= y < H]

    if not valid_indices:
        return np.empty((0, len(wavelengths)))

    ys_valid, xs_valid = zip(*valid_indices)
    arr = np.squeeze(img[ys_valid, xs_valid, :])

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    return arr.astype(float)


def setup_stat_figure(labels):
    """Creates the statistical comparison figure and initializes the radio buttons."""
    global stat_fig, stat_axes

    if stat_fig:
        plt.figure(stat_fig.number)
        return

    stat_fig = plt.figure(figsize=(15, 8))
    gs = stat_fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 0.8, 0.8],
                               height_ratios=[1, 1, 0])

    stat_axes['ax1'] = stat_fig.add_subplot(gs[0:2, 0])
    stat_axes['ax2'] = stat_fig.add_subplot(gs[0:2, 1])
    stat_axes['ax3'] = stat_fig.add_subplot(gs[0:2, 2])
    stat_axes['ax_metrics'] = stat_fig.add_subplot(gs[0, 3:])
    stat_axes['ax_norm'] = stat_fig.add_subplot(gs[1, 3:])

    stat_fig.subplots_adjust(left=0.07, right=0.96, wspace=0.35, hspace=0.6)

    for ax_name in ['ax1', 'ax2', 'ax3']:
        ax = stat_axes[ax_name]
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.grid(True)

    stat_axes['labels'] = labels
    stat_axes['current_norm'] = ["Min–Max"]
    stat_axes['objects'] = None

    # === Radio Button Callbacks (Defined to use the global stat_axes state) ===

    def change_norm(label):
        # *** CRITICAL FIX: Update the global state variable
        stat_axes['current_norm'][0] = label
        update_stat_plots()

    # ---------- Normalization radio ----------
    stat_axes['ax_norm'].set_title("Normalization", fontsize=11)
    stat_axes['norm_radio'] = RadioButtons(stat_axes['ax_norm'],
                                           ["None", "Min–Max", "L2", "Z-Score"], active=1)
    stat_axes['norm_radio'].on_clicked(change_norm)  # Widget reference is stored in stat_axes


def update_stat_plots():
    """Updates the content of the statistical figure based on current state."""

    if not stat_fig or stat_axes['objects'] is None:
        return

    objects = stat_axes['objects']
    labels = stat_axes['labels']
    current_norm = stat_axes['current_norm'][0]

    objA, objB = objects[0], objects[1]
    A = collect_spectra_for_stat(objA)
    B = collect_spectra_for_stat(objB)

    ax1, ax2, ax3 = stat_axes['ax1'], stat_axes['ax2'], stat_axes['ax3']
    ax_metrics = stat_axes['ax_metrics']

    if A.size == 0 or B.size == 0:
        for ax in (ax1, ax2, ax3): ax.cla(); ax.set_title("No Data Selected")
        ax_metrics.clear();
        ax_metrics.axis("off")
        ax_metrics.text(0.5, 0.5, "Error: No valid spectra.", ha="center")
        stat_fig.canvas.draw_idle()
        return

    # Normalization logic
    A_norm = normalize_array(A, current_norm)
    B_norm = normalize_array(B, current_norm)

    meanA, stdA = np.mean(A_norm, axis=0), np.std(A_norm, axis=0)
    meanB, stdB = np.mean(B_norm, axis=0), np.std(B_norm, axis=0)

    # Re-plot Object A (ax1)
    ax1.cla()
    for s in A_norm:
        ax1.plot(wavelengths, s, color='orange', alpha=0.4, linewidth=0.8)
    ax1.plot(wavelengths, meanA, color='darkorange', linewidth=2.2)
    ax1.fill_between(wavelengths, meanA - stdA, meanA + stdA, color='orange', alpha=0.2)
    ax1.set_title(f"{labels[0]} ({current_norm})")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectance")
    ax1.grid(True)

    # Re-plot Object B (ax2)
    ax2.cla()
    for s in B_norm:
        ax2.plot(wavelengths, s, color='skyblue', alpha=0.4, linewidth=0.8)
    ax2.plot(wavelengths, meanB, color='navy', linewidth=2.2)
    ax2.fill_between(wavelengths, meanB - stdB, meanB + stdB, color='blue', alpha=0.2)
    ax2.set_title(f"{labels[1]} ({current_norm})")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Reflectance")
    ax2.grid(True)

    # Re-plot Comparison (ax3)
    ax3.cla()
    ax3.plot(wavelengths, meanA, color='darkorange', linewidth=2.0, label=f"{labels[0]} mean")
    ax3.plot(wavelengths, meanB, color='navy', linewidth=2.0, label=f"{labels[1]} mean")
    ax3.fill_between(wavelengths, meanA - stdA, meanA + stdA, color='orange', alpha=0.1)
    ax3.fill_between(wavelengths, meanB - stdB, meanB + stdB, color='blue', alpha=0.1)
    ax3.set_title("Mean Spectra Comparison")
    ax3.legend()
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Reflectance")
    ax3.grid(True)

    # Update Metrics (ax_metrics)
    metrics = compute_metrics(meanA, meanB)
    ax_metrics.clear()
    ax_metrics.axis("off")
    if "error" in metrics:
        txt = metrics["error"]
    else:
        txt = (f"Compare: {labels[0]} vs {labels[1]}\n\n"
               f"SAM: {metrics['SAM']:.3f}°\n"
               f"SID: {metrics['SID']:.4f}\n"
               f"Correlation: {metrics['Corr']:.3f}\n"
               f"Euclidean: {metrics['Euclid']:.3f}\n")

    ax_metrics.text(0.05, 0.95, txt, va="top", ha="left", fontsize=10,
                    bbox=dict(boxstyle="round", fc="khaki", alpha=0.7))

    stat_fig.canvas.draw_idle()


def show_stat_calc(event):
    """Event handler for the 'Statistic calculation' button."""
    global stat_fig, stat_axes

    if len(object1) == 0 or len(object2) == 0:
        print("\n🛑 Error: Must save at least one pixel to both Object 1 and Object 2 before calculating statistics.")
        return

    objects = [object1, object2]
    labels = ["Object 1", "Object 2"]

    # 1. Setup figure if it doesn't exist (CRITICAL for widget initialization)
    setup_stat_figure(labels)

    # 2. Update global state for statistical plots
    stat_axes['objects'] = objects
    stat_axes['labels'] = labels

    # 3. Update the plots based on the new data
    update_stat_plots()

    # 4. Bring the figure to the front
    plt.figure(stat_fig.number)
    stat_fig.show()


# --- MAIN FIGURE AND WIDGETS ---

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

# === Tooltip ===
tooltip = ax_img.annotate(
    "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
    arrowprops=dict(arrowstyle="->")
)
tooltip.set_visible(False)

# === Widgets (MUST be defined and stored as variables before plt.show()) ===
axbox1 = plt.axes([0.20, 0.03, 0.15, 0.05])
object_button1 = Button(axbox1, "Save Object 1 & Clear")
object_button1.on_clicked(lambda event: take_object(event, 1))

axbox2 = plt.axes([0.40, 0.03, 0.15, 0.05])
statistic_button = Button(axbox2, "Statistic Calculation")
statistic_button.on_clicked(show_stat_calc)

axbox3 = plt.axes([0.60, 0.03, 0.15, 0.05])
object_button2 = Button(axbox3, "Save Object 2 & Clear")
object_button2.on_clicked(lambda event: take_object(event, 2))

# === Connect event handlers ===
fig.canvas.mpl_connect('button_press_event', onclick)

# === Final Display ===
plt.show()
