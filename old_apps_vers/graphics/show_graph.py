import time
import spectral as spy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

# === Load hyperspectral image ===
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
    line, = ax_spec.plot(wavelengths, spectrum / np.sum(spectrum), color=color, label=f"prob of ({x},{y})") # prob vector representation
    print(f"{np.sum(spectrum / np.sum(spectrum))} \n {np.mean(spectrum / np.sum(spectrum))}")
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
        text_box.set_val(f"{x},{y}")
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
def submit(text):
    try:
        xs, ys = text.split(',')
        x, y = int(xs.strip()), int(ys.strip())
        plot_spectrum(x, y)
    except Exception:
        print("❌ Invalid input. Use format: x,y")


def clear_all(event):
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

# === Widgets ===
axbox = plt.axes([0.20, 0.03, 0.15, 0.05])
text_box = TextBox(axbox, "Pixel (x,y): ", initial="")
text_box.on_submit(submit)

axclear = plt.axes([0.60, 0.03, 0.15, 0.05])
clear_button = Button(axclear, "Clear All")
clear_button.on_clicked(clear_all)

# === Connect only press events ===
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
