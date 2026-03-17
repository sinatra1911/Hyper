import spectral as spy
import numpy as np
import matplotlib.pyplot as plt

image_path = r"C:\Users\Public\HyperData\BEIT_JAMAL\40m_try\vnir\raw_76000_rd_rf.hdr"
threshold = 0.1 #threshold of the angle
px=30
py=200

img = spy.open_image(image_path).load()
wavelengths = np.array([float(w) for w in img.metadata['wavelength']])

pixel_test = np.squeeze(img[py,px,:])

def angle_computation(vec1, vec2):
    np.array(vec1)
    np.array(vec2)
    return np.acos((vec1 @ vec2.T) / ((vec1 @ vec1.T)**(0.5) * (vec2 @ vec2.T)**(0.5)))

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

for i in range(2000):
    for j in range(640):
        pixel_ref = np.squeeze(img[i,j,:])
        alpha = angle_computation(pixel_ref, pixel_test)
        if alpha < threshold:
            rgb[i,j] = [1,0,0]

fig, ax = plt.subplots()
ax.imshow(rgb)
ax.axis("off")
plt.show()



