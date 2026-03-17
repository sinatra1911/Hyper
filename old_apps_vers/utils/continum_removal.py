import spectral as spy
import numpy as np
import matplotlib.pyplot as plt

image_path = r"C:\Users\Public\HyperData\BEIT_JAMAL\40m_try\vnir\raw_76000_rd_rf.hdr"
px=30
py=200

img = spy.open_image(image_path).load()
wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
pixel_ref = []
norm = []
continuum = []

pixel_ref.append(np.squeeze(img[py,px,:]))
pixel_ref.append(np.squeeze(img[200,300,:]))

for i in range(0,len(pixel_ref)):
    coeffs = np.polyfit(wavelengths, pixel_ref[i], 50)
    poly = np.poly1d(coeffs)
    continuum.append(poly(wavelengths))
    norm.append(np.squeeze(pixel_ref[i] / continuum[i]))
    plt.plot(wavelengths,pixel_ref[i])
    plt.plot(wavelengths,continuum[i])
    plt.plot(wavelengths,norm[i])

plt.show()
