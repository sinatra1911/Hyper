
"""
1. take statiasc of 10 pixels of the same object - mean and std for every wavelength and compute the avarege reflectence graph
2. compere the avarege reflectence graphs of stone and a target and see how difrebt it is
3. using the right metric, cpmpute how close they are - i can calculate the diffrence in every wavelenth and take the sum of it
"""

import spectral as spy
import numpy as np
import matplotlib.pyplot as plt

image_path = r"C:\Users\Public\HyperData\BEIT_JAMAL\40m_try\vnir\raw_76000_rd_rf.hdr"
px=50
py=200
obj_x_arr = [50,51]
obj_y_arr = [200,201]

img = spy.open_image(image_path).load()
wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
pixels = []

# realize choosing few more pixels and cac
pixels.append(np.squeeze(img[py,px,:]))

pixels.append(np.squeeze(img[py+1,px+1,:]))

num_of_pixels=len(pixels)

std = np.std(pixels, axis=0)
mean = np.mean(pixels, axis=0)

# takes out pixels with high noise
for _ in range(num_of_pixels):
    pixels[_] = (pixels[_] - np.min(pixels[_])) / (np.max(pixels[_]) - np.min(pixels[_]))
    for i in range(len(pixels[0])):
        if std[i] > 0.1:
            mean[i] = None

plt.plot(wavelengths, mean,'k')
plt.plot(wavelengths,pixels[0])
plt.plot(wavelengths,pixels[1])
plt.show()