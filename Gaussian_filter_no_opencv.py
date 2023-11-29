
import numpy as np
import math
class GaussianFilter():
    def gauss_kernel(sigma):
        radius = round(3 * sigma)
        k = radius * 2 + 1
        kernel = np.zeros((k, k))
        y = radius
        for i in range(k):
            x = -radius
            for j in range(k):
                kernel[i, j] = (1 / (2 * math.pi * (sigma * sigma))) * math.exp(
                    ((-(x * x) - (y * y))) / (2 * (sigma * sigma)))
                x += 1
            y -= 1
        return kernel / np.sum(kernel), k, radius

    def gauss_blur(img, radius):
        blur = np.sum([np.multiply(img[(i - radius):(i + radius + 1), (j - radius):(j + radius + 1)], kernel) for i in
                       range(radius, img.shape[0] - radius) for j in range(radius, img.shape[1] - radius)])
        return blur
