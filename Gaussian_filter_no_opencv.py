import numpy as np
import math


class GaussianFilter:
    sigma_ = 2

    # kernelSize_ = None
    # kernel = []

    def __init__(self, image, kernel_size):
        # assert self.kernelSize_ % 2 != 1, "Kernel size must be uneven number"

        self.kernelSize_ = kernel_size
        self.image = image
        radius = kernel_size // 2
        kernel_not_norm = np.zeros((kernel_size, kernel_size), dtype=float)
        y = radius
        for i in range(kernel_size):
            x = -radius
            for j in range(kernel_size):
                kernel_not_norm[i, j] = (1 / (2 * math.pi * (self.sigma_ ** 2))) * math.exp(
                    (-(x ** 2) - (y ** 2)) / (2 * (self.sigma_ ** 2)))
                x += 1
            y -= 1
        kernel_norm = kernel_not_norm / np.sum(kernel_not_norm)
        self.kernel = kernel_norm

    def gauss_blur(self):
        radius = self.kernelSize_ // 2
        prom = np.zeros(self.image.shape, dtype=float)
        for i in range(radius, self.image.shape[0] - radius):
            for j in range(radius, self.image.shape[1] - radius):
                prom[i, j] = np.sum(
                    np.multiply(self.image[(i - radius):(i + radius + 1), (j - radius):(j + radius + 1)],
                                self.kernel))
        blur = prom
        return blur
