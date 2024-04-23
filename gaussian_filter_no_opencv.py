import numpy as np
import spicy
# from numba import jit

class GaussianFilter:
    sigma_ = 3

    def __init__(self, kernel_size):
        # assert self.kernelSize_ % 2 != 1, "Kernel size must be uneven number"

        self.kernelSize_ = kernel_size
        radius = np.arange(-kernel_size // 2, kernel_size // 2, 1)

        kernel_not_norm = (1 / np.sqrt(2 * np.pi * (self.sigma_ ** 2))) * np.exp((-(radius ** 2)) / (2 * (self.sigma_ ** 2)))

        kernel_norm = kernel_not_norm / np.sum(kernel_not_norm)
        self.kernel = kernel_norm

    # @jit(nopython=True)
    # def gauss_gorizontal(self, image):
    #     radius = self.kernelSize_ // 2
    #     prom = np.zeros(image.shape, dtype=float)
    #     for i in range(image.shape[0]):
    #         for j in range(radius, image.shape[1] - radius):
    #             prom[i, j] = np.sum(np.multiply(image[i, (j-radius):(j+radius+1)], self.kernel))
    #             # s = 0
    #             # for k in range(-radius, radius+1):
    #             #     s += image[i, j+k] * self.kernel[k+radius]
    #             # prom[i, j] = s
    #
    #     blur = prom
    #     return blur
    #
    # @jit(nopython=True)
    # def gauss_vertical(self, image):
    #     radius = self.kernelSize_ // 2
    #     prom = np.zeros(image.shape, dtype=float)
    #     for j in range(image.shape[1]):
    #         for i in range(radius, image.shape[0] - radius):
    #             prom[i, j] = np.sum(np.multiply(image[(i - radius):(i + radius + 1), j], self.kernel))
    #
    #     blur = prom
    #     return blur

    def gauss_blur(self, img):
        # image = self.gauss_vertical(img)
        # image = self.gauss_gorizontal(image)
        image = spicy.ndimage.convolve(img, self.kernel[np.newaxis, :], mode='reflect')
        return image
