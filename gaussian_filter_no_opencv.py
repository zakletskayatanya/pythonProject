import numpy as np
import spicy
from numba import jit, prange


@jit(nopython=True)
def gauss_kernel(kernel_size):
    assert kernel_size % 2 == 1, "Kernel size must be uneven number"
    sigma_ = 2
    radius = np.arange(-kernel_size // 2, kernel_size // 2, 1)

    kernel_not_norm = (1 / np.sqrt(2 * np.pi * (sigma_ ** 2))) * np.exp(
        (-(radius ** 2)) / (2 * (sigma_ ** 2)))

    kernel_norm = kernel_not_norm / np.sum(kernel_not_norm)
    kernel = kernel_norm
    return kernel


@jit(nopython=True)
def gauss_gorizontal(image, kernel, kernel_size):
    radius = kernel_size // 2
    prom = np.zeros(image.shape, dtype=float)
    for i in prange(image.shape[0]):
        for j in range(radius, image.shape[1] - radius):
            prom[i, j] = np.sum(np.multiply(image[i, (j - radius):(j + radius + 1)], kernel))

    blur = prom
    return blur


@jit(nopython=True)
def gauss_vertical(image, kernel, kernel_size):
    radius = kernel_size // 2
    prom = np.zeros(image.shape, dtype=float)
    for j in prange(image.shape[1]):
        for i in range(radius, image.shape[0] - radius):
            prom[i, j] = np.sum(np.multiply(image[(i - radius):(i + radius + 1), j], kernel))

    blur = prom
    return blur


@jit(nopython=True)
def gauss_blur(img, kernel_size):
    kernel = gauss_kernel(kernel_size)
    image = gauss_vertical(img,  kernel, kernel_size)
    image = gauss_gorizontal(image,  kernel, kernel_size)
    # image = spicy.ndimage.convolve(img, self.kernel[np.newaxis, :], mode='reflect')
    return image
