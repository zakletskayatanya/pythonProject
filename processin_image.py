import numpy as np
import cv2
import gaussian_filter_no_opencv as gauss
from scipy.signal import convolve2d


def blur_image(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = gauss.GaussianFilter(7).gauss_blur(result)
    return result


def blur_diff_image(image1, image2):
    gray_img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray_diff = cv2.absdiff(gray_img1, gray_img2)
    blur_diff = gauss.GaussianFilter(7).gauss_blur(gray_diff)
    return blur_diff


sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])


def gradient_image(image):
    gradient_x = convolve2d(image, sobel_x, mode='same')
    gradient_y = convolve2d(image, sobel_y, mode='same')
    return gradient_x, gradient_y
