import cv2
import numpy as np
import scipy.signal as sig


kx = np.array([[-1, 0, 1]])
ky = np.array([[-1], [0], [1]])


def read_left(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    dx = sig.convolve2d(img, kx, 'same')
    dy = sig.convolve2d(img, ky, 'same')
    img = np.expand_dims(img, -1)
    dx = np.expand_dims(dx, -1)
    dy = np.expand_dims(dy, -1)
    img = img.astype('float32') / 127.5 - 1.0
    dx = dx.astype('float32') / 127.5
    dy = dy.astype('float32') / 127.5
    return img, dx, dy


def read_right(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = np.expand_dims(img, -1)
    img = img.astype('float32') / 127.5 - 1.0
    return img
