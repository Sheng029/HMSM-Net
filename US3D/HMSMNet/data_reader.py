import cv2
import numpy as np
import scipy.signal as sig


kx = np.array([[-1, 0, 1]])
ky = np.array([[-1], [0], [1]])


def read_left(filename):
    rgb = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    rdx, rdy = sig.convolve2d(rgb[:, :, 2], kx, 'same'), sig.convolve2d(rgb[:, :, 2], ky, 'same')
    gdx, gdy = sig.convolve2d(rgb[:, :, 1], kx, 'same'), sig.convolve2d(rgb[:, :, 1], ky, 'same')
    bdx, bdy = sig.convolve2d(rgb[:, :, 0], kx, 'same'), sig.convolve2d(rgb[:, :, 0], ky, 'same')
    dx = cv2.merge([bdx, gdx, rdx])
    dy = cv2.merge([bdy, gdy, rdy])
    rgb = rgb.astype('float32') / 127.5 - 1.0
    dx = dx.astype('float32') / 127.5
    dy = dy.astype('float32') / 127.5
    return rgb, dx, dy


def read_right(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype('float32') / 127.5 - 1.0
    return image
