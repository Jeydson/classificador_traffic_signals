import numpy as np
import cv2
import skimage.morphology as morp
from skimage.filters import rank


def gray(image):
    """Converts an RGB image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def local_histo_equalize(image):
    """Applies local histogram equalization."""
    kernel = morp.disk(30)
    return rank.equalize(image, selem=kernel)


def image_normalize(image):
    """Normalizes image pixels to range [0, 1]."""
    return image.astype(np.float32) / 255.0


def preprocess(data):
    """Runs preprocessing pipeline on a batch of images."""
    gray_images = list(map(gray, data))
    equalized = list(map(local_histo_equalize, gray_images))
    normalized = np.array([image_normalize(img) for img in equalized])
    return normalized[..., None]
