import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    assert I.dtype == np.float32

    # Step 1: Compute Idx and Idy with cv2.Sobel
    idx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    idy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)

    # Step 2: Ixx Iyy Ixy from Idx and Idy
    ixx = idx * idx
    iyy = idy * idy
    ixy = idx * idy

    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur
    # Use sdev = 1 and kernelSize = (3, 3) in cv2.GaussianBlur
    a = cv2.GaussianBlur(ixx, (3, 3), 1)
    b = cv2.GaussianBlur(iyy, (3, 3), 1)
    c = cv2.GaussianBlur(ixy, (3, 3), 1)

    # Step 4: Compute the harris response with the determinant and the trace of T
    det = a * b - c * c
    trace = a + b
    r = det - k * trace * trace

    return r, a, b, c, idx, idy


def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """

    # Step 1 (recommended): Pad the response image to facilitate vectorization
    padded_R = np.pad(R, 1, mode="edge")

    # Step 2 (recommended): Create one image for every offset in the 3x3 neighborhood
    shifted_images = [
        padded_R[1:-1, 1:-1],  # center
        padded_R[:-2, :-2],    # top-left
        padded_R[:-2, 1:-1],   # top-center
        padded_R[:-2, 2:],     # top-right
        padded_R[1:-1, :-2],   # middle-left
        padded_R[1:-1, 2:],    # middle-right
        padded_R[2:, :-2],     # bottom-left
        padded_R[2:, 1:-1],    # bottom-center
        padded_R[2:, 2:]       # bottom-right
    ]

    # Step 3 (recommended): Compute the greatest neighbor of every pixel
    max_neighbors = np.max(shifted_images, axis=0)

    # Step 4 (recommended): Compute a boolean image with only all key-points set to True
    key_points = np.logical_and(R > threshold, R == max_neighbors)

    # Step 5 (recommended): Use np.nonzero to compute the locations of the key-points from the boolean image
    y, x = np.nonzero(key_points)

    return x, y


def detect_edges(R: np.array, edge_threshold: float = -0.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant

    Returns:
        A boolean image with edge pixels set to True.
    """

    # Step 1 (recommended): Pad the response image to facilitate vectorization
    padded_R = np.pad(R, 1, mode="edge")

    # Step 2 (recommended): Calculate significant response pixels
    significant = R < edge_threshold

    # Step 3 (recommended): Create two images with the smaller x-axis and y-axis neighbors respectively
    x_neighbors = np.minimum(padded_R[1:-1, :-2], padded_R[1:-1, 2:])

    # Step 4 (recommended): Calculate pixels that are lower than either their x-axis or y-axis neighbors
    axis_minimal = np.logical_or(R < x_neighbors, R < np.minimum(padded_R[:-2, 1:-1], padded_R[2:, 1:-1]))

    # Step 5 (recommended): Calculate valid edge pixels by combining significant and axis_minimal pixels
    edge_pixels = np.logical_and(significant, axis_minimal)

    return edge_pixels
