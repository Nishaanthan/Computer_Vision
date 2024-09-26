import cv2
import sys
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.spatial as spatial
from typing import Tuple, Dict, List

# These are type hints, they mostly make the code readable and testable
t_img = np.array
t_disparity = np.array
t_points = np.array
t_descriptors = np.array


def extract_features(img: t_img, num_features: int = 500) -> Tuple[t_points, t_descriptors]:
    """Extract keypoints and their descriptors.
    The OpenCV implementation of ORB is used as a backend.
    https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF
    fast robust local feature detector, It is based on the FAST keypoint detector and a modified version of the visual descriptor BRIEF (Binary Robust Independent Elementary Features).
    Its aim is to provide a fast and efficient alternative to SIFT.

    Args:
        img: a numpy array of [H x Wx 3] size with byte values.
        num_features: an integer signifying how many points we desire.

    Returns:
        A tuple containing a numpy array of [N x 2] and numpy array of [N x 32]
    """
    
    #TODO : Hint - you will need cv2.ORB_create. 
    #TODO You have already implmented this function in Exercise 2: Panorama Stitching.
    ... 
    raise NotImplemented


def filter_and_align_descriptors(f1: Tuple[t_points, t_descriptors], f2: Tuple[t_points, t_descriptors],
                                 similarity_threshold=.7, similarity_metric='hamming') -> Tuple[t_points, t_points]:
    """Aligns pairs of keypoints from two images.
    Aligns keypoints from two images based on descriptor similarity.
    If K points have been detected in image1 and J points have been detected in image2, the result will be to sets of N
    points representing points with similar descriptors; where N <= J and K <=points.

    Args:
        f1: A tuple of two numpy arrays with the first array having dimensions [N x 2] and the second one [N x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        f2: A tuple of two numpy arrays with the first array having dimensions [J x 2] and the second one [J x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        similarity_threshold: The ratio the distance of most similar descriptor in image2 to the distance of the second
            most similar ratio.
        similarity_metric: A string with the name of the metric by which distances are calculated. It must be compatible
            with the ones that are defined for scipy.spatial.distance.cdist.

    Returns:
        A tuple of numpy arrays both sized [ N x 2]representing the similar point locations.

    """
    #TODO You have already implmented this function in Exercise 2: Panorama Stitching.

    raise NotImplemented

def get_max_translation(src: t_img, dst: t_img, well_aligned_thr=.1) -> int:
    """finds the maximum translation/shift between two images

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        well_aligned_thr: a float representing the maximum y wise distance between valid matching points.

    Returns:
        an integer value representing the maximum translation of the camera from src to dst image
    """

    # Step 1: Generate features/descriptors, filter and align them (courtesy of exercise 2)

    # Step 2: filter out correspondences that are not horizontally aligned using well aligned threshold

    # Step 3: Find the translation across the image using the descriptors and return the maximum value

    raise NotImplemented


def render_disparity_hypothesis(src: t_img, dst: t_img, offset: int, pad_size: int) -> t_disparity:
    """Calculates the agreement between the shifted src image and the dst image.

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        offset: an integer value by which the image is shifted
        pad_size: an integer value to pad the images for computation

    Returns:
        a numpy array of shape [H x W] containing the euclidean distance between RGB values of the shifted src and dst
        images.
    """

    # Step 1: Pad necessary values to src and dst

    # Step 2: find the disparity value and return

    raise NotImplemented

def disparity_map(src: t_img, dst: t_img, offset: int, pad_size: int, sigma_x: int, sigma_z: int,
                  median_filter_size: int) -> t_disparity:
    """calculates the best/minimum disparity map for a given pair of images

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        offset: an integer value by which the image is shifted
        pad_size: an integer value to pad the images for computation
        sigma_x: an integer value for standard deviation in x-direction for gaussian filter
        sigma_z: an integer value for standard deviation in z-direction for gaussian filter
        median_filter_size: an integer value representing the window size for applying median filter

    Returns:
        a numpy array of shape [H x W] containing the minimum/best disparity values for a pair of images
    """

    # Step 1: Construct a stack of all reasonable disparity hypotheses.

    # Step 2: Enforce the coherence between x-axis and disparity-axis using a 3D gaussian filter onto
    # the stack of disparity hypotheses

    # Step 3: Choose the best disparity hypothesis for every pixel

    # Step 4: Apply the median filter to enhance local consensus

    raise NotImplemented


def bilinear_grid_sample(img: t_img, x_array: t_img, y_array: t_img) -> t_img:
    """Sample an image according to a sampling vector field.

    Args:
        img: one image, numpy array of shape [H x W x 3]
        x_array: a numpy array of [H' x W'] representing the x coordinates src x-direction
        y_array: a numpy array of [H' x W'] representing interpolation in y-direction

    Returns:
        An image of size [H' x W'] containing the sampled points in
    """

    # Step 1: Estimate the left, top, right, bottom integer parts (l, r, t, b)
    # and the corresponding coefficients (a, b, 1-a, 1-b) of each pixel

    # Step 2: Take care of out of image coordinates

    # Step 3: Produce a weighted sum of each rounded corner of the pixel

    # Step 4: Accumulate and return all the weighted four corners

    raise NotImplemented

#
