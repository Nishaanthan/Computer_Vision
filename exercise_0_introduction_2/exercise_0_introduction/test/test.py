import pytest
import numpy as np
import sys
import cv2
sys.path.append(".")
from utils import save_images, scale_down, separate_channels
import re

epsilon = .0001


def all_similar(t1, t2):
    """Test the maximum square error is lesser than epsilon."""
    delta = (t1 - t2) ** 2
    correct = delta > epsilon
    return correct.reshape(-1).mean() == 0


def test_save_images(tmp_path):
    color_img = (np.random.rand(100, 110, 3) * 255).astype(np.uint8)
    gray_img = (np.random.rand(100, 100) * 255).astype(np.uint8)
    d = tmp_path / "sub"
    d.mkdir()
    color_path = d / "color.png"
    gray_path = d / "gray.png"
    save_images([color_img, gray_img], [str(color_path), str(gray_path)])
    loaded_color = cv2.imread(str(color_path))
    loaded_gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
    assert all_similar(color_img, loaded_color)
    assert all_similar(gray_img, loaded_gray)


def test_scale(tmp_path):
    large_img = np.zeros([100, 101, 3], dtype=np.uint8)
    large_img[:50, :50, :] = 255
    large_img[:50, 50, :] = 127
    small_img = np.zeros([50, 50, 3], dtype=np.uint8)
    small_img[:25, :25, :] = 255
    computed_small_img = scale_down(large_img)
    assert all_similar(computed_small_img, small_img)


def test_color(tmp_path):
    color_img = (np.random.rand(100, 110, 3) * 255).astype(np.uint8)
    blue, green, red = separate_channels(color_img)
    assert all_similar(color_img, blue + green + red)
