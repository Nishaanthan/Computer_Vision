import sys
sys.path.append(".")
import pytest
import numpy as np

from ex4.util import computeBilinerWeights, computeGaussianWeights, invertMatrix2x2

epsilon = 1e-6


def test_bilinear_weights():

    p = np.array([0.125, 0.82656])
    weights = np.array([0.15176, 0.02168, 0.72324, 0.10332])

    err = computeBilinerWeights(p) - weights
    e = np.linalg.norm(err)

    assert e < epsilon


def test_gaussian_weights():
    kernel = np.array(
        [[0.1690133273455962, 0.3291930023422986, 0.4111123050281957, 0.3291930023422986, 0.1690133273455962],
        [0.3291930023422986, 0.6411803997536073, 0.8007374099875735, 0.6411803997536073, 0.3291930023422986],
        [0.4111123050281957, 0.8007374099875735, 1, 0.8007374099875735, 0.4111123050281957],
        [0.3291930023422986, 0.6411803997536073, 0.8007374099875735, 0.6411803997536073, 0.3291930023422986],
        [0.1690133273455962, 0.3291930023422986, 0.4111123050281957, 0.3291930023422986, 0.1690133273455962]])

    res = computeGaussianWeights((5, 5), 0.3)
    err = res - kernel
    e = np.linalg.norm(err)
    assert e < epsilon


def test_matrix_inversion():
    A = np.array([[12, 4], [4, 8]])
    Ainv = np.array([[0.1, -0.05], [-0.05, 0.15]])
    err = invertMatrix2x2(A) - Ainv
    e = np.linalg.norm(err)
    assert e < 1e-10
