import numpy as np
import scipy
import serial


def test_imports():
    assert np.__version__
    assert scipy.__version__
    assert serial.VERSION


def test_numpy_math():
    x = np.arange(4)
    assert x.sum() == 6