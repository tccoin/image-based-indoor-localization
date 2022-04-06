import ctypes
import platform
import os

lib_paths = {
    "Linux": os.path.dirname(__file__)+"/cmake-build-release/libsearch.so",
    "Windows": os.path.dirname(__file__)+"/cmake-build-release/search.dll"
}

_lib = ctypes.CDLL(lib_paths[platform.system()])
_lib.init_data.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_size_t)
_lib.find_nearest.argtypes = (ctypes.POINTER(ctypes.c_double),)


def init_data(data):
    flat = data.flatten().tolist()
    size = len(flat)
    array_type = ctypes.c_double * size
    _lib.init_data(array_type(*flat), ctypes.c_size_t(size))


def find_nearest(input):
    flat = input.flatten().tolist()
    array_type = ctypes.c_double * 128
    return _lib.find_nearest(array_type(*flat))


def cleanup():
    _lib.cleanup()


if __name__ == "__main__":
    import numpy as np

    data = np.array(range(4000 * 128), dtype=np.float64).reshape((4000, 128))
    data /= 128.0

    init_data(data)

    input = np.ones(128) * 4000.
    nearest = find_nearest(input)

    print(nearest)
    cleanup()
