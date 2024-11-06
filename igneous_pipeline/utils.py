import os
import inspect
from joblib import Memory
from dirhash import dirhash
from functools import wraps
import numpy as np
import argparse
from omegaconf import OmegaConf
from typing import List, Union, Tuple


class DotDict(dict):
    # modified from https://stackoverflow.com/a/13520518/10702372
    """
    A dictionary that supports dot notation as well as dictionary access notation.
    Usage: d = DotDict() or d = DotDict({'val1':'first'})
    Set attributes: d.val2 = 'second' or d['val2'] = 'second'
    Get attributes: d.val2 or d['val2']

    NOTE: asserts that dictionary does not contain tuples (YAML)
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if isinstance(value, dict):
                value = DotDict(value)
            elif isinstance(value, list):
                value = self._convert_list(value)
            self[key] = value

    def _convert_list(self, lst):
        new_list = []
        for item in lst:
            if isinstance(item, dict):
                new_list.append(DotDict(item))
            elif isinstance(item, list):
                new_list.append(self._convert_list(item))
            else:
                new_list.append(item)
        return new_list

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)


def get_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        help="List of configuration files.",
        required=True,
    )

    args = parser.parse_args()

    confs = [OmegaConf.load(c) for c in args.config]
    conf = OmegaConf.merge(*confs)

    # cast to dictionary, because hash of OmegaConf fields depend on greater object
    conf = OmegaConf.to_container(conf, resolve=True)
    assert isinstance(conf, dict), "conf must be a dictionary"
    # allow dot access
    conf = DotDict(conf)

    return conf


def pad_slice(
    vol: np.ndarray, slices: List[Union[slice, int]], mode: str
) -> np.ndarray:
    """
    Given a n-dim volume (np-like array which supports np.s_ slicing) and a slice which may be out of bounds,
    zero-pad the volume to the dimensions of the slice

    the slices have to be in one of the following formats:
        - int (e.g. vol[0])
        - slice(None, None, None) (e.g. vol[:])
        - slice(start, stop, None) (e.g. vol[0:10]) -> the dimensions here will be padded

    output dimension will be
        - 1 if the slice is an int
        - (stop - start) if start and stop are not None
        - vol.shape[i] if start is None and stop is None

    notably, it does not handle ellipsis or np.newaxis

    Parameters
    ----------
    vol
    slices
    mode: np.pad mode
    """
    assert len(vol.shape) == len(
        slices
    ), f"Volume and slices must have the same number of dimensions, given {len(vol.shape)} and {len(slices)}"
    for i, s in enumerate(slices):
        if isinstance(s, int):
            continue
        else:
            assert isinstance(
                s, slice
            ), f"Slice must be an int or a slice, given {type(s)}"
            assert s.step is None, f"Slice step must be None, given {s.step}"
            assert (s.start is None) == (
                s.stop is None
            ), f"Slice start and stop must both be None or not None, given {s.start} and {s.stop}"
            if s.start is not None:
                assert (
                    s.start < s.stop
                ), f"Slice start must be less than stop, given {s.start} and {s.stop}"
                # NOTE: s.start is allowed to be negative
                assert (
                    s.start < vol.shape[i]
                ), f"Slice start must be less than volume shape, given {s.start} and {vol.shape[i]}"
                assert s.stop > 0, f"Slice stop must be greater than 0, given {s.stop}"

    output_shape = []
    for i, s in enumerate(slices):
        if isinstance(s, int):
            output_shape.append(1)
            assert (
                0 <= s < vol.shape[i]
            ), f"Slice {s} is out of bounds for dimension {i}, which has size {vol.shape[i]}"
        else:
            output_shape.append(
                s.stop - s.start if s.start is not None else vol.shape[i]
            )

    input_slices = []
    for i, s in enumerate(slices):
        if isinstance(s, int):
            input_slices.append(s)
        else:
            if s.start is None:
                input_slices.append(slice(None))
            else:
                input_slices.append(slice(max(0, s.start), min(vol.shape[i], s.stop)))

    pad_widths = []
    for i, s in enumerate(slices):
        if isinstance(s, int) or s.start is None:
            pad_widths.append((0, 0))
        else:
            pad_widths.append(
                (
                    max(0, -s.start),
                    max(0, s.stop - vol.shape[i]),
                )
            )

    # so if scalar is indexed i.e. np.arange(5)[0], shape will be [1] instead of ()
    output = np.zeros(output_shape, dtype=vol.dtype)
    output[:] = np.pad(vol[tuple(input_slices)], pad_widths, mode=mode)

    return output


def groupby(array: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    given an array of shape [N, ...] and an idx of shape [N], returns unique_idx and the grouped arrays
    [np.stack([array[i] where idx matches], axis=0), ...]

    heavily based on https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function

    check unit_test.py
    """
    assert idx.ndim == 1, f"idx must be 1D, given {idx.ndim}"
    assert array.shape[0] == idx.shape[0], f"array and idx must have same length"

    order = np.argsort(idx)
    array = array[order]
    idx = idx[order]

    unique_idx, index = np.unique(idx, return_index=True)
    groups = np.split(array, index[1:], axis=0)

    return unique_idx, groups


def hash_file_or_dir(path: str, algorithm: str = "md5") -> str:
    """
    Given a path to a file or directory, return the hash of the file or directory

    Parameters
    ----------
    path
    algorithm

    Returns
    -------
    str
    """
    assert os.path.exists(path), f"Path argument does not exist: {path}"
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return dirhash(path, algorithm)
    else:
        dirpath = os.path.dirname(path)
        filename = os.path.basename(path)
        return dirhash(dirpath, algorithm, match=[filename])


def cache_path(mem: Memory, path_args: Union[str, List[str]]):
    """
    Given a function that takes in path arguments, @mem.cache the function based on the hash of the path arguments
    Works by passing the hash into a cached function that ignores the hash and calls the original function

    Usage:
    @cache_path(mem, "path")
    def my_func(path):
        pass

    Parameters
    ----------
    mem
    path_args
    """
    # adapted from decorator pattern from https://stackoverflow.com/a/42581103/10702372

    if isinstance(path_args, str):
        path_args = [path_args]

    def decorator(function):
        # assigns __name__ to ignore_path_hash, necessary since mem.cache assigns based on __name__
        @wraps(function)
        def ignore_path_hash(path_hash, *args, **kwargs):
            return function(*args, **kwargs)

        def wrapper(*args, **kwargs):
            callargs = inspect.getcallargs(function, *args, **kwargs)
            assert all(
                [key in callargs for key in path_args]
            ), f"Missing keys: {path_args}"

            path_hash = [hash_file_or_dir(callargs[key]) for key in path_args]

            return mem.cache(ignore_path_hash)(path_hash, *args, **kwargs)

        return wrapper

    return decorator
