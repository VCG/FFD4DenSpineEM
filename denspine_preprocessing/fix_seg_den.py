import os
from tqdm import tqdm
import h5py
import numpy as np

from utils import get_conf
from to_precomputed import get_chunks
from typing import Tuple

"""
Given broken_raw, broken_spine, and broken_seg

where the following assertions are broken
spine_data > 0 iff seg_data > 0
seg_data > 0 -> raw_data > 0

but the following holds:
spine_data > 0 -> raw_data > 0

and force seg labels to start from max(raw_data) + 1

unfortunately this cannot be parallelized since h5 does not support parallel write
"""


def implies(a, b):
    return np.logical_or(np.logical_not(a), b)


def fix_chunk(
    chunk: Tuple[slice, slice, slice],
    stats,
    broken_raw_file,
    broken_spine_file,
    broken_seg_file,
    broken_new_branches_files,
    raw_file,
    spine_file,
    seg_file,
):
    broken_raw_chunk = broken_raw_file[chunk]
    broken_spine_chunk = broken_spine_file[chunk]
    broken_seg_chunk = broken_seg_file[chunk]
    broken_new_branches_chunks = {
        k: v[chunk] for k, v in broken_new_branches_files.items()
    }

    # integrate new missing branches
    for trunk_id in broken_new_branches_chunks:
        broken_raw_chunk = np.maximum(
            broken_raw_chunk,
            (trunk_id * (broken_new_branches_chunks[trunk_id] > 0)).astype(
                broken_raw_chunk.dtype
            ),
        )

    assert np.all(
        implies(broken_spine_chunk > 0, broken_raw_chunk > 0)
    ), "spine_data > 0 -> raw_data > 0 failed"
    assert np.array_equal(
        stats["raw"], stats["spine"]
    ), "unique raw and spine values are not equal"
    for trunk_id in broken_new_branches_chunks:
        assert trunk_id in stats["raw"], f"{trunk_id} not in raw values"
    offset = stats["raw"].max()
    assert offset > 0
    assert (
        offset + max(stats["seg"]) < np.iinfo(seg_file.dtype).max
    ), "overflow in seg labels"
    # increase non-zero seg_labels by offset

    offseted_seg_chunk = broken_seg_chunk + (offset * (broken_seg_chunk > 0))

    valid_mask = (broken_spine_chunk > 0) == (offseted_seg_chunk > 0)

    raw_file[chunk] = broken_raw_chunk * valid_mask
    spine_file[chunk] = broken_spine_chunk * valid_mask
    seg_file[chunk] = offseted_seg_chunk * valid_mask

    num_invalid = np.sum(np.logical_not(valid_mask))

    return num_invalid


def fix(conf):
    stats = np.load(conf.data.broken_debug, allow_pickle=True)["merged_res"].item()

    broken_raw_file = h5py.File(conf.data.broken_raw, "r")[conf.data.broken_raw_key]
    broken_spine_file = h5py.File(conf.data.broken_spine, "r")[
        conf.data.broken_spine_key
    ]
    broken_seg_file = h5py.File(conf.data.broken_seg, "r")[conf.data.broken_seg_key]
    broken_new_branches_files = {
        x["trunk_id"]: h5py.File(x["file"], "r")[x["key"]]
        for x in conf.data.broken_new_branches
    }

    assert broken_raw_file.shape == broken_spine_file.shape == broken_seg_file.shape

    raw_file = h5py.File(conf.data.raw, "w")
    raw_file.create_dataset(
        conf.data.raw_key,
        shape=broken_raw_file.shape,
        chunks=broken_raw_file.chunks,
        dtype=broken_raw_file.dtype,
        compression=broken_raw_file.compression,
    )
    spine_file = h5py.File(conf.data.spine, "w")
    spine_file.create_dataset(
        conf.data.spine_key,
        shape=broken_spine_file.shape,
        chunks=broken_spine_file.chunks,
        dtype=broken_spine_file.dtype,
        compression=broken_spine_file.compression,
    )
    seg_file = h5py.File(conf.data.seg, "w")
    seg_file.create_dataset(
        conf.data.seg_key,
        shape=broken_seg_file.shape,
        chunks=broken_seg_file.chunks,
        dtype=broken_seg_file.dtype,
        compression=broken_seg_file.compression,
    )
    raw_file, spine_file, seg_file = (
        raw_file[conf.data.raw_key],
        spine_file[conf.data.spine_key],
        seg_file[conf.data.seg_key],
    )

    chunks = get_chunks(tuple(seg_file.shape) + (1,), tuple(seg_file.chunks))

    num_invalid = 0
    for chunk in tqdm(chunks, desc="Fixing chunks"):
        num_invalid += fix_chunk(
            chunk,
            stats,
            broken_raw_file,
            broken_spine_file,
            broken_seg_file,
            broken_new_branches_files,
            raw_file,
            spine_file,
            seg_file,
        )
    print(f"Fixed {num_invalid} invalid values")


if __name__ == "__main__":
    conf = get_conf()
    fix(conf)
