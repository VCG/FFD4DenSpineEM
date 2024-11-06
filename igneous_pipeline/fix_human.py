import os
import numpy as np
import imageio
import glob
import h5py
from joblib import Parallel, delayed
from to_precomputed import get_chunks

from utils import get_conf
from tqdm import tqdm


def fix_chunk(
    chunk,
    stats,
    broken_raw_file,
    broken_spine_file,
    broken_seg_file,
    broken_10_png_vol_file,
    raw_file,
    spine_file,
    seg_file,
):
    all_seg_unique = set(stats["raw"].tolist() + stats["seg"].tolist()) - set([0])
    new_trunk_unique = set(stats["new_trunk"]) - set([0])
    # assert 0 intersection
    assert len(all_seg_unique.intersection(new_trunk_unique)) == 0
    assert 10 in stats["new_trunk"]
    assert 10 not in stats["raw"]

    broken_raw_chunk = broken_raw_file[chunk]
    broken_spine_chunk = broken_spine_file[chunk]
    broken_seg_chunk = broken_seg_file[chunk]
    new_trunk = broken_10_png_vol_file[chunk]

    # integrate new missing trunk
    raw_file[chunk] = broken_raw_chunk + (10 * (new_trunk > 0)).astype(
        broken_raw_chunk.dtype
    )
    spine_file[chunk] = broken_spine_chunk + (10 * (new_trunk > 10)).astype(
        broken_spine_chunk.dtype
    )
    seg_file[chunk] = broken_seg_chunk + ((new_trunk > 10) * new_trunk).astype(
        broken_seg_chunk.dtype
    )


def fix(conf):
    stats = np.load(conf.data.broken_debug, allow_pickle=True)["merged_res"].item()

    # add the 10th dendrite to seg
    broken_raw_file = h5py.File(conf.data.broken_raw, "r")[conf.data.broken_raw_key]
    broken_spine_file = h5py.File(conf.data.broken_spine, "r")[
        conf.data.broken_spine_key
    ]
    broken_seg_file = h5py.File(conf.data.broken_seg, "r")[conf.data.broken_seg_key]
    broken_10_png_vol_file = h5py.File(conf.data.broken_10_png_vol, "r")[
        conf.data.broken_10_png_vol_key
    ]
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

    # NOTE: overriding chunks
    chunks = get_chunks(tuple(seg_file.shape) + (1,), (256, 512, 512))
    for chunk in tqdm(chunks, desc="Fixing chunks"):
        fix_chunk(
            chunk,
            stats,
            broken_raw_file,
            broken_spine_file,
            broken_seg_file,
            broken_10_png_vol_file,
            raw_file,
            spine_file,
            seg_file,
        )


if __name__ == "__main__":
    conf = get_conf()
    fix(conf)
