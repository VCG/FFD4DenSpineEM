import os
import glob
import numpy as np
from joblib import Parallel, delayed
import h5py
from to_precomputed import get_chunks
from tqdm import tqdm
from utils import get_conf
import imageio

"""
just used to debug seg_den_seg
for some reason the ids are between 0-255, despite the dtype being uint16
there are like 300 unique segments in the mapping returned by to_precomputed output_layer for mouse and human
"""


def get_unique_chunk(
    chunk, raw, raw_key, spine, spine_key, seg, seg_key, new_trunk, new_trunk_key
):
    raw_chunk = h5py.File(raw, "r")[raw_key][chunk]
    spine_chunk = h5py.File(spine, "r")[spine_key][chunk]
    seg_chunk = h5py.File(seg, "r")[seg_key][chunk]
    new_trunk_chunk = h5py.File(new_trunk, "r")[new_trunk_key][chunk]

    raw_unique = np.unique(raw_chunk)
    spine_unique = np.unique(spine_chunk)
    seg_unique = np.unique(seg_chunk)
    new_trunk_unique = np.unique(new_trunk_chunk)

    num_contradict_spine_seg = np.sum((spine_chunk > 0) != (seg_chunk > 0))
    num_contradict_new_trunk_raw = np.sum((new_trunk_chunk > 0) & (raw_chunk > 0))

    res = {
        "raw": raw_unique,
        "spine": spine_unique,
        "seg": seg_unique,
        "num_contradict_spine_seg": num_contradict_spine_seg,
        "num_contradict_new_trunk_raw": num_contradict_new_trunk_raw,
        "new_trunk": new_trunk_unique,
    }

    return res


def generate_10_png_vol(dir: str, output: str, output_key: str):
    files = sorted(glob.glob(os.path.join(dir, "*.png")))
    imgs = list(
        tqdm(
            Parallel(n_jobs=conf.n_jobs_debug, return_as="generator")(
                delayed(imageio.imread)(f) for f in files
            ),
            total=len(files),
            leave=False,
        )
    )
    new_seg = np.stack(imgs, axis=0).astype(np.uint16)

    file = h5py.File(output, "w")
    file.create_dataset(output_key, data=new_seg)


def main(conf):
    generate_10_png_vol(
        conf.data.broken_10_png_dir,
        conf.data.broken_10_png_vol,
        conf.data.broken_10_png_vol_key,
    )

    seg_file = h5py.File(conf.data.broken_seg, "r")[conf.data.seg_key]
    # NOTE: overriding chunks
    chunks = get_chunks(tuple(seg_file.shape) + (1,), (256, 512, 512))
    res = list(
        tqdm(
            Parallel(n_jobs=conf.n_jobs_debug, return_as="generator")(
                delayed(get_unique_chunk)(
                    c,
                    conf.data.broken_raw,
                    conf.data.broken_raw_key,
                    conf.data.broken_spine,
                    conf.data.broken_spine_key,
                    conf.data.broken_seg,
                    conf.data.broken_seg_key,
                    conf.data.broken_10_png_vol,
                    conf.data.broken_10_png_vol_key,
                )
                for c in chunks
            ),
            total=len(chunks),
            leave=False,
        )
    )

    merged_res = {
        "raw": set(),
        "spine": set(),
        "seg": set(),
        "new_trunk": set(),
        "num_contradict_spine_seg": 0,
        "num_contradict_new_trunk_raw": 0,
    }
    for r in res:
        merged_res["raw"].update(r["raw"])
        merged_res["spine"].update(r["spine"])
        merged_res["seg"].update(r["seg"])
        merged_res["new_trunk"].update(r["new_trunk"])
        merged_res["num_contradict_spine_seg"] += r["num_contradict_spine_seg"]
        merged_res["num_contradict_new_trunk_raw"] += r["num_contradict_new_trunk_raw"]
    merged_res["raw"] = np.array(sorted(merged_res["raw"]))
    merged_res["spine"] = np.array(sorted(merged_res["spine"]))
    merged_res["seg"] = np.array(sorted(merged_res["seg"]))
    merged_res["new_trunk"] = np.array(sorted(merged_res["new_trunk"]))

    np.savez(conf.data.broken_debug, merged_res=merged_res)


"""
# np.savez("seg_den_debug.npz", res=res)
merged_res
{'raw': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
      dtype=uint8), 'spine': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
      dtype=uint8), 'seg': array([   0,    1,    2, ..., 4224, 4225, 4226], dtype=uint16), 'num_contradict': 27926940}
>>> merged_res["num_contradict"]
27926940
>>> seg_file = h5py.File(conf.data.seg, "r")[conf.data.seg_key]
>>> seg_file.shape
(1849, 12120, 10740)
>>> np.prod(seg_file.shape)
240682111200
>>> 27926940/240682111200
0.00011603247063423632
"""


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
