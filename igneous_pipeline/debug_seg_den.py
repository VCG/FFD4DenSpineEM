import numpy as np
from joblib import Parallel, delayed
import h5py
from to_precomputed import get_chunks
from tqdm import tqdm
from utils import get_conf
from collections import defaultdict

"""
just used to debug seg_den_seg
for some reason the ids are between 0-255, despite the dtype being uint16
there are like 300 unique segments in the mapping returned by to_precomputed output_layer for mouse and human
"""


def get_unique_chunk(chunk, raw, raw_key, spine, spine_key, seg, seg_key, new_branches):
    raw_chunk = h5py.File(raw, "r")[raw_key][chunk]
    spine_chunk = h5py.File(spine, "r")[spine_key][chunk]
    seg_chunk = h5py.File(seg, "r")[seg_key][chunk]

    new_branch_trunk_ids = [x["trunk_id"] for x in new_branches]
    new_branches_chunks = [
        h5py.File(x["file"], "r")[x["key"]][chunk] for x in new_branches
    ]

    raw_unique = np.unique(raw_chunk)
    spine_unique = np.unique(spine_chunk)
    seg_unique = np.unique(seg_chunk)
    new_branches_unique = [np.unique(x) for x in new_branches_chunks]

    num_contradict = np.sum((spine_chunk > 0) != (seg_chunk > 0))

    res = {
        "raw": raw_unique,
        "spine": spine_unique,
        "seg": seg_unique,
        "num_contradict": num_contradict,
        # make it dict {trunk_id: unique}
        "new_branches": {
            k: v for k, v in zip(new_branch_trunk_ids, new_branches_unique)
        },
    }

    return res


def main(conf):
    seg_file = h5py.File(conf.data.broken_seg, "r")[conf.data.seg_key]
    chunks = get_chunks(tuple(seg_file.shape) + (1,), tuple(seg_file.chunks))
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
                    # list of dicts {trunk_id: int, file: str, key: str}
                    conf.data.broken_new_branches,
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
        "new_branches": defaultdict(set),
        "num_contradict": 0,
    }
    for r in res:
        merged_res["raw"].update(r["raw"])
        merged_res["spine"].update(r["spine"])
        merged_res["seg"].update(r["seg"])
        for k, v in r["new_branches"].items():
            merged_res["new_branches"][k].update(v)
        merged_res["num_contradict"] += r["num_contradict"]
    merged_res["raw"] = np.array(sorted(merged_res["raw"]))
    merged_res["spine"] = np.array(sorted(merged_res["spine"]))
    merged_res["seg"] = np.array(sorted(merged_res["seg"]))
    merged_res["new_branches"] = dict(
        {k: np.array(sorted(v)) for k, v in merged_res["new_branches"].items()}
    )

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
