import numpy as np
import h5py
import zarr
from utils import get_conf, groupby
from to_precomputed import get_chunks
from joblib import Parallel, delayed
import numcodecs
from tqdm import tqdm
from typing import List, Tuple


def append_to_point_cloud(
    feature: zarr.Array,
    group: zarr.Group,
    chunk: Tuple[slice, slice, slice],
    raw: str,
    raw_key: str,
    spine: str,
    spine_key: str,
    seg: str,
    seg_key: str,
    structured_dtype: List[Tuple[str, np.dtype]],
) -> int:
    """
    Given the Euclidean feature transform, and raw, spine, seg chunks
    append point clouds and raw/spine/seg values to the corresponding group dataset.

    The data is given in [Nx7] structured array format
    z, y, x, feature, raw, spine, seg

    Returns the number of points added in total, used for verifying that parallelism appends each row

    Parameters
    ----------
    feature
    group
    chunk
    raw
    raw_key
    spine
    spine_key
    seg
    seg_key
    """

    """
    When using multiple processes to parallelize reads or writes on arrays using the Blosc compression library, it may be necessary to set numcodecs.blosc.use_threads = False, as otherwise Blosc may share incorrect global state amongst processes causing programs to hang.
    """
    # https://zarr.readthedocs.io/en/stable/tutorial.html#parallel-computing-and-synchronization
    numcodecs.blosc.use_threads = False

    assert len(structured_dtype) == 7
    assert [x[0] for x in structured_dtype] == [
        "z",
        "y",
        "x",
        "feature",
        "raw",
        "spine",
        "seg",
    ]

    feature_chunk = feature[chunk]
    with h5py.File(raw, "r") as f:
        raw_chunk = f[raw_key][chunk]
    with h5py.File(spine, "r") as f:
        spine_chunk = f[spine_key][chunk]
    with h5py.File(seg, "r") as f:
        seg_chunk = f[seg_key][chunk]

    # [z*y*x, 3]
    chunk_indices = np.indices(feature_chunk.shape).reshape(3, -1).T
    chunk_offset = np.array([s.start for s in chunk])

    absolute_positions = chunk_indices + chunk_offset

    # NOTE: this assumes raw is complete
    is_valid = raw_chunk > 0
    num_elements = np.sum(is_valid)
    data = np.empty(num_elements, structured_dtype)

    data["z"] = absolute_positions[is_valid.reshape(-1), 0]
    data["y"] = absolute_positions[is_valid.reshape(-1), 1]
    data["x"] = absolute_positions[is_valid.reshape(-1), 2]
    data["feature"] = feature_chunk[is_valid]
    data["raw"] = raw_chunk[is_valid]
    data["spine"] = spine_chunk[is_valid]
    data["seg"] = seg_chunk[is_valid]
    del feature_chunk, raw_chunk, spine_chunk, seg_chunk
    del chunk_indices, absolute_positions, is_valid

    unique_ids, grouped_data = groupby(data, data["feature"])
    del data

    # Append the grouped positions to the corresponding group dataset
    for seed_id, data_group in zip(unique_ids, grouped_data):
        dataset = group[f"{seed_id}"]
        assert hasattr(dataset, "_cache_metadata")
        # see https://github.com/zarr-developers/zarr-python/issues/2077
        # NOTE: no way to open dataset in group with cache_metadata=False
        dataset._cache_metadata = False

        dataset.append(data_group)

    return num_elements


def to_point_cloud(conf):
    feature = zarr.open(conf.data.feature_zarr, "r")
    seed = np.load(conf.data.seed)
    seed_ids = seed["seed_data"]["seed_id"]

    group = zarr.group(
        conf.data.pc_zarr,
        overwrite=True,
        synchronizer=zarr.ProcessSynchronizer(conf.data.pc_zarr_sync),
    )

    structured_dtype = [
        ("z", np.min_scalar_type(max(feature.shape))),
        ("y", np.min_scalar_type(max(feature.shape))),
        ("x", np.min_scalar_type(max(feature.shape))),
        ("feature", feature.dtype),
        ("raw", h5py.File(conf.data.raw, "r")[conf.data.raw_key].dtype),
        ("spine", h5py.File(conf.data.spine, "r")[conf.data.spine_key].dtype),
        ("seg", h5py.File(conf.data.seg, "r")[conf.data.seg_key].dtype),
    ]

    # NOTE: cache_metadata=False is necessary, see
    # https://github.com/zarr-developers/zarr-python/issues/2077
    for seed_id in tqdm(seed_ids, desc="Initializing datasets"):
        group.zeros(
            f"{seed_id}",
            shape=0,
            chunks=conf.point_cloud.chunk_size,
            dtype=structured_dtype,
            cache_metadata=False,
        )

    chunks = get_chunks(tuple(feature.shape) + (1,), conf.chunk_size)

    res = list(
        tqdm(
            Parallel(n_jobs=conf.n_jobs_pc, return_as="generator")(
                delayed(append_to_point_cloud)(
                    feature,
                    group,
                    c,
                    conf.data.raw,
                    conf.data.raw_key,
                    conf.data.spine,
                    conf.data.spine_key,
                    conf.data.seg,
                    conf.data.seg_key,
                    structured_dtype,
                )
                for c in chunks
            ),
            total=len(chunks),
        )
    )
    lengths = {}
    total_gt = sum(res)
    total_pred = 0
    for seed_id in tqdm(seed_ids, desc="Verifying"):
        lengths[seed_id] = len(group[seed_id])
        total_pred += lengths[seed_id]
    assert total_gt == total_pred, "Parallel append bug"

    np.savez(conf.data.pc_lengths, lengths=lengths)


if __name__ == "__main__":
    conf = get_conf()
    to_point_cloud(conf)
