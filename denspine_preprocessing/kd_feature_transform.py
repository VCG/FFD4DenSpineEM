import numpy as np
from typing import List
from utils import pad_slice, groupby
from dataloader import get_closest


def _kd_feature_transform_chunk(
    seed_vol: np.ndarray, modified_vol: np.ndarray, anisotropy: np.ndarray
) -> np.ndarray:
    """
    Given a seed and a segmentation, for each voxel in the segmentation, find the closest seed within that segment

    Parameters
    ----------
    seed_vol: [z, y, x]
    modified_vol : [z, y, x]
        segmentation including TRUNKS, not the seg_vol as is
        alternatively, this can be the SPINE volume, if seed_vol only contains trunk points (not implemented)
    anisotropy : List[float]

    Returns
    -------
    np.ndarray of type seed.dtype, 0s where there is no seed or for background
    """
    # NOTE: later verify that there are no 0s
    assert len(anisotropy) == 3
    assert seed_vol.shape == modified_vol.shape and len(seed_vol.shape) == 3

    # [N, 3]
    seed_idx = np.argwhere(seed_vol != 0)
    seed_val = seed_vol[seed_idx[:, 0], seed_idx[:, 1], seed_idx[:, 2]]
    seed_seg = modified_vol[seed_idx[:, 0], seed_idx[:, 1], seed_idx[:, 2]]
    # create structured data
    seed_data = np.zeros(
        len(seed_idx),
        dtype=[
            ("z", seed_idx.dtype),
            ("y", seed_idx.dtype),
            ("x", seed_idx.dtype),
            ("seed_id", seed_val.dtype),
            ("seg_id", seed_seg.dtype),
        ],
    )
    seed_data["z"] = seed_idx[:, 0]
    seed_data["y"] = seed_idx[:, 1]
    seed_data["x"] = seed_idx[:, 2]
    seed_data["seed_id"] = seed_val
    seed_data["seg_id"] = seed_seg

    unique_seed_seg, seed_data_grouped = groupby(seed_data, seed_seg)
    seed_seg_to_seed_data = dict(zip(unique_seed_seg, seed_data_grouped))

    # [M, 3]
    seg_idx = np.argwhere(modified_vol != 0)
    seg_ids = modified_vol[seg_idx[:, 0], seg_idx[:, 1], seg_idx[:, 2]]

    # NOTE: segments unmapped to seeds will be 0
    output = np.zeros_like(seed_vol)

    unique_seg_ids, seg_idx_grouped = groupby(seg_idx, seg_ids)
    for unique_seg_id, seg_idx_group in zip(unique_seg_ids, seg_idx_grouped):
        if unique_seg_id not in seed_seg_to_seed_data:
            continue

        seed_data_group = seed_seg_to_seed_data[unique_seg_id]
        seed_pc = np.stack(
            [seed_data_group["z"], seed_data_group["y"], seed_data_group["x"]], axis=1
        ) * np.array(anisotropy)
        seg_pc = seg_idx_group * np.array(anisotropy)

        dist, idx = get_closest(seg_pc, seed_pc)
        closest_seed_id = seed_data_group["seed_id"][idx]

        output[
            seg_idx_group[:, 0], seg_idx_group[:, 1], seg_idx_group[:, 2]
        ] = closest_seed_id

    return output


def kd_feature_transform_chunk(
    seed,
    modified,
    feature,
    input_slice: List[slice],
    anisotropy: List[float],
    offset: float,
):
    """
    Given two zarr arrays, seed and feature, and the slice for seed, calculate the feature transform
    i.e., for every voxel calculate the closest seed point (skeleton point), taking into account anisotropy
    and also adding padding to the slices to handle chunking

    Parameters
    ----------
    seed
    feature
    slice
    anisotropy
    offset
    """
    assert (
        len(modified.shape) == 4
        and modified.shape[-1] == 1
        and modified.shape[:3] == seed.shape
    )

    offset = (offset / np.array(anisotropy)).astype(int)
    dilated_slice = np.s_[
        input_slice[0].start - offset[0] : input_slice[0].stop + offset[0],
        input_slice[1].start - offset[1] : input_slice[1].stop + offset[1],
        input_slice[2].start - offset[2] : input_slice[2].stop + offset[2],
    ]
    sliced_seed = pad_slice(seed, dilated_slice, mode="constant")
    sliced_modified = pad_slice(
        modified, dilated_slice + (slice(None),), mode="constant"
    ).squeeze(-1)
    assert sliced_seed.shape == sliced_modified.shape
    # so you can sliced_seed[undilated_slice] = original slice
    undilated_slice = np.s_[
        offset[0] : -offset[0],
        offset[1] : -offset[1],
        offset[2] : -offset[2],
    ]

    closest_to_idx = _kd_feature_transform_chunk(
        sliced_seed, sliced_modified, anisotropy
    )[undilated_slice]
    # check that there (modified > 0) => (closest_to_idx > 0)

    num_invalid = np.sum(
        (np.logical_and(sliced_modified[undilated_slice] > 0, closest_to_idx == 0))
    )
    if num_invalid > 0:
        print(
            f"WARNING: {num_invalid} modified > 0 where closest_to_idx is 0, try increasing offset, chunk: {input_slice}"
        )

    feature[input_slice] = closest_to_idx
