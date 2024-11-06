from tqdm import tqdm
import os
import numpy as np
from joblib import Parallel, delayed
from cloudvolume import CloudVolume, Skeleton

import zarr
from utils import get_conf, pad_slice
from to_precomputed import get_chunks
from visualize import read_mappings
from typing import List, Tuple, Union

import scipy

from kd_feature_transform import kd_feature_transform_chunk


def feature_transform_chunk(
    seed, feature, slice: List[slice], anisotropy: List[float], offset: float
):
    raise NotImplementedError("Use kd_feature_transform_chunk, this is deprecated")
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
    offset = (offset / np.array(anisotropy)).astype(int)
    dilated_slice = np.s_[
        slice[0].start - offset[0] : slice[0].stop + offset[0],
        slice[1].start - offset[1] : slice[1].stop + offset[1],
        slice[2].start - offset[2] : slice[2].stop + offset[2],
    ]
    sliced_seed = pad_slice(seed, dilated_slice, mode="constant")
    # so you can sliced_seed[undilated_slice] = original slice
    undilated_slice = np.s_[
        offset[0] : -offset[0],
        offset[1] : -offset[1],
        offset[2] : -offset[2],
    ]

    edt, inds = scipy.ndimage.distance_transform_edt(
        sliced_seed == 0, sampling=anisotropy, return_indices=True
    )
    closest_to_idx = sliced_seed[inds[0], inds[1], inds[2]][undilated_slice]

    feature[slice] = closest_to_idx


def feature_transform(conf):
    """
    Seed volume with indices of all skeletons (both trunks and spines)
    max radii * radii_multiplier for skeletonization is offset used for chunking

    Generates
    skeletons {id:Skeleton}

    and seeds
    skeleton_ids: [which skeleton (either trunk/skel)]
    vertex_ids: [order of vertices in skeleton]
    seed_ids: [indices of vertices in skeleton] (one-based indexing)
    seed_coords: [coordinates of vertices in skeleton] (isotropic)

    Parameters
    ----------
    conf
    """
    vol = CloudVolume(f"file://{conf.data.output_layer}")
    mapping = np.load(conf.data.mapping)
    # touching = np.load(conf.data.touching)

    seg_to_trunk, trunk_to_segs = read_mappings(mapping)
    seg_ids = sorted(seg_to_trunk.keys())
    trunk_ids = sorted(trunk_to_segs.keys())

    # NOTE: some of these skeletons are empty
    empty_seg_ids = []

    try:
        skeletons = {int(k): vol.skeleton.get(k) for k in seg_ids}
    except Exception as e:
        print(f"Try decreasing dust_threshold, error: {e}")
        raise e

    for seg_id in seg_to_trunk.keys():
        if len(skeletons[seg_id].vertices) == 0:
            empty_seg_ids.append(seg_id)
    if len(empty_seg_ids) > 0:
        raise ValueError(f"Empty seg_ids: {empty_seg_ids}, tune kimimaro params")

    # get canonical labelling for seeding across all skeletons
    skeleton_ids = []
    vertex_ids = []
    seed_coords = []
    max_radius = 0
    for seg_id in sorted(seg_ids):
        skeleton = skeletons[seg_id]
        skeleton_ids.append(np.full(len(skeleton.vertices), seg_id))
        vertex_ids.append(np.arange(len(skeleton.vertices)))
        seed_coords.append(skeleton.vertices)
        max_radius = max(max_radius, np.max(skeleton.radii))
    print(f"max radius: {max_radius}")
    skeleton_ids = np.concatenate(skeleton_ids)
    vertex_ids = np.concatenate(vertex_ids)
    seed_coords = np.concatenate(seed_coords)
    # NOTE: one-based indexing
    seed_dtype = np.min_scalar_type(len(seed_coords))
    seed_ids = np.arange(1, len(seed_coords) + 1, dtype=seed_dtype)

    structured_seeds = np.zeros(
        len(seed_coords),
        dtype=[
            ("skeleton_id", skeleton_ids.dtype),
            ("vertex_id", vertex_ids.dtype),
            ("seed_coord_z", seed_coords.dtype),
            ("seed_coord_y", seed_coords.dtype),
            ("seed_coord_x", seed_coords.dtype),
            ("seed_id", seed_dtype),
        ],
    )
    structured_seeds["skeleton_id"] = skeleton_ids
    structured_seeds["vertex_id"] = vertex_ids
    structured_seeds["seed_coord_z"] = seed_coords[:, 0]
    structured_seeds["seed_coord_y"] = seed_coords[:, 1]
    structured_seeds["seed_coord_x"] = seed_coords[:, 2]
    structured_seeds["seed_id"] = seed_ids
    np.savez(
        conf.data.seed,
        seed_data=structured_seeds,
        skeletons=skeletons,
        max_radius=max_radius,
    )

    assert (
        len(vol.shape) == 4 and vol.shape[-1] == 1
    ), "Expected 4D volume with last dimension 1"

    # input
    seed = zarr.open(
        conf.data.seed_zarr,
        mode="w",
        shape=vol.shape[:-1],  # [z,y,x]
        dtype=seed_dtype,
        chunks=tuple(conf.chunk_size),
        synchronizer=zarr.ProcessSynchronizer(conf.data.seed_zarr_sync),
    )

    anisotropic_seed_coords = (seed_coords / np.array(conf.anisotropy)).astype(int)
    assert not (0 in seed_ids)

    # clamp to within volume
    max_deviation = max(0, np.max(anisotropic_seed_coords - np.array(vol.shape[:-1])))
    print("max deviation of voxels from volume shape (should be small)", max_deviation)

    anisotropic_seed_coords = np.clip(
        anisotropic_seed_coords, 0, np.array(vol.shape[:-1]) - 1
    )

    seed[
        anisotropic_seed_coords[:, 0],
        anisotropic_seed_coords[:, 1],
        anisotropic_seed_coords[:, 2],
    ] = seed_ids

    # output
    feature = zarr.open(
        conf.data.feature_zarr,
        mode="w",
        shape=vol.shape[:-1],  # [z,y,x]
        dtype=seed_dtype,
        chunks=tuple(conf.chunk_size),
        synchronizer=zarr.ProcessSynchronizer(conf.data.feature_zarr_sync),
    )

    chunks = get_chunks(vol.shape, conf.chunk_size)
    # kd_feature_transform_chunk(
    #     seed,
    #     vol,
    #     feature,
    #     chunks[0],
    #     conf.anisotropy,
    #     conf.feature_transform.radii_multiplier * max_radius,
    # )

    res = list(
        tqdm(
            Parallel(n_jobs=conf.n_jobs_feature_transform, return_as="generator")(
                delayed(kd_feature_transform_chunk)(
                    seed,
                    vol,
                    feature,
                    c,
                    conf.anisotropy,
                    conf.feature_transform.radii_multiplier * max_radius,
                )
                for c in chunks
            ),
            total=len(chunks),
            leave=False,
        )
    )


if __name__ == "__main__":
    conf = get_conf()
    feature_transform(conf)
