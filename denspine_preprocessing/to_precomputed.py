from tqdm import tqdm
import os
import numpy as np
from cloudvolume import CloudVolume
from joblib import Parallel, delayed

# import cloudvolume as cv
import h5py
from typing import List, Tuple
from utils import get_conf


def initialize_cloudvolume(
    dataset: str,
    dataset_key: str,
    dataset_layer: str,
    chunk_size: List[int],
    anisotropy: List[int],
):
    output_dir = os.path.dirname(dataset_layer)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert len(chunk_size) == 3
    assert len(anisotropy) == 3

    with h5py.File(dataset, "r") as f:
        # NOTE: not actually reading
        data = f[dataset_key]
        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type="segmentation",
            data_type=data.dtype,
            # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, zfpc, compresso, crackle
            encoding="raw",
            resolution=anisotropy,
            voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
            mesh="mesh",
            skeletons="skeletons",
            # Pick a convenient size for your underlying chunk representation
            # Powers of two are recommended, doesn't need to cover image exactly
            chunk_size=chunk_size,
            volume_size=data.shape,
        )

        # vol is [z, y, x, 1]
        vol = CloudVolume(f"file://{dataset_layer}", info=info)
        vol.commit_info()

    return vol


def write_modified_chunk(
    slice: Tuple[slice],
    output_layer: str,
    raw_dataset: str,
    raw_key: str,
    spine_dataset: str,
    spine_key: str,
    seg_dataset: str,
    seg_key: str,
):
    # raw: [0-10/50] of entire dendrite
    # spine: [0-10/50] if spine
    # seg: cc3d of components
    raw_data = h5py.File(raw_dataset, "r")[raw_key][slice]
    spine_data = h5py.File(spine_dataset, "r")[spine_key][slice]
    seg_data = h5py.File(seg_dataset, "r")[seg_key][slice]

    # seg_den_seg currently triggers this, waiting for new files
    assert np.all((spine_data > 0) == (seg_data > 0))
    # check that seg_data > 0 implies raw_data > 0
    assert np.all(np.logical_or(seg_data == 0, raw_data > 0))

    # give trunks id of original seg
    # NOTE: assumes that raw_labels and seg_labels are disjoint
    output_data = raw_data * (seg_data == 0) + seg_data

    seg_raw_mapping = np.stack(
        [output_data[output_data > 0], raw_data[output_data > 0]], axis=-1
    )
    seg_raw_mapping = np.unique(seg_raw_mapping, axis=0)

    assert len(np.unique(seg_raw_mapping[:, 0])) == seg_raw_mapping.shape[0]

    vol = CloudVolume(f"file://{output_layer}")
    vol[slice] = output_data

    return seg_raw_mapping


def write_chunk(
    slice: Tuple[slice], dataset: str, dataset_key: str, dataset_layer: str
):
    vol = CloudVolume(f"file://{dataset_layer}")
    with h5py.File(dataset, "r") as f:
        data = f[dataset_key][slice]
        vol[slice] = data


def get_chunks(shape: Tuple[int, int, int, int], chunk_size: Tuple[int, int, int]):
    assert len(shape) == 4
    assert len(chunk_size) == 3

    chunks = []
    for z in range(0, shape[0], chunk_size[0]):
        for y in range(0, shape[1], chunk_size[1]):
            for x in range(0, shape[2], chunk_size[2]):
                chunks.append(
                    np.s_[
                        z : min(z + chunk_size[0], shape[0]),
                        y : min(y + chunk_size[1], shape[1]),
                        x : min(x + chunk_size[2], shape[2]),
                    ]
                )
    return chunks


def save_mapping(res, output_file):
    mapping = np.concatenate(res, axis=0)
    mapping = np.unique(mapping, axis=0)

    assert len(np.unique(mapping[:, 0])) == mapping.shape[0]

    np.save(output_file, mapping)


if __name__ == "__main__":
    conf = get_conf()

    output_vol = initialize_cloudvolume(
        conf.data["seg"],
        conf.data["seg_key"],
        conf.data["output_layer"],
        conf.chunk_size,
        conf.anisotropy,
    )
    chunks = get_chunks(tuple(output_vol.shape), tuple(output_vol.chunk_size))

    res = list(
        tqdm(
            Parallel(n_jobs=conf.n_jobs_precompute, return_as="generator")(
                delayed(write_modified_chunk)(
                    c,
                    conf.data["output_layer"],
                    conf.data["raw"],
                    conf.data["raw_key"],
                    conf.data["spine"],
                    conf.data["spine_key"],
                    conf.data["seg"],
                    conf.data["seg_key"],
                )
                for c in chunks
            ),
            total=len(chunks),
            leave=False,
        )
    )

    save_mapping(res, conf.data["mapping"])
