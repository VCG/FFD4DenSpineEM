import numpy as np
from cloudvolume import CloudVolume
from utils import get_conf


def viewer(conf):
    raw = CloudVolume(f"file://{conf.data.raw_layer}")
    spine = CloudVolume(f"file://{conf.data.spine_layer}")
    seg = CloudVolume(f"file://{conf.data.seg_layer}")
    raw_unique = raw.unique(bbox=np.s_[:, :, :])
    spine_unique = spine.unique(bbox=np.s_[:, :, :])
    seg_unique = seg.unique(bbox=np.s_[:, :, :])
    breakpoint()


if __name__ == "__main__":
    conf = get_conf()

    viewer(conf)
