import numpy as np
import cloudvolume
import igneous.task_creation as tc
from taskqueue import LocalTaskQueue
from utils import get_conf
from visualize import read_mappings


def main(conf):
    tq = LocalTaskQueue(parallel=conf.n_jobs_skeletonize)

    mapping = np.load(conf.data.mapping)
    seg_to_trunk, trunk_to_segs = read_mappings(mapping)

    seg_ids = sorted(seg_to_trunk.keys())
    trunk_ids = sorted(trunk_to_segs.keys())
    spine_ids = sorted(list(set(seg_ids) - set(trunk_ids)))

    layer = f"file://{conf.data.output_layer}"

    trunk_conf = conf.skeletonize_trunk
    trunk_teasar_conf = trunk_conf.pop("teasar_params")
    assert isinstance(trunk_teasar_conf, list) and len(trunk_teasar_conf) >= 1
    if len(trunk_teasar_conf) == 1:
        mip = trunk_teasar_conf[0].pop("mip")
        skeletonize_trunk_tasks = tc.create_skeletonizing_tasks(
            layer,
            **trunk_conf,
            mip=mip,
            teasar_params=trunk_teasar_conf[0],
            object_ids=trunk_ids,
        )
        tq.insert(skeletonize_trunk_tasks)
    else:
        assert (
            sorted([item for sublist in trunk_teasar_conf for item in sublist.ids])
            == trunk_ids
        ), "Trunk ids do not match, either duplicate or missing"
        for teasar_params in trunk_teasar_conf:
            object_ids = teasar_params.pop("ids")
            mip = teasar_params.pop("mip")
            skeletonize_trunk_tasks = tc.create_skeletonizing_tasks(
                layer,
                **trunk_conf,
                mip=mip,
                teasar_params=teasar_params,
                object_ids=object_ids,
            )
            tq.insert(skeletonize_trunk_tasks)

    spine_conf = conf.skeletonize_spines
    spine_teasar_conf = spine_conf.pop("teasar_params")
    assert isinstance(spine_teasar_conf, list) and len(spine_teasar_conf) >= 1
    if len(spine_teasar_conf) == 1:
        mip = spine_teasar_conf[0].pop("mip")
        skeletonize_spine_tasks = tc.create_skeletonizing_tasks(
            layer,
            **spine_conf,
            mip=mip,
            teasar_params=spine_teasar_conf[0],
            object_ids=spine_ids,
        )
        tq.insert(skeletonize_spine_tasks)
    else:
        assert (
            sorted([item for sublist in spine_teasar_conf for item in sublist.ids])
            == spine_ids
        ), "spine ids do not match, either duplicate or missing"
        for teasar_params in spine_teasar_conf:
            object_ids = teasar_params.pop("ids")
            mip = teasar_params.pop("mip")
            skeletonize_spine_tasks = tc.create_skeletonizing_tasks(
                layer,
                **spine_conf,
                mip=mip,
                teasar_params=teasar_params,
                object_ids=object_ids,
            )
            tq.insert(skeletonize_spine_tasks)

    tq.execute()

    merge_tasks = tc.create_unsharded_skeleton_merge_tasks(
        layer, **conf.skeletonize_merge
    )
    tq.insert(merge_tasks)
    tq.execute()


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
