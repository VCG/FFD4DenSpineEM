import igneous.task_creation as tc
from taskqueue import LocalTaskQueue
from utils import get_conf
import cloudvolume


def main(conf):
    tq = LocalTaskQueue(parallel=conf.n_jobs_skeletonize)

    layer = f"file://{conf.data.output_layer}"
    print(conf.downsample)

    try:
        volume = cloudvolume.CloudVolume(layer)
    except:
        print(
            "delete previous mip files, and skeleton files which may have been created with previous mips"
        )
        raise UserWarning("Layer does not exist")

    for i in range(len(conf.downsample.factors)):
        print(f"Downsampling mip {i}")
        downsample_tasks = tc.create_downsampling_tasks(
            layer,
            mip=i,
            num_mips=1,
            factor=tuple(conf.downsample.factors[i]),
            chunk_size=tuple(conf.downsample.chunk_size),
        )
        tq.insert(downsample_tasks)
        tq.execute()


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
