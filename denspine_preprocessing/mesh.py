from cloudvolume import Vec
import igneous.task_creation as tc
from taskqueue import LocalTaskQueue
from utils import get_conf


def main(conf):
    tq = LocalTaskQueue(parallel=conf.n_jobs_mesh)
    layer = f"file://{conf.data.output_layer}"

    tasks = tc.create_meshing_tasks(  # First Pass
        layer, **conf.mesh  # Which data layer
    )
    tq.insert(tasks)
    tq.execute()

    tasks = tc.create_mesh_manifest_tasks(layer, **conf.mesh_merge)
    tq.insert(tasks)
    tq.execute()


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
