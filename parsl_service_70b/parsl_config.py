import parsl
from parsl.config import Config
# PBSPro is the right provider for Sunspot:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# Use the MPI launcher
from parsl.launchers import MpiExecLauncher
from parsl.addresses import address_by_interface
from parsl.app.app import python_app, bash_app
import os

# Adjust your user-specific options here:
worker_init = '; '.join(open(f"{os.path.join(os.path.dirname(os.path.abspath(__file__)),'worker_init.sh')}").read().strip().split('\n'))
user_opts = {
    "worker_init": worker_init,
    "scheduler_options":"" ,
    "account":          "Aurora_deployment",
    "queue":            "workq",
    "walltime":         "00:60:00",
    "run_dir":          f"{os.path.dirname(os.path.abspath(__file__))}/runinfo",
    "nodes_per_block":  1, # think of a block as one job on sunspot
    "cpus_per_node":    208, # this is the number of threads available on one sunspot node
    "strategy":         "simple",
}

# Set the name of accelerators.  We will treat each tile as an accelerator (12 total)  
accel_ids=[]
for gid in range(6):
    for tid in range(2):
        accel_ids.append(f"{gid}.{tid}")

sunspot_config = Config(
    run_dir=user_opts["run_dir"],
    retries=2,
    executors=[
        HighThroughputExecutor(
            label="sunspot_llm_70b",
            available_accelerators=accel_ids,  # Ensures one worker per accelerator
            address=address_by_interface("bond0"),
            cpu_affinity="block",  # Assigns cpus in sequential order
            prefetch_capacity=0,  # Increase if you have many more tasks than workers
            max_workers=12,
            cores_per_worker=16, # How many workers per core dictates total workers per node
            provider=PBSProProvider(
                account=user_opts["account"],
                queue=user_opts["queue"],
                worker_init=user_opts["worker_init"],
                walltime=user_opts["walltime"],
                scheduler_options=user_opts["scheduler_options"],
                launcher=MpiExecLauncher(
                    bind_cmd="--cpu-bind", overrides="--depth=208 --ppn 1"
                ),  # Ensures 1 manger per node and allows it to divide work among all 208 threads
                select_options="system=sunspot,place=scatter",
                nodes_per_block=user_opts["nodes_per_block"],
                min_blocks=0,
                max_blocks=1, # Can increase more to have more parallel batch jobs
                cpus_per_node=user_opts["cpus_per_node"],
            ),
        ),
    ]
)