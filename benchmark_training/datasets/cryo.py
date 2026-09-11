"""Synthetic cryo-ET dataset for ``tomo_ei`` training benchmarks.

Single-axis tilt series of a synthetic volume, split into two interleaved
half-sets — the synthetic stand-in for an EMPIAR-11830-like setup
on. No files are read: the volume is generated and both sinograms simulated,
so only the compute path is being measured.
"""

from benchopt import BaseDataset, config
from deepinv.distributed import DistributedContext

from toolsbench.invprob import CryoEIInvProb, InvProbConfig
from toolsbench.utils import setup_distributed_env


class Dataset(BaseDataset):
    # Name of the Dataset, used to select it in the CLI
    name = "cryo"

    parameters = {
        # Astra order (n_slices=Y, n_rows=Z, n_cols=X); Y is the tilt axis.
        "volume_size": [(64, 32, 64)],
        "num_angles": [41],
        "tilt_min": [-60.0],
        "tilt_max": [60.0],
        "noise_level": [0.1],
        "seed": [42],
    }

    def prepare(self):
        return

    def get_data(self):
        """Simulate both half-set tilt series (single batch)."""
        setup_distributed_env()

        # cleanup=False keeps the process group alive for the solver.
        with DistributedContext(seed=self.seed, cleanup=False) as ctx:
            print(f"DistributedContext: rank {ctx.rank} / {ctx.world_size}")

            invprob_conf = InvProbConfig(
                size=tuple(int(s) for s in self.volume_size),
                batch_size=1,
                channels=1,
                device=ctx.device,
                data_path=config.get_data_path(key="synthetic"),
                params={
                    "num_angles": self.num_angles,
                    "tilt_min": self.tilt_min,
                    "tilt_max": self.tilt_max,
                    "noise_level": self.noise_level,
                    "seed": self.seed,
                },
            )
            invprob = CryoEIInvProb().get_invprob(invprob_conf)

        return invprob.asdict()
