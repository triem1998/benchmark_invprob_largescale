#!/usr/bin/env python3
"""Submitit/local launcher for cryo-ET demos.

Usage
-----
# Local run (uses GPU if available):
    python main.py --config conf_supervised_local.yml

# SLURM via submitit:
    python main.py --config conf_supervised.yml

"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

# Allow `python main.py` from repo root without editable install.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import submitit  # noqa: E402

if TYPE_CHECKING:
    pass


def _print_config(cfg, header: str = "RunConfig") -> None:
    lines = [f"[config] {header}"]
    for key, val in asdict(cfg).items():
        lines.append(f"  {key}: {val}")
    print("\n".join(lines), flush=True)


class CryoTrainingJob:
    def __init__(self, method: str, cfg_dict: dict):
        self.method = method
        self.cfg_dict = cfg_dict
        # Store src path at construction time so it survives pickling to worker nodes.
        self._src = str(Path(__file__).resolve().parent / "src")

    def __call__(self):
        import sys

        if self._src not in sys.path:
            sys.path.insert(0, self._src)

        submitit.helpers.TorchDistributedEnvironment().export(
            set_cuda_visible_devices=False
        )

        env = submitit.JobEnvironment()

        if self.method == "supervised":
            from supervised.run_supervised import RunConfig, run_training

            cfg = RunConfig(**self.cfg_dict)
            cfg.output_dir = str(Path(cfg.output_dir) / f"slurm-{env.job_id}")
        else:
            raise ValueError(f"Unknown method: {self.method}")

        print(
            f"[submitit] job_id={env.job_id} rank={env.global_rank} "
            f"local_rank={env.local_rank} world_size={env.num_tasks}",
            flush=True,
        )
        if env.global_rank == 0:
            _print_config(cfg)
        return run_training(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submitit/local launcher for supervised cryo-ET demo (Method 1)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "conf_supervised_local.yml"),
        help="Path to YAML config file (conf_supervised_local.yml or conf_supervised.yml).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force local mode regardless of general.execution_mode in config.",
    )
    return parser.parse_args()


def _require_section(conf: dict, name: str) -> dict:
    section = conf.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid '{name}' section in config.")
    return section


def load_config(path: str | Path) -> dict:
    import yaml

    conf_path = Path(path)
    if not conf_path.exists():
        raise FileNotFoundError(f"Config file not found: {conf_path}")

    with conf_path.open("r", encoding="utf-8") as f:
        conf = yaml.safe_load(f) or {}

    if not isinstance(conf, dict):
        raise ValueError("Top-level YAML config must be a dictionary.")

    for section_name in ("general", "training", "slurm"):
        _require_section(conf, section_name)

    return conf


def _normalize_checkpoint_batches(value):
    if value is None:
        return None
    if str(value).lower() in {"none", "null"}:
        return None
    return value


def _parse_target_shape(value):
    """Return a (D, H, W) int tuple, or None if value is None/null."""
    if value is None:
        return None
    return tuple(int(v) for v in value)


def build_run_config(conf: dict):
    method = str(conf.get("method", "supervised")).lower()
    if method == "supervised":
        return "supervised", _build_supervised_config(conf)
    else:
        raise ValueError(f"Unknown method '{method}'. Only 'supervised' is supported.")


def _build_supervised_config(conf: dict):
    from supervised.run_supervised import RunConfig

    general = _require_section(conf, "general")
    training = _require_section(conf, "training")
    distributed = _require_section(conf, "distributed")
    slurm = _require_section(conf, "slurm")

    default = RunConfig()

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = str(
        general.get("run_name", slurm.get("job_name", "demo-cryo-supervised"))
    )
    output_root = Path(general.get("output_root", "./runs"))
    out_dir = output_root / f"{run_name}_{timestamp}"

    return RunConfig(
        output_dir=str(out_dir),
        input_dir=str(general.get("input_dir", default.input_dir)),
        batch_size=int(general.get("batch_size", default.batch_size)),
        num_workers=int(general.get("num_workers", default.num_workers)),
        pin_memory=bool(general.get("pin_memory", default.pin_memory)),
        prefetch_factor=int(general.get("prefetch_factor", default.prefetch_factor)),
        persistent_workers=bool(
            general.get("persistent_workers", default.persistent_workers)
        ),
        max_train_vols=general.get("max_train_vols", default.max_train_vols),
        max_val_vols=int(general.get("max_val_vols", default.max_val_vols)),
        target_shape=_parse_target_shape(
            general.get("target_shape", default.target_shape)
        ),
        seed=int(general.get("seed", default.seed)),
        num_epochs=int(training.get("num_epochs", default.num_epochs)),
        learning_rate=float(training.get("learning_rate", default.learning_rate)),
        grad_clip=training.get("grad_clip", default.grad_clip),
        grad_accumulation_steps=int(
            training.get("grad_accumulation_steps", default.grad_accumulation_steps)
        ),
        ckp_interval=int(training.get("ckp_interval", default.ckp_interval)),
        eval_interval=int(training.get("eval_interval", default.eval_interval)),
        distribute_model=bool(
            distributed.get("distribute_model", default.distribute_model)
        ),
        patch_size=tuple(
            int(v) for v in distributed.get("patch_size", default.patch_size)
        ),
        overlap=tuple(int(v) for v in distributed.get("overlap", default.overlap)),
        max_batch_size=distributed.get("max_batch_size", default.max_batch_size),
        checkpoint_batches=_normalize_checkpoint_batches(
            distributed.get("checkpoint_batches", default.checkpoint_batches)
        ),
    )


def submit_job(method: str, cfg, slurm: dict) -> None:
    submitit_folder = Path(cfg.output_dir) / "submitit_logs"
    submitit_folder.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=str(submitit_folder), slurm_python="python")
    gpus_per_node = int(slurm.get("gpus_per_node", 1))
    additional_params = dict(slurm.get("additional_parameters", {}))
    additional_params.update(
        {
            "ntasks-per-node": int(slurm.get("ntasks_per_node", 1)),
            "cpus-per-task": int(slurm.get("cpus_per_task", 4)),
            "account": str(slurm.get("account", "fio@h100")),
            "constraint": str(slurm.get("constraint", "h100")),
            "qos": str(slurm.get("qos", "qos_gpu_h100-dev")),
        }
    )

    executor.update_parameters(
        name=str(slurm.get("job_name", "demo-cryo-supervised")),
        nodes=int(slurm.get("nodes", 1)),
        slurm_gres=str(slurm.get("gres", f"gpu:{gpus_per_node}")),
        slurm_time=str(slurm.get("time", "02:00:00")),
        slurm_stderr_to_stdout=bool(slurm.get("stderr_to_stdout", True)),
        slurm_additional_parameters=additional_params,
        slurm_setup=list(
            slurm.get(
                "setup",
                [
                    "module purge",
                    "module load arch/h100",
                    "module load pytorch-gpu/py3/2.7.0",
                    "export NCCL_DEBUG=INFO",
                ],
            )
        ),
    )

    job = executor.submit(CryoTrainingJob(method, asdict(cfg)))
    print(f"Submitted job: {job.job_id}")
    print(f"Submitit logs: {submitit_folder.resolve()}")


def main() -> None:
    args = parse_args()
    conf = load_config(args.config)
    general = _require_section(conf, "general")
    slurm = _require_section(conf, "slurm")

    execution_mode = str(general.get("execution_mode", "local")).lower()
    if args.local:
        execution_mode = "local"

    method, cfg = build_run_config(conf)

    if execution_mode == "local":
        if method == "supervised":
            from supervised.run_supervised import run_training
        else:
            raise ValueError(f"Unknown method: {method}")

        _print_config(cfg)
        run_training(cfg)
        return

    submit_job(method, cfg, slurm)


if __name__ == "__main__":
    main()
