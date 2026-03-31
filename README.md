# benchmark_invprob_largescale

Large-scale benchmarking suite for inverse-problem pipelines with:
- an **inference benchmark** (`benchmark_inference/`),
- a **workflow/streaming benchmark** (`benchmark_workflows/`),
- shared utilities in the **`toolsbench`** package (`src/toolsbench/`).

This repository is organized as a Python package and is designed to run with `uv` + `benchopt`.

## Repository Structure

```text
benchmark_invprob_largescale/
├── benchmark_inference/           # Benchopt benchmark: large-scale inference
│   ├── configs/                   # YAML configs (highres, tomography, ...)
│   ├── datasets/
│   ├── solvers/
│   └── objective.py
├── benchmark_workflows/           # Benchopt benchmark: workflow/streaming setting
│   ├── datasets/
│   ├── solvers/
│   └── objective.py
├── src/toolsbench/                # Shared package used by both benchmarks
│   ├── utils/
│   └── analysis_plots/
└── docs/                          # Sphinx documentation
```

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Optional for radio benchmark components: Singularity/Apptainer + cluster tooling

## Setup With `uv`

From the repository root:

```bash
uv sync
```

This creates/updates `.venv` and installs:
- core benchmark dependencies (`torch`, `deepinv`, `simai-bench`, `benchopt`, etc.),
- local package `toolsbench` in editable mode.

Optional extras:

```bash
# Radio-specific extras
uv sync --extra radio

# Documentation extras
uv sync --extra docs
```

## Running Benchmarks

Benchopt CLI (v1.9+) uses:

```bash
benchopt run <benchmark_path>
```

So with `uv`:

### 1) Inference benchmark with a config

```bash
uv run benchopt run benchmark_inference/. \
  --config benchmark_inference/configs/highres_imaging.yml
```

### 2) Workflow benchmark

```bash
uv run benchopt run benchmark_workflows/. \
  --max-runs 1 --n-repetitions 1 --no-plot --no-html --no-display
```

### 3) Parallel runs (cluster config)

```bash
uv run benchopt run benchmark_inference/. \
  --parallel-config benchmark_inference/configs/config_parallel.yml \
  --config benchmark_inference/configs/highres_imaging.yml
```

## Useful Benchopt Commands

```bash
# List discovered datasets/solvers/objective
uv run benchopt info benchmark_inference/.
uv run benchopt info benchmark_workflows/.
```

## Outputs

By default, results are written under:
- `benchmark_inference/outputs/`
- `benchmark_workflows/outputs/`

with generated parquet files and HTML reports.

## Documentation

Build docs locally:

```bash
uv sync --extra docs
uv run make -C docs html
```

Generated site:

```text
docs/build/html/
```
