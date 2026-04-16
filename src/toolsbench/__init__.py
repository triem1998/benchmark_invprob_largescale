"""toolsbench package."""

import builtins
import typing


# SimAI-Bench may eagerly import Dragon symbols in worker contexts.
# Provide harmless fallbacks so non-Dragon runs keep working.
if not hasattr(builtins, "Task"):
    builtins.Task = object
if not hasattr(builtins, "Any"):
    builtins.Any = typing.Any
if not hasattr(builtins, "Sequence"):
    builtins.Sequence = typing.Sequence


def main():
    """Console entry point placeholder."""
    print(
        "toolsbench installs shared benchmark utilities. "
        "Run benchmarks with `benchopt run <benchmark_path>`."
    )
