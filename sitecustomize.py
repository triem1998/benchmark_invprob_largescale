"""Global Python startup customizations for benchmark worker processes.

This module is imported automatically by Python at startup (when available on
``sys.path``). We define Dragon symbols eagerly imported by SimAI-Bench so
non-Dragon environments (e.g., local Ray workers) can import cleanly.
"""

import builtins
import typing


if not hasattr(builtins, "Task"):
    builtins.Task = object
if not hasattr(builtins, "Any"):
    builtins.Any = typing.Any
if not hasattr(builtins, "Sequence"):
    builtins.Sequence = typing.Sequence
