"""RoutingBench: backbone-independent evaluation for dynamic agent routers."""

from .environment import RoutingEnvironment
from .evaluator import evaluate_track_a, evaluate_track_b
from .loader import BenchmarkBundle, load_benchmark
from .routers import FixedSequenceRouter, OracleRouter, RandomRouter

__all__ = [
    "BenchmarkBundle",
    "FixedSequenceRouter",
    "OracleRouter",
    "RandomRouter",
    "RoutingEnvironment",
    "evaluate_track_a",
    "evaluate_track_b",
    "load_benchmark",
]
