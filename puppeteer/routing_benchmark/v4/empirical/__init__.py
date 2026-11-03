"""Empirical-output data pipeline for RoutingBench V4.

The package deliberately lives beside, rather than inside, the deterministic
V4-beta generator.  This keeps the published beta immutable while empirical
outputs, judge annotations, and human review are assembled into a new release.
"""

from .schema import EMPIRICAL_SCHEMA_VERSION

__all__ = ["EMPIRICAL_SCHEMA_VERSION"]
