"""Materialization and execution gates for RoutingBench V4 domain packets."""

from .builder import build_domain_packets
from .roles import DOMAIN_ROLE_CARDS, role_cards_for

__all__ = ["DOMAIN_ROLE_CARDS", "build_domain_packets", "role_cards_for"]
