"""
Theoretical Framework for Emergent Specialization.

This module provides formal definitions, propositions, and proof sketches
for the emergent specialization phenomenon in multi-agent systems.
"""

from .definitions import (
    RegimeDefinition,
    NichePartitioningTheory,
    verify_regime_properties,
)

from .propositions import (
    Proposition1_EquilibriumSpecialization,
    Proposition2_SIConvergence,
)

__all__ = [
    "RegimeDefinition",
    "NichePartitioningTheory",
    "verify_regime_properties",
    "Proposition1_EquilibriumSpecialization",
    "Proposition2_SIConvergence",
]
