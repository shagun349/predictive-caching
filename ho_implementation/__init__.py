"""
Hippopotamus Optimization (HO) Algorithm for Predictive Multi-Tier Caching.
===========================================================================
"""
from ho_implementation.ho_hippo import Hippo
from ho_implementation.ho_engine import HOEngine
from ho_implementation.ho_predictive_policy import HOPredictivePolicy

__all__ = ["Hippo", "HOEngine", "HOPredictivePolicy"]
