"""
Credit Health Intelligence Engine

A backend system for credit agent classification and monitoring.
"""

__version__ = "0.1.0"

# Import key components to make them available at the package level
from .credit_health_engine import CreditHealthEngine
from .feature_engineering import FeatureEngineer
from .agent_classifier import AgentClassifier, TierThresholds

__all__ = [
    'CreditHealthEngine',
    'FeatureEngineer',
    'AgentClassifier',
    'TierThresholds',
]
