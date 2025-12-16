"""ML models for violence event insights."""

import sys
import os

# Setup sys.path for importing from parent packages
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(os.path.dirname(_current_dir))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


class BaseAnalyzer:
    """Base class for all insight analyzers."""
    
    def __init__(self):
        """Initialize base analyzer."""
        self.is_fitted = False
    
    def _check_fitted(self) -> None:
        """
        Raise error if analyzer not fitted.
        
        Raises:
            ValueError: If not fitted
        """
        if not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} not fitted. Call fit() first.")
    
    def _ensure_fitted(self) -> None:
        """Alias for _check_fitted() for consistency."""
        self._check_fitted()


from .cluster_analyzer import ClusterAnalyzer
from .association_analyzer import AssociationRuleAnalyzer
from .risk_predictor import RiskPredictor

__all__ = [
    "BaseAnalyzer",
    "ClusterAnalyzer",
    "AssociationRuleAnalyzer",
    "RiskPredictor",
]

