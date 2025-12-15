"""
Association Rule Mining using FP-Growth Algorithm.

Discovers "IF-THEN" patterns in violence events:
  "IF Saturday AND Evening → High severity"
  "IF Parking Lot AND Night → Violence likely"

Uses mlxtend's FP-Growth for efficient itemset mining.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from insights.core.schema import ViolenceEvent

# FP-Growth imports
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


class AssociationRuleAnalyzer:
    """
    Discovers association rules in violence events using FP-Growth.
    """
    
    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.5,
        min_lift: float = 1.0,
    ):
        """
        Initialize association rule analyzer.
        
        Args:
            min_support: Minimum support threshold (0-1)
            min_confidence: Minimum confidence threshold (0-1)
            min_lift: Minimum lift threshold (>1 means positive correlation)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        
        # Results
        self.transactions: List[List[str]] = []
        self.frequent_itemsets: Optional[pd.DataFrame] = None
        self.rules: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False
        
    def _event_to_items(self, event: ViolenceEvent) -> List[str]:
        """Convert event to transaction items (categorical features)."""
        items = []
        
        # Temporal features
        items.append(f"day_{event.day_name}")
        items.append(f"period_{event.time_period}")
        items.append("is_weekend" if event.is_weekend else "is_weekday")
        
        # Hour buckets
        hour = event.hour
        if hour < 6:
            items.append("hour_late_night")
        elif hour < 12:
            items.append("hour_morning")
        elif hour < 18:
            items.append("hour_afternoon")
        else:
            items.append("hour_evening")
        
        # Location and severity
        items.append(f"camera_{event.camera_name.replace(' ', '_')}")
        items.append(f"severity_{event.severity}")
        
        # Confidence level
        if event.confidence >= 0.85:
            items.append("confidence_high")
        elif event.confidence >= 0.7:
            items.append("confidence_medium")
        else:
            items.append("confidence_low")
        
        return items
    
    def fit(self, events: List[ViolenceEvent]) -> "AssociationRuleAnalyzer":
        """
        Discover association rules from events.
        
        Args:
            events: List of ViolenceEvent instances (min. 10 events required)
            
        Returns:
            self for method chaining
        """
        if len(events) < 10:
            raise ValueError("Need at least 10 events for meaningful rules")
        
        # Convert events to transactions
        self.transactions = [self._event_to_items(e) for e in events]
        
        # Encode transactions
        te = TransactionEncoder()
        te_array = te.fit_transform(self.transactions)
        df = pd.DataFrame(te_array, columns=te.columns_)
        
        # Find frequent itemsets
        self.frequent_itemsets = fpgrowth(
            df,
            min_support=self.min_support,
            use_colnames=True,
        )
        
        if len(self.frequent_itemsets) == 0:
            print("Warning: No frequent itemsets found. Try lowering min_support.")
            self.is_fitted = True
            return self
        
        # Generate association rules
        self.rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence,
        )
        
        # Filter by lift
        if len(self.rules) > 0:
            self.rules = self.rules[self.rules['lift'] >= self.min_lift]
        
        self.is_fitted = True
        return self
    
    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self.is_fitted:
            raise ValueError("Analyzer not fitted. Call fit() first.")
    
    def get_frequent_itemsets(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get top frequent itemsets.
        
        Returns:
            List of itemset dicts with items and support
        """
        self._check_fitted()
        
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            return []
        
        # Sort by support
        sorted_itemsets = self.frequent_itemsets.sort_values('support', ascending=False)
        
        results = []
        for _, row in sorted_itemsets.head(top_n).iterrows():
            results.append({
                "items": list(row['itemsets']),
                "support": round(row['support'], 3),
                "support_pct": round(row['support'] * 100, 1),
            })
        
        return results
    
    def get_rules(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get top association rules.
        
        Returns:
            List of rule dicts with antecedent, consequent, confidence, lift
        """
        self._check_fitted()
        
        if self.rules is None or len(self.rules) == 0:
            return []
        
        # Sort by lift (most interesting rules first)
        sorted_rules = self.rules.sort_values('lift', ascending=False)
        
        results = []
        for _, row in sorted_rules.head(top_n).iterrows():
            antecedent = list(row['antecedents'])
            consequent = list(row['consequents'])
            
            results.append({
                "antecedent": antecedent,
                "consequent": consequent,
                "antecedent_str": " AND ".join(antecedent),
                "consequent_str": " AND ".join(consequent),
                "support": round(row['support'], 3),
                "confidence": round(row['confidence'], 3),
                "lift": round(row['lift'], 3),
                "rule_str": f"IF {' AND '.join(antecedent)} → {' AND '.join(consequent)}",
            })
        
        return results
    
    def get_rules_for_target(self, target: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get rules that lead to a specific target (consequent).
        
        Args:
            target: Target item to find rules for (e.g., "severity_High")
            top_n: Number of rules to return
            
        Returns:
            List of rules leading to target
        """
        self._check_fitted()
        
        if self.rules is None or len(self.rules) == 0:
            return []
        
        # Filter rules with target in consequent
        target_rules = self.rules[
            self.rules['consequents'].apply(lambda x: target in x)
        ]
        
        if len(target_rules) == 0:
            return []
        
        # Sort by confidence
        sorted_rules = target_rules.sort_values('confidence', ascending=False)
        
        results = []
        for _, row in sorted_rules.head(top_n).iterrows():
            antecedent = list(row['antecedents'])
            
            results.append({
                "antecedent": antecedent,
                "antecedent_str": " AND ".join(antecedent),
                "target": target,
                "confidence": round(row['confidence'], 3),
                "confidence_pct": round(row['confidence'] * 100, 1),
                "lift": round(row['lift'], 3),
                "insight": f"When {' AND '.join(antecedent)}, {round(row['confidence'] * 100, 1)}% chance of {target}",
            })
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of association rule mining results."""
        self._check_fitted()
        
        return {
            "algorithm": "FP-Growth",
            "parameters": {
                "min_support": self.min_support,
                "min_confidence": self.min_confidence,
                "min_lift": self.min_lift,
            },
            "total_transactions": len(self.transactions),
            "frequent_itemsets_count": len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0,
            "rules_count": len(self.rules) if self.rules is not None else 0,
            "top_itemsets": self.get_frequent_itemsets(10),
            "top_rules": self.get_rules(10),
            "high_severity_rules": self.get_rules_for_target("severity_High", 5),
        }
