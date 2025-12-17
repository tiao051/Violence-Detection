"""
Association Rule Mining using FP-Growth Algorithm.

Discovers patterns in violence events:
  "IF Saturday AND Evening → High severity"
  "IF Parking Lot AND Night → Violence likely"
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from . import BaseAnalyzer
from ..time_utils import categorize_hour
from insights.core.schema import ViolenceEvent

from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


class AssociationRuleAnalyzer(BaseAnalyzer):
    """
    Discovers association rules in violence events using FP-Growth.
    """
    
    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.5,
        min_lift: float = 1.0,
    ):
        super().__init__()
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        
        self.transactions: List[List[str]] = []
        self.frequent_itemsets: Optional[pd.DataFrame] = None
        self.rules: Optional[pd.DataFrame] = None
        
    def _event_to_items(self, event: ViolenceEvent) -> List[str]:
        items = []
        items.append(f"day_{event.day_name}")
        items.append(f"period_{event.time_period}")
        items.append("is_weekend" if event.is_weekend else "is_weekday")
        items.append(f"hour_{categorize_hour(event.hour)}")
        items.append(f"camera_{event.camera_name.replace(' ', '_')}")
        items.append(f"severity_{event.severity}")
        
        if event.confidence >= 0.85:
            items.append("confidence_high")
        elif event.confidence >= 0.7:
            items.append("confidence_medium")
        else:
            items.append("confidence_low")
        
        return items
    
    def fit(self, events: List[ViolenceEvent]) -> "AssociationRuleAnalyzer":
        if len(events) < 10:
            raise ValueError("Need at least 10 events for meaningful rules")
        
        self.transactions = [self._event_to_items(e) for e in events]
        
        te = TransactionEncoder()
        te_array = te.fit_transform(self.transactions)
        df = pd.DataFrame(te_array, columns=te.columns_)
        
        self.frequent_itemsets = fpgrowth(df, min_support=self.min_support, use_colnames=True)
        
        if len(self.frequent_itemsets) == 0:
            print("Warning: No frequent itemsets found. Try lowering min_support.")
            self.is_fitted = True
            return self
        
        self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        
        if len(self.rules) > 0:
            self.rules = self.rules[self.rules['lift'] >= self.min_lift]
        
        self.is_fitted = True
        return self
    
    def get_frequent_itemsets(self, top_n: int = 20) -> List[Dict[str, Any]]:
        self._check_fitted()
        
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            return []
        
        sorted_itemsets = self.frequent_itemsets.sort_values('support', ascending=False)
        
        return [
            {
                "items": list(row['itemsets']),
                "support": round(row['support'], 3),
                "support_pct": round(row['support'] * 100, 1),
            }
            for _, row in sorted_itemsets.head(top_n).iterrows()
        ]
    
    def get_rules(self, top_n: int = 20) -> List[Dict[str, Any]]:
        self._check_fitted()
        
        if self.rules is None or len(self.rules) == 0:
            return []
        
        sorted_rules = self.rules.sort_values('lift', ascending=False)
        
        return [
            {
                "antecedent": list(row['antecedents']),
                "consequent": list(row['consequents']),
                "antecedent_str": " AND ".join(row['antecedents']),
                "consequent_str": " AND ".join(row['consequents']),
                "support": round(row['support'], 3),
                "confidence": round(row['confidence'], 3),
                "lift": round(row['lift'], 3),
                "rule_str": f"IF {' AND '.join(row['antecedents'])} → {' AND '.join(row['consequents'])}",
            }
            for _, row in sorted_rules.head(top_n).iterrows()
        ]
    
    def get_rules_for_target(self, target: str, top_n: int = 10) -> List[Dict[str, Any]]:
        self._check_fitted()
        
        if self.rules is None or len(self.rules) == 0:
            return []
        
        target_rules = self.rules[self.rules['consequents'].apply(lambda x: target in x)]
        
        if len(target_rules) == 0:
            return []
        
        sorted_rules = target_rules.sort_values('confidence', ascending=False)
        
        return [
            {
                "antecedent": list(row['antecedents']),
                "antecedent_str": " AND ".join(row['antecedents']),
                "target": target,
                "confidence": round(row['confidence'], 3),
                "confidence_pct": round(row['confidence'] * 100, 1),
                "lift": round(row['lift'], 3),
                "insight": f"When {' AND '.join(row['antecedents'])}, {round(row['confidence'] * 100, 1)}% chance of {target}",
            }
            for _, row in sorted_rules.head(top_n).iterrows()
        ]
    
    def get_summary(self) -> Dict[str, Any]:
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
