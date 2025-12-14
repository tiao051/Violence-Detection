"""
VIOLENCE INSIGHTS - DEMO FOR TASK GIVER

This script demonstrates all 3 ML models built for violence insights.

Run: python ai_service/insights/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insights.data import ViolenceEventGenerator
from insights.algorithms import ClusterAnalyzer, AssociationRuleAnalyzer, RiskPredictor


def print_header(title):
    print("\n")
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("\n" * 2)
    print("#" * 70)
    print("#" + " " * 20 + "VIOLENCE INSIGHTS DEMO" + " " * 26 + "#")
    print("#" + " " * 15 + "3 ML Models for Pattern Discovery" + " " * 19 + "#")
    print("#" * 70)
    
    # Generate sample data
    print("\n[1] Generating sample violence events...")
    generator = ViolenceEventGenerator(seed=42)
    events = generator.generate(n_events=500)
    print(f"    Generated {len(events)} events for analysis")
    
    # ==================== MODEL 1: K-MEANS CLUSTERING ====================
    print_header("MODEL 1: K-MEANS CLUSTERING")
    print("Algorithm: K-means (scikit-learn)")
    print("Purpose: Group similar violence events into patterns")
    
    cluster_analyzer = ClusterAnalyzer(n_clusters=3)
    cluster_analyzer.fit(events)
    
    insights = cluster_analyzer.get_cluster_insights()
    print("\nDiscovered Patterns:")
    for i, pattern in enumerate(insights):
        print(f"\n  Cluster {i+1}: {pattern['size']} events ({pattern['percentage']}%)")
        print(f"    - Time: {pattern['top_period']} (avg hour: {pattern['avg_hour']:.0f}:00)")
        print(f"    - Location: {pattern['top_camera']}")
        print(f"    - Weekend: {pattern['weekend_pct']}%")
        print(f"    - Severity: {pattern['high_severity_pct']}% high severity")
    
    # ==================== MODEL 2: FP-GROWTH ASSOCIATION RULES ====================
    print_header("MODEL 2: FP-GROWTH ASSOCIATION RULES")
    print("Algorithm: FP-Growth (mlxtend)")
    print("Purpose: Find association rules like 'IF A AND B THEN C'")
    
    assoc_analyzer = AssociationRuleAnalyzer(min_support=0.05, min_confidence=0.5)
    assoc_analyzer.fit(events)
    
    rules = assoc_analyzer.get_rules(top_n=5)
    print("\nTop Association Rules:")
    for i, rule in enumerate(rules[:5], 1):
        print(f"\n  Rule {i}: {rule['rule_str']}")
        print(f"    Confidence: {rule['confidence']:.0%} | Lift: {rule['lift']:.2f}")
    
    # High severity rules
    high_sev_rules = assoc_analyzer.get_rules_for_target("severity_High", top_n=3)
    if high_sev_rules:
        print("\nRules leading to HIGH severity:")
        for rule in high_sev_rules:
            print(f"  - {rule['insight']}")
    
    # ==================== MODEL 3: RANDOM FOREST PREDICTION ====================
    print_header("MODEL 3: RANDOM FOREST PREDICTION")
    print("Algorithm: Random Forest Classifier (scikit-learn)")
    print("Purpose: Predict risk level and % change vs average")
    
    predictor = RiskPredictor(n_estimators=100)
    predictor.fit(events)
    
    print(f"\nModel Accuracy: {predictor.accuracy:.1%}")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print("\nFeature Importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "#" * int(imp * 30)
        print(f"  {feat:<12} {bar} {imp:.1%}")
    
    # Example predictions
    print("\nExample Predictions:")
    test_cases = [
        {"hour": 20, "day": "Saturday", "camera": "Ngã ba Âu Cơ"},
        {"hour": 8, "day": "Monday", "camera": "Ngã tư Lê Trọng Tấn"},
        {"hour": 23, "day": "Friday", "camera": "Ngã tư Cộng Hòa"},
    ]
    for case in test_cases:
        result = predictor.predict(**case)
        print(f"  - {result['insight']}")
    
    # High risk conditions
    high_risk = predictor.get_high_risk_conditions(top_n=3)
    print("\nHigh Risk Conditions (when to be most alert):")
    for cond in high_risk:
        print(f"  - {cond['day']} {cond['hour']:02d}:00 at {cond['camera']}: {cond['high_prob']:.0%} HIGH risk")
    
    # ==================== SUMMARY ====================
    print_header("SUMMARY")
    print("""
    +------------------+------------------+--------------------------------+
    | Model            | Algorithm        | Output                         |
    +------------------+------------------+--------------------------------+
    | ClusterAnalyzer  | K-means          | Groups similar events          |
    | AssociationRule  | FP-Growth        | IF-THEN rules with confidence  |
    | RiskPredictor    | Random Forest    | Risk prediction with % change  |
    +------------------+------------------+--------------------------------+
    
    All models use ACTUAL ML ALGORITHMS from scikit-learn and mlxtend,
    not just SQL queries or simple statistics.
    """)
    
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
