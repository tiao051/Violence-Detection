"""
Demo Script - Violence Insights Models (Plain Text Version)

Run this script to see the full output of all analytics models.
No emoji/unicode for Windows compatibility.

Usage:
    python ai_service/insights/demo_plain.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insights.data import ViolenceEventGenerator, ViolenceEvent
from insights.models import TimePatternAnalyzer, LocationAnalyzer


def print_section(title: str):
    """Print a section header."""
    print("\n")
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_mock_data_generation():
    """Demonstrate mock data generation."""
    print_section("STEP 1: MOCK DATA GENERATION")
    
    generator = ViolenceEventGenerator(seed=42)
    
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    events = generator.generate(
        n_events=300,
        start_date=start_date,
        end_date=end_date,
    )
    
    print(f"\n[OK] Generated {len(events)} mock violence events")
    print(f"     Time range: {start_date.date()} to {end_date.date()}")
    
    print("\nSample Events (first 5):")
    print("-" * 70)
    for i, event in enumerate(events[:5], 1):
        print(f"   {i}. {event.camera_name}")
        print(f"      Time: {event.timestamp.strftime('%Y-%m-%d %H:%M')} ({event.day_name}, {event.time_period})")
        print(f"      Confidence: {event.confidence:.2f} (Severity: {event.severity})")
        print()
    
    return events


def demo_time_pattern_analysis(events):
    """Demonstrate time pattern analysis."""
    print_section("STEP 2: TIME PATTERN ANALYSIS")
    
    analyzer = TimePatternAnalyzer()
    analyzer.fit(events)
    summary = analyzer.get_summary()
    
    # 1. Peak Hours
    print("\n[!] PEAK HOURS (Most Dangerous):")
    print("-" * 40)
    for item in summary['peak_hours']:
        bar = "#" * (item['count'] // 3)
        print(f"   {item['label']:>5} : {item['count']:>3} events  {bar}")
    
    # 2. Safest Hours
    print("\n[OK] SAFEST HOURS:")
    print("-" * 40)
    for hour, count in analyzer.get_safest_hours(3):
        print(f"   {hour:02d}:00 : {count:>3} events")
    
    # 3. Hourly Distribution
    print("\n[*] HOURLY DISTRIBUTION:")
    print("-" * 40)
    hourly = summary['hourly_distribution']
    for hour in range(24):
        pct = hourly.get(str(hour), hourly.get(hour, 0))
        bar_len = int(pct * 2)
        bar = "#" * bar_len
        print(f"   {hour:02d}:00 | {bar:<30} {pct:.1f}%")
    
    # 4. Daily Distribution
    print("\n[*] DAILY DISTRIBUTION:")
    print("-" * 40)
    daily = summary['daily_distribution']
    for day, pct in daily.items():
        bar_len = int(pct * 2)
        bar = "#" * bar_len
        print(f"   {day:<9} | {bar:<30} {pct:.1f}%")
    
    # 5. Time Period Analysis
    print("\n[*] TIME PERIOD DISTRIBUTION:")
    print("-" * 40)
    period_dist = summary['period_distribution']
    period_labels = {
        "Morning": "Morning (6-12)",
        "Afternoon": "Afternoon (12-18)",
        "Evening": "Evening (18-22)",
        "Night": "Night (22-6)",
    }
    for period, pct in period_dist.items():
        bar_len = int(pct / 2)
        bar = "#" * bar_len
        print(f"   {period_labels[period]:<22} | {bar:<20} {pct:.1f}%")
    
    # 6. Most Dangerous Period
    dangerous = summary['most_dangerous_period']
    print(f"\n[!!!] MOST DANGEROUS: {dangerous['period']} with {dangerous['count']} events ({dangerous['percentage']}%)")
    
    # 7. Weekend vs Weekday
    print("\n[*] WEEKEND VS WEEKDAY COMPARISON:")
    print("-" * 40)
    weekend = summary['weekend_analysis']
    print(f"   Weekday total: {weekend['weekday_total']} events ({weekend['weekday_percentage']}%)")
    print(f"   Weekend total: {weekend['weekend_total']} events ({weekend['weekend_percentage']}%)")
    print(f"   Weekday avg per day: {weekend['weekday_avg_per_day']:.1f}")
    print(f"   Weekend avg per day: {weekend['weekend_avg_per_day']:.1f}")
    print(f"   >>> Weekend risk ratio: {weekend['weekend_risk_ratio']:.2f}x higher")
    
    # 8. Top Hotspots
    print("\n[HOTSPOTS] Top Hour + Day Combinations:")
    print("-" * 40)
    for i, hs in enumerate(summary['top_hotspots'], 1):
        print(f"   {i}. {hs['day_name']} at {hs['hour_label']}: {hs['count']} events")
    
    return analyzer


def demo_location_analysis(events):
    """Demonstrate location/camera analysis."""
    print_section("STEP 3: LOCATION PATTERN ANALYSIS")
    
    analyzer = LocationAnalyzer()
    analyzer.fit(events)
    summary = analyzer.get_summary()
    
    # 1. Overview
    print(f"\n[*] OVERVIEW:")
    print(f"   Total Events: {summary['total_events']}")
    print(f"   Total Cameras: {summary['total_cameras']}")
    
    # 2. Hotspot Cameras
    print("\n[!] HOTSPOT CAMERAS (Most Incidents):")
    print("-" * 40)
    for cam in summary['hotspot_cameras']:
        bar_len = int(cam['percentage'])
        bar = "#" * bar_len
        print(f"   {cam['camera_name']:<15} | {bar:<30} {cam['event_count']} ({cam['percentage']}%)")
    
    # 3. Camera Distribution
    print("\n[*] ALL CAMERAS DISTRIBUTION:")
    print("-" * 40)
    for cam_id, pct in summary['camera_distribution'].items():
        cam_name = summary['camera_stats'][cam_id]['camera_name']
        bar_len = int(pct)
        bar = "#" * bar_len
        print(f"   {cam_name:<15} | {bar:<30} {pct:.1f}%")
    
    # 4. Risk Ranking
    print("\n[!] CAMERA RISK RANKING:")
    print("-" * 60)
    print(f"   {'Rank':<5} {'Camera':<15} {'Risk':<8} {'Events':<8} {'High Sev%':<10}")
    print("-" * 60)
    for i, cam in enumerate(summary['risk_ranking'], 1):
        risk_level = "HIGH" if i <= 2 else "MED" if i <= 4 else "LOW"
        print(f"   [{risk_level}] {i:<2} {cam['camera_name']:<15} {cam['risk_score']:.3f}   {cam['event_count']:<8} {cam['high_severity_pct']:.1f}%")
    
    # 5. Confidence Statistics
    print("\n[*] OVERALL CONFIDENCE STATISTICS:")
    print("-" * 40)
    conf = summary['overall_confidence']
    print(f"   Mean:   {conf['mean']:.3f}")
    print(f"   Median: {conf['median']:.3f}")
    print(f"   Min:    {conf['min']:.3f}")
    print(f"   Max:    {conf['max']:.3f}")
    print(f"   Std:    {conf['std']:.3f}")
    
    # 6. Camera Details
    print("\n[*] DETAILED CAMERA STATS:")
    print("-" * 70)
    for cam_id, stats in summary['camera_stats'].items():
        sev = stats['severity_distribution']
        print(f"\n   {stats['camera_name']}:")
        print(f"      Events: {stats['event_count']} ({stats['percentage']}%)")
        print(f"      Avg Confidence: {stats['confidence']['mean']:.3f}")
        print(f"      Severity: Low={sev['Low']}, Medium={sev['Medium']}, High={sev['High']}")
    
    return analyzer


def main():
    """Run the full demo."""
    print("\n")
    print("=" * 70)
    print("         VIOLENCE INSIGHTS DEMO - Analytics Models")
    print("=" * 70)
    
    # Step 1: Generate mock data
    events = demo_mock_data_generation()
    
    # Step 2: Time pattern analysis
    time_analyzer = demo_time_pattern_analysis(events)
    
    # Step 3: Location analysis
    location_analyzer = demo_location_analysis(events)
    
    # Summary
    print_section("DEMO COMPLETE")
    print("\nWhat the models can do:")
    print("   * TimePatternAnalyzer: Identify WHEN violence is most likely")
    print("   * LocationAnalyzer: Identify WHERE violence is most likely")
    print("\nNext steps:")
    print("   * Integrate with real Firestore data")
    print("   * Create API endpoints for frontend")
    print("   * Add visualization charts")
    print("\n")


if __name__ == "__main__":
    main()
