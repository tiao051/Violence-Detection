"""
Time Pattern Analyzer

Analyzes temporal patterns in violence detection events to identify:
- Peak hours for violence incidents
- Day-of-week patterns
- Time period analysis (Morning/Afternoon/Evening/Night)
- Heatmap data for hour x day visualization

This is a statistical analysis model, not ML-based.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from insights.data.event_schema import ViolenceEvent


class TimePatternAnalyzer:
    """
    Analyzes temporal patterns in violence events.
    
    Provides insights on when violence is most likely to occur,
    enabling proactive security measures during high-risk periods.
    
    Example:
        >>> from insights.data import ViolenceEventGenerator
        >>> generator = ViolenceEventGenerator(seed=42)
        >>> events = generator.generate(n_events=200)
        >>> 
        >>> analyzer = TimePatternAnalyzer()
        >>> analyzer.fit(events)
        >>> 
        >>> print(analyzer.get_peak_hours(top_n=3))
        >>> print(analyzer.get_hourly_distribution())
    """
    
    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    TIME_PERIODS = ["Morning", "Afternoon", "Evening", "Night"]
    
    def __init__(self):
        """Initialize the analyzer."""
        self.events: List[ViolenceEvent] = []
        self.is_fitted: bool = False
        
        # Computed statistics
        self._hourly_counts: Dict[int, int] = {}
        self._daily_counts: Dict[int, int] = {}
        self._period_counts: Dict[str, int] = {}
        self._heatmap: np.ndarray = None  # 24 x 7 matrix
        
    def fit(self, events: List[ViolenceEvent]) -> "TimePatternAnalyzer":
        """
        Fit the analyzer with event data.
        
        Args:
            events: List of ViolenceEvent instances
            
        Returns:
            self for method chaining
        """
        if not events:
            raise ValueError("Cannot fit with empty events list")
        
        self.events = events
        
        # Compute hourly distribution
        hours = [e.hour for e in events]
        self._hourly_counts = dict(Counter(hours))
        
        # Fill missing hours with 0
        for h in range(24):
            if h not in self._hourly_counts:
                self._hourly_counts[h] = 0
        
        # Compute daily distribution
        days = [e.day_of_week for e in events]
        self._daily_counts = dict(Counter(days))
        
        # Fill missing days with 0
        for d in range(7):
            if d not in self._daily_counts:
                self._daily_counts[d] = 0
        
        # Compute time period distribution
        periods = [e.time_period for e in events]
        self._period_counts = dict(Counter(periods))
        
        # Build heatmap matrix (24 hours x 7 days)
        self._heatmap = np.zeros((24, 7), dtype=int)
        for event in events:
            self._heatmap[event.hour, event.day_of_week] += 1
        
        self.is_fitted = True
        return self
    
    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self.is_fitted:
            raise ValueError("Analyzer not fitted. Call fit() first.")
    
    # ==================== Hour Analysis ====================
    
    def get_hourly_distribution(self) -> Dict[int, int]:
        """
        Get event count for each hour of the day.
        
        Returns:
            Dict mapping hour (0-23) to event count
        """
        self._check_fitted()
        return dict(sorted(self._hourly_counts.items()))
    
    def get_hourly_percentages(self) -> Dict[int, float]:
        """
        Get percentage of events for each hour.
        
        Returns:
            Dict mapping hour (0-23) to percentage (0-100)
        """
        self._check_fitted()
        total = sum(self._hourly_counts.values())
        if total == 0:
            return {h: 0.0 for h in range(24)}
        return {h: round(count / total * 100, 2) 
                for h, count in sorted(self._hourly_counts.items())}
    
    def get_peak_hours(self, top_n: int = 3) -> List[Tuple[int, int]]:
        """
        Get the hours with most incidents.
        
        Args:
            top_n: Number of top hours to return
            
        Returns:
            List of (hour, count) tuples, sorted by count descending
        """
        self._check_fitted()
        sorted_hours = sorted(self._hourly_counts.items(), 
                              key=lambda x: x[1], reverse=True)
        return sorted_hours[:top_n]
    
    def get_safest_hours(self, top_n: int = 3) -> List[Tuple[int, int]]:
        """
        Get the hours with fewest incidents.
        
        Args:
            top_n: Number of safest hours to return
            
        Returns:
            List of (hour, count) tuples, sorted by count ascending
        """
        self._check_fitted()
        sorted_hours = sorted(self._hourly_counts.items(), 
                              key=lambda x: x[1])
        return sorted_hours[:top_n]
    
    # ==================== Day Analysis ====================
    
    def get_daily_distribution(self) -> Dict[str, int]:
        """
        Get event count for each day of the week.
        
        Returns:
            Dict mapping day name to event count
        """
        self._check_fitted()
        return {self.DAY_NAMES[d]: count 
                for d, count in sorted(self._daily_counts.items())}
    
    def get_daily_percentages(self) -> Dict[str, float]:
        """
        Get percentage of events for each day.
        
        Returns:
            Dict mapping day name to percentage (0-100)
        """
        self._check_fitted()
        total = sum(self._daily_counts.values())
        if total == 0:
            return {self.DAY_NAMES[d]: 0.0 for d in range(7)}
        return {self.DAY_NAMES[d]: round(count / total * 100, 2) 
                for d, count in sorted(self._daily_counts.items())}
    
    def get_peak_days(self, top_n: int = 2) -> List[Tuple[str, int]]:
        """
        Get the days with most incidents.
        
        Args:
            top_n: Number of top days to return
            
        Returns:
            List of (day_name, count) tuples
        """
        self._check_fitted()
        sorted_days = sorted(self._daily_counts.items(), 
                             key=lambda x: x[1], reverse=True)
        return [(self.DAY_NAMES[d], count) for d, count in sorted_days[:top_n]]
    
    def get_weekend_vs_weekday(self) -> Dict[str, Any]:
        """
        Compare weekend vs weekday incident rates.
        
        Returns:
            Dict with weekend/weekday stats and comparison
        """
        self._check_fitted()
        
        weekday_count = sum(self._daily_counts.get(d, 0) for d in range(5))
        weekend_count = sum(self._daily_counts.get(d, 0) for d in range(5, 7))
        
        total = weekday_count + weekend_count
        
        # Average per day
        weekday_avg = weekday_count / 5
        weekend_avg = weekend_count / 2
        
        return {
            "weekday_total": weekday_count,
            "weekend_total": weekend_count,
            "weekday_avg_per_day": round(weekday_avg, 2),
            "weekend_avg_per_day": round(weekend_avg, 2),
            "weekend_risk_ratio": round(weekend_avg / weekday_avg, 2) if weekday_avg > 0 else 0,
            "weekday_percentage": round(weekday_count / total * 100, 2) if total > 0 else 0,
            "weekend_percentage": round(weekend_count / total * 100, 2) if total > 0 else 0,
        }
    
    # ==================== Time Period Analysis ====================
    
    def get_period_distribution(self) -> Dict[str, int]:
        """
        Get event count for each time period.
        
        Returns:
            Dict mapping period name to event count
        """
        self._check_fitted()
        return {period: self._period_counts.get(period, 0) 
                for period in self.TIME_PERIODS}
    
    def get_period_percentages(self) -> Dict[str, float]:
        """
        Get percentage of events for each time period.
        
        Returns:
            Dict mapping period name to percentage (0-100)
        """
        self._check_fitted()
        total = sum(self._period_counts.values())
        if total == 0:
            return {period: 0.0 for period in self.TIME_PERIODS}
        return {period: round(self._period_counts.get(period, 0) / total * 100, 2)
                for period in self.TIME_PERIODS}
    
    def get_most_dangerous_period(self) -> Tuple[str, int, float]:
        """
        Get the time period with most incidents.
        
        Returns:
            Tuple of (period_name, count, percentage)
        """
        self._check_fitted()
        if not self._period_counts:
            return ("Unknown", 0, 0.0)
        
        max_period = max(self._period_counts.items(), key=lambda x: x[1])
        total = sum(self._period_counts.values())
        percentage = round(max_period[1] / total * 100, 2) if total > 0 else 0
        
        return (max_period[0], max_period[1], percentage)
    
    # ==================== Heatmap Analysis ====================
    
    def get_heatmap_data(self) -> Dict[str, Any]:
        """
        Get heatmap data for visualization (hour x day).
        
        Returns:
            Dict with matrix data and labels
        """
        self._check_fitted()
        
        return {
            "matrix": self._heatmap.tolist(),
            "x_labels": self.DAY_NAMES,
            "y_labels": [f"{h:02d}:00" for h in range(24)],
            "max_value": int(self._heatmap.max()),
            "min_value": int(self._heatmap.min()),
        }
    
    def get_hotspots(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top hour-day combinations with most incidents.
        
        Args:
            top_n: Number of hotspots to return
            
        Returns:
            List of hotspot dicts with hour, day, and count
        """
        self._check_fitted()
        
        # Flatten and get indices
        flat_indices = np.argsort(self._heatmap.flatten())[::-1][:top_n]
        
        hotspots = []
        for idx in flat_indices:
            hour = idx // 7
            day = idx % 7
            count = self._heatmap[hour, day]
            
            if count > 0:
                hotspots.append({
                    "hour": hour,
                    "hour_label": f"{hour:02d}:00",
                    "day": day,
                    "day_name": self.DAY_NAMES[day],
                    "count": int(count),
                })
        
        return hotspots
    
    # ==================== Summary ====================
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive time pattern summary.
        
        Returns:
            Dict with all key insights
        """
        self._check_fitted()
        
        peak_hours = self.get_peak_hours(3)
        peak_days = self.get_peak_days(2)
        dangerous_period = self.get_most_dangerous_period()
        weekend_stats = self.get_weekend_vs_weekday()
        hotspots = self.get_hotspots(3)
        
        return {
            "total_events": len(self.events),
            "analysis_period": {
                "start": min(e.timestamp for e in self.events).isoformat(),
                "end": max(e.timestamp for e in self.events).isoformat(),
            },
            "peak_hours": [
                {"hour": h, "label": f"{h:02d}:00", "count": c} 
                for h, c in peak_hours
            ],
            "peak_days": [
                {"day": d, "count": c} for d, c in peak_days
            ],
            "most_dangerous_period": {
                "period": dangerous_period[0],
                "count": dangerous_period[1],
                "percentage": dangerous_period[2],
            },
            "weekend_analysis": weekend_stats,
            "top_hotspots": hotspots,
            "hourly_distribution": self.get_hourly_percentages(),
            "daily_distribution": self.get_daily_percentages(),
            "period_distribution": self.get_period_percentages(),
        }
    
    def print_report(self) -> None:
        """Print a human-readable analysis report."""
        self._check_fitted()
        
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("📊 TIME PATTERN ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"\n📅 Analysis Period:")
        print(f"   {summary['analysis_period']['start'][:10]} to {summary['analysis_period']['end'][:10]}")
        print(f"   Total Events: {summary['total_events']}")
        
        print(f"\n⏰ Peak Hours (most incidents):")
        for item in summary['peak_hours']:
            print(f"   {item['label']}: {item['count']} events")
        
        print(f"\n📆 Peak Days:")
        for item in summary['peak_days']:
            print(f"   {item['day']}: {item['count']} events")
        
        period = summary['most_dangerous_period']
        print(f"\n🌙 Most Dangerous Time Period:")
        print(f"   {period['period']}: {period['count']} events ({period['percentage']}%)")
        
        weekend = summary['weekend_analysis']
        print(f"\n📊 Weekend vs Weekday:")
        print(f"   Weekday avg: {weekend['weekday_avg_per_day']} events/day")
        print(f"   Weekend avg: {weekend['weekend_avg_per_day']} events/day")
        print(f"   Weekend risk ratio: {weekend['weekend_risk_ratio']}x")
        
        print(f"\n🔥 Top Hotspots (hour + day combinations):")
        for hs in summary['top_hotspots']:
            print(f"   {hs['day_name']} {hs['hour_label']}: {hs['count']} events")
        
        print("\n" + "=" * 60)


# Quick test
if __name__ == "__main__":
    from insights.data import ViolenceEventGenerator
    
    print("Testing TimePatternAnalyzer...")
    
    # Generate mock data
    generator = ViolenceEventGenerator(seed=42)
    events = generator.generate(n_events=300)
    
    # Analyze
    analyzer = TimePatternAnalyzer()
    analyzer.fit(events)
    
    # Print report
    analyzer.print_report()
    
    # Get summary for API
    summary = analyzer.get_summary()
    print(f"\n✅ Summary keys: {list(summary.keys())}")
