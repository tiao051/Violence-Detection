"""
Violence Insights Engine

Main entry point for violence event analytics.
Combines all analyzers into a single, easy-to-use interface.

Usage:
    from insights import InsightsEngine
    
    engine = InsightsEngine()
    engine.load_mock_data(n_events=300)  # or engine.load_events(events)
    
    # Get full report
    report = engine.get_full_report()
    
    # Or access individual analyzers
    time_insights = engine.time_analyzer.get_summary()
    location_insights = engine.location_analyzer.get_summary()
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from .data import ViolenceEventGenerator, ViolenceEvent
from .models import TimePatternAnalyzer, LocationAnalyzer


class InsightsEngine:
    """
    Main interface for violence event insights.
    
    Provides a unified API to analyze violence detection events
    and extract actionable insights about patterns and risks.
    
    Features:
    - Load data from mock generator or external source
    - Time pattern analysis (when violence occurs)
    - Location analysis (where violence occurs)  
    - Combined summary report
    - Export to JSON/dict for API responses
    
    Example:
        >>> engine = InsightsEngine()
        >>> engine.load_mock_data(n_events=200, days=30)
        >>> 
        >>> # Quick summary
        >>> print(engine.get_quick_summary())
        >>> 
        >>> # Full report
        >>> report = engine.get_full_report()
        >>> print(json.dumps(report, indent=2))
    """
    
    def __init__(self):
        """Initialize the engine."""
        self.events: List[ViolenceEvent] = []
        self.is_loaded: bool = False
        
        # Analyzers
        self.time_analyzer = TimePatternAnalyzer()
        self.location_analyzer = LocationAnalyzer()
        
        # Metadata
        self.data_source: str = "none"
        self.load_time: Optional[datetime] = None
    
    def load_events(self, events: List[ViolenceEvent]) -> "InsightsEngine":
        """
        Load events from external source.
        
        Args:
            events: List of ViolenceEvent instances
            
        Returns:
            self for method chaining
        """
        if not events:
            raise ValueError("Cannot load empty events list")
        
        self.events = events
        self.data_source = "external"
        self.load_time = datetime.now()
        
        # Fit analyzers
        self.time_analyzer.fit(events)
        self.location_analyzer.fit(events)
        
        self.is_loaded = True
        return self
    
    def load_mock_data(
        self,
        n_events: int = 200,
        days: int = 30,
        seed: Optional[int] = 42,
    ) -> "InsightsEngine":
        """
        Load mock data for testing/demo.
        
        Args:
            n_events: Number of events to generate
            days: Number of days to span
            seed: Random seed for reproducibility
            
        Returns:
            self for method chaining
        """
        generator = ViolenceEventGenerator(seed=seed)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        events = generator.generate(
            n_events=n_events,
            start_date=start_date,
            end_date=end_date,
        )
        
        self.data_source = f"mock (n={n_events}, days={days})"
        return self.load_events(events)
    
    def load_from_dicts(self, data: List[Dict]) -> "InsightsEngine":
        """
        Load events from list of dictionaries (e.g., from Firestore).
        
        Args:
            data: List of event dictionaries
            
        Returns:
            self for method chaining
        """
        events = [ViolenceEvent.from_dict(d) for d in data]
        self.data_source = "firestore"
        return self.load_events(events)
    
    def _check_loaded(self) -> None:
        """Raise error if data not loaded."""
        if not self.is_loaded:
            raise ValueError("No data loaded. Call load_events() or load_mock_data() first.")
    
    # ==================== Reports ====================
    
    def get_quick_summary(self) -> Dict[str, Any]:
        """
        Get a quick summary with key highlights.
        
        Returns:
            Dict with key metrics and insights
        """
        self._check_loaded()
        
        time_summary = self.time_analyzer.get_summary()
        location_summary = self.location_analyzer.get_summary()
        
        peak_hours = time_summary['peak_hours']
        peak_days = time_summary['peak_days']
        dangerous_period = time_summary['most_dangerous_period']
        weekend = time_summary['weekend_analysis']
        hotspots = location_summary['hotspot_cameras']
        risk_ranking = location_summary['risk_ranking']
        
        return {
            "total_events": len(self.events),
            "analysis_period": time_summary['analysis_period'],
            "data_source": self.data_source,
            
            # Time highlights
            "peak_hour": peak_hours[0] if peak_hours else None,
            "peak_day": peak_days[0] if peak_days else None,
            "dangerous_period": dangerous_period['period'],
            "weekend_risk_ratio": weekend['weekend_risk_ratio'],
            
            # Location highlights
            "hotspot_camera": hotspots[0] if hotspots else None,
            "highest_risk_camera": risk_ranking[0] if risk_ranking else None,
            
            # Quick stats
            "avg_confidence": location_summary['overall_confidence']['mean'],
        }
    
    def get_full_report(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis report.
        
        Returns:
            Dict containing all insights from all analyzers
        """
        self._check_loaded()
        
        return {
            "metadata": {
                "total_events": len(self.events),
                "data_source": self.data_source,
                "generated_at": datetime.now().isoformat(),
                "analysis_period": {
                    "start": min(e.timestamp for e in self.events).isoformat(),
                    "end": max(e.timestamp for e in self.events).isoformat(),
                },
            },
            "time_patterns": self.time_analyzer.get_summary(),
            "location_patterns": self.location_analyzer.get_summary(),
            "quick_summary": self.get_quick_summary(),
        }
    
    def get_actionable_insights(self) -> List[Dict[str, Any]]:
        """
        Get list of actionable insights/recommendations.
        
        Returns:
            List of insight dicts with type, message, priority
        """
        self._check_loaded()
        
        insights = []
        
        # Time-based insights
        time_summary = self.time_analyzer.get_summary()
        
        peak_hours = time_summary['peak_hours']
        if peak_hours:
            top_hour = peak_hours[0]
            insights.append({
                "type": "time",
                "category": "peak_hour",
                "priority": "high",
                "message": f"Most incidents occur at {top_hour['label']} ({top_hour['count']} events). Consider increased monitoring during this hour.",
                "data": top_hour,
            })
        
        weekend = time_summary['weekend_analysis']
        if weekend['weekend_risk_ratio'] > 1.3:
            insights.append({
                "type": "time",
                "category": "weekend_risk",
                "priority": "medium",
                "message": f"Weekends have {weekend['weekend_risk_ratio']:.1f}x higher incident rate than weekdays. Consider additional weekend staffing.",
                "data": weekend,
            })
        
        dangerous = time_summary['most_dangerous_period']
        if dangerous['percentage'] > 35:
            insights.append({
                "type": "time",
                "category": "dangerous_period",
                "priority": "high",
                "message": f"{dangerous['period']} is particularly dangerous with {dangerous['percentage']:.1f}% of all incidents.",
                "data": dangerous,
            })
        
        # Location-based insights
        location_summary = self.location_analyzer.get_summary()
        
        hotspots = location_summary['hotspot_cameras']
        if hotspots:
            top_camera = hotspots[0]
            if top_camera['percentage'] > 25:
                insights.append({
                    "type": "location",
                    "category": "hotspot",
                    "priority": "high",
                    "message": f"{top_camera['camera_name']} is a major hotspot with {top_camera['percentage']:.1f}% of all incidents. Consider camera repositioning or additional coverage.",
                    "data": top_camera,
                })
        
        risk_ranking = location_summary['risk_ranking']
        if risk_ranking:
            high_risk = [r for r in risk_ranking if r['high_severity_pct'] > 30]
            for cam in high_risk[:2]:
                insights.append({
                    "type": "location",
                    "category": "high_severity",
                    "priority": "high",
                    "message": f"{cam['camera_name']} has {cam['high_severity_pct']:.1f}% high-severity incidents. This area may need urgent attention.",
                    "data": cam,
                })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda x: priority_order.get(x['priority'], 99))
        
        return insights
    
    # ==================== Export ====================
    
    def to_json(self, filepath: str) -> None:
        """
        Export full report to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        report = self.get_full_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report exported to {filepath}")
    
    def print_summary(self) -> None:
        """Print a human-readable summary."""
        self._check_loaded()
        
        summary = self.get_quick_summary()
        
        print("\n" + "=" * 60)
        print("  VIOLENCE INSIGHTS SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal Events: {summary['total_events']}")
        print(f"Data Source: {summary['data_source']}")
        print(f"Period: {summary['analysis_period']['start'][:10]} to {summary['analysis_period']['end'][:10]}")
        
        print("\n[TIME PATTERNS]")
        if summary['peak_hour']:
            print(f"  Peak Hour: {summary['peak_hour']['label']} ({summary['peak_hour']['count']} events)")
        if summary['peak_day']:
            print(f"  Peak Day: {summary['peak_day']['day']} ({summary['peak_day']['count']} events)")
        print(f"  Most Dangerous Period: {summary['dangerous_period']}")
        print(f"  Weekend Risk: {summary['weekend_risk_ratio']:.2f}x higher")
        
        print("\n[LOCATION PATTERNS]")
        if summary['hotspot_camera']:
            cam = summary['hotspot_camera']
            print(f"  Hotspot Camera: {cam['camera_name']} ({cam['event_count']} events, {cam['percentage']}%)")
        if summary['highest_risk_camera']:
            cam = summary['highest_risk_camera']
            print(f"  Highest Risk: {cam['camera_name']} (score: {cam['risk_score']:.3f})")
        
        print(f"\n[DETECTION QUALITY]")
        print(f"  Average Confidence: {summary['avg_confidence']:.3f}")
        
        # Actionable insights
        insights = self.get_actionable_insights()
        if insights:
            print("\n[ACTIONABLE INSIGHTS]")
            for i, insight in enumerate(insights[:5], 1):
                priority_icon = "[!]" if insight['priority'] == 'high' else "[*]"
                print(f"  {priority_icon} {insight['message']}")
        
        print("\n" + "=" * 60)


# Quick test
if __name__ == "__main__":
    print("Testing InsightsEngine...")
    
    # Create engine and load mock data
    engine = InsightsEngine()
    engine.load_mock_data(n_events=300, days=60, seed=42)
    
    # Print summary
    engine.print_summary()
    
    # Show actionable insights
    print("\nActionable Insights:")
    for insight in engine.get_actionable_insights():
        print(f"  [{insight['priority'].upper()}] {insight['message']}")
    
    print("\n[OK] InsightsEngine test complete!")
