"""Time and date utilities for insight analysis."""

from typing import List

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PERIODS = ["Morning", "Afternoon", "Evening", "Night"]

# Period boundaries
PERIOD_BOUNDARIES = {
    "Morning": (6, 12),     
    "Afternoon": (12, 18), 
    "Evening": (18, 22),    
    "Night": (22, 6),      
}

PERIOD_MAP = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}


def get_time_period(hour: int) -> str:
    """Convert hour to time period."""
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 22:
        return "Evening"
    else:
        return "Night"


def get_hour_bucket(hour: int) -> str:
    """Get hour bucket label for categorization (for association analysis)."""
    if hour < 6:
        return "hour_late_night"
    elif hour < 12:
        return "hour_morning"
    elif hour < 18:
        return "hour_afternoon"
    else:
        return "hour_evening"


def get_time_description(hour: int) -> str:
    """Get human-readable time description."""
    if hour < 6:
        return "late night"
    elif hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "evening/night"


def get_day_names() -> List[str]:
    """Get list of day names."""
    return DAYS.copy()


def get_period_names() -> List[str]:
    """Get list of period names."""
    return PERIODS.copy()


def is_weekend(day_of_week: int) -> bool:
    """Check if day is weekend (Saturday or Sunday)."""
    return day_of_week >= 5


def get_day_name(day_of_week: int) -> str:
    """Get day name from index (0=Monday, 6=Sunday)."""
    return DAYS[day_of_week % 7]
