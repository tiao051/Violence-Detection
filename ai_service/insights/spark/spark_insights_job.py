"""
Spark Insights Job - Distributed Analytics Processing

REQUIRES PySpark and HDFS - No fallback to pandas.

Reads violence detection events from HDFS and produces actionable insights:
- Trend analysis (week vs week, month vs month)
- Anomaly detection (Z-score based)
- Camera performance metrics
- Patrol schedule recommendations

Runs on-demand or weekly via API trigger.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

logger = logging.getLogger(__name__)


@dataclass
class TrendResult:
    """Result of trend comparison"""
    camera_id: str
    camera_name: str
    current_count: int
    previous_count: int
    change_pct: float
    trend: str  # "up", "down", "stable"
    is_significant: bool  # >20% change


@dataclass 
class AnomalyResult:
    """Anomaly detection result for a camera"""
    camera_id: str
    camera_name: str
    current_count: int
    expected_count: float
    z_score: float
    is_anomaly: bool
    severity: str  # "critical", "warning", "normal"


@dataclass
class PatrolSlot:
    """Recommended patrol time slot"""
    camera_id: str
    camera_name: str
    start_hour: int
    end_hour: int
    priority: str  # "high", "medium", "low"
    reason: str
    expected_incidents: float


# Camera name mapping
CAMERA_NAMES = {
    "cam1": "Luy Ban Bich Street",
    "cam2": "Au Co Junction",
    "cam3": "Tan Ky Tan Quy Street",
    "cam4": "Tan Phu Market",
    "cam5": "Dam Sen Park",
}


class SparkInsightsJob:
    """
    PySpark job for processing HDFS event data.
    
    REQUIRES:
    - PySpark installed
    - HDFS accessible at hdfs://namenode:9000
    
    Will raise error if Spark or HDFS is not available.
    """
    
    def __init__(
        self,
        hdfs_namenode: str = None,
        hdfs_path: str = "/analytics/raw",
        spark_master: str = None
    ):
        self.hdfs_namenode = hdfs_namenode or os.getenv("HDFS_NAMENODE", "namenode:9000")
        self.hdfs_path = hdfs_path
        self.spark_master = spark_master or os.getenv("SPARK_MASTER", "local[*]")
        
        self.spark: Optional[SparkSession] = None
        self.results: Dict[str, Any] = {}
        
    def _init_spark(self) -> SparkSession:
        """
        Initialize Spark session.
        Raises error if Spark is not available.
        """
        logger.info(f"Initializing Spark session (master: {self.spark_master})")
        
        self.spark = SparkSession.builder \
            .appName("ViolenceInsightsJob") \
            .master(self.spark_master) \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.hadoop.fs.defaultFS", f"hdfs://{self.hdfs_namenode}") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark session created successfully")
        
        return self.spark
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full insights job using Spark.
        
        Returns:
            Dict with trends, anomalies, and patrol schedule
            
        Raises:
            RuntimeError: If Spark or HDFS is not available
        """
        logger.info("=" * 60)
        logger.info("SPARK INSIGHTS JOB STARTED")
        logger.info(f"  HDFS: hdfs://{self.hdfs_namenode}{self.hdfs_path}")
        logger.info(f"  Spark Master: {self.spark_master}")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Initialize Spark (will raise if not available)
            self._init_spark()
            
            # Load data from HDFS (will raise if HDFS not accessible)
            df = self._load_from_hdfs()
            
            total_count = df.count()
            if total_count == 0:
                raise RuntimeError("No data found in HDFS")
            
            logger.info(f"Loaded {total_count} events from HDFS")
            
            # Run distributed analyses
            weekly_trends = self._compute_weekly_trends_spark(df)
            monthly_trends = self._compute_monthly_trends_spark(df)
            anomalies = self._detect_anomalies_spark(df)
            patrol_schedule = self._generate_patrol_schedule_spark(df)
            camera_stats = self._compute_camera_stats_spark(df)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            self.results = {
                "success": True,
                "computed_at": datetime.now().isoformat(),
                "processing_time_seconds": round(elapsed, 2),
                "engine": "Apache Spark",
                "hdfs_path": f"hdfs://{self.hdfs_namenode}{self.hdfs_path}",
                "total_events": total_count,
                "weekly_trends": weekly_trends,
                "monthly_trends": monthly_trends,
                "anomalies": anomalies,
                "patrol_schedule": patrol_schedule,
                "camera_stats": camera_stats,
            }
            
            logger.info(f"Spark insights job completed in {elapsed:.1f}s")
            logger.info(f"  Weekly trends: {len(weekly_trends)} cameras")
            logger.info(f"  Monthly trends: {len(monthly_trends)} cameras")
            logger.info(f"  Anomalies detected: {len([a for a in anomalies if a['is_anomaly']])}")
            logger.info(f"  Patrol slots: {len(patrol_schedule)}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Spark insights job failed: {e}", exc_info=True)
            raise RuntimeError(f"Spark insights job failed: {e}")
        finally:
            if self.spark:
                self.spark.stop()
                self.spark = None
    
    def _load_from_hdfs(self):
        """
        Load event data from HDFS using Spark.
        
        Raises:
            RuntimeError: If HDFS is not accessible or no data found
        """
        hdfs_full_path = f"hdfs://{self.hdfs_namenode}{self.hdfs_path}/*.csv"
        
        logger.info(f"Reading from HDFS: {hdfs_full_path}")
        
        try:
            df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(hdfs_full_path)
            
            # Add datetime column from timestamp
            if "timestamp" in df.columns:
                df = df.withColumn("datetime", F.from_unixtime(F.col("timestamp") / 1000))
            
            # Add time-based columns
            df = df.withColumn("hour", F.hour("datetime"))
            df = df.withColumn("day_of_week", F.dayofweek("datetime"))
            df = df.withColumn("week", F.weekofyear("datetime"))
            df = df.withColumn("month", F.month("datetime"))
            df = df.withColumn("date", F.to_date("datetime"))
            
            # Filter to violence events only
            if "label" in df.columns:
                df = df.filter(F.col("label") == "violence")
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to read from HDFS: {e}")
    
    def _compute_weekly_trends_spark(self, df) -> List[Dict[str, Any]]:
        """Compare this week vs last week per camera using Spark."""
        now = datetime.now()
        current_week_start = now - timedelta(days=now.weekday())
        last_week_start = current_week_start - timedelta(days=7)
        
        # Current week counts
        current_week = df.filter(F.col("datetime") >= current_week_start.strftime("%Y-%m-%d"))
        current_counts = current_week.groupBy("camera_id").count().collect()
        current_dict = {row["camera_id"]: row["count"] for row in current_counts}
        
        # Last week counts
        last_week = df.filter(
            (F.col("datetime") >= last_week_start.strftime("%Y-%m-%d")) &
            (F.col("datetime") < current_week_start.strftime("%Y-%m-%d"))
        )
        last_counts = last_week.groupBy("camera_id").count().collect()
        last_dict = {row["camera_id"]: row["count"] for row in last_counts}
        
        # Combine results
        all_cameras = set(current_dict.keys()) | set(last_dict.keys())
        
        results = []
        for cam_id in all_cameras:
            curr = current_dict.get(cam_id, 0)
            prev = last_dict.get(cam_id, 0)
            
            if prev > 0:
                change = ((curr - prev) / prev) * 100
            elif curr > 0:
                change = 100
            else:
                change = 0
            
            trend = "up" if change > 10 else "down" if change < -10 else "stable"
            
            results.append({
                "camera_id": cam_id,
                "camera_name": CAMERA_NAMES.get(cam_id, cam_id),
                "current_count": curr,
                "previous_count": prev,
                "change_pct": round(change, 1),
                "trend": trend,
                "is_significant": abs(change) > 20,
            })
        
        results.sort(key=lambda x: x["change_pct"], reverse=True)
        return results
    
    def _compute_monthly_trends_spark(self, df) -> List[Dict[str, Any]]:
        """Compare this month vs last month per camera using Spark."""
        now = datetime.now()
        current_month_start = now.replace(day=1)
        last_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
        
        # Current month
        current_month = df.filter(F.col("datetime") >= current_month_start.strftime("%Y-%m-%d"))
        current_counts = current_month.groupBy("camera_id").count().collect()
        current_dict = {row["camera_id"]: row["count"] for row in current_counts}
        
        # Last month
        last_month = df.filter(
            (F.col("datetime") >= last_month_start.strftime("%Y-%m-%d")) &
            (F.col("datetime") < current_month_start.strftime("%Y-%m-%d"))
        )
        last_counts = last_month.groupBy("camera_id").count().collect()
        last_dict = {row["camera_id"]: row["count"] for row in last_counts}
        
        all_cameras = set(current_dict.keys()) | set(last_dict.keys())
        
        results = []
        for cam_id in all_cameras:
            curr = current_dict.get(cam_id, 0)
            prev = last_dict.get(cam_id, 0)
            
            if prev > 0:
                change = ((curr - prev) / prev) * 100
            elif curr > 0:
                change = 100
            else:
                change = 0
            
            trend = "up" if change > 10 else "down" if change < -10 else "stable"
            
            results.append({
                "camera_id": cam_id,
                "camera_name": CAMERA_NAMES.get(cam_id, cam_id),
                "current_count": curr,
                "previous_count": prev,
                "change_pct": round(change, 1),
                "trend": trend,
                "is_significant": abs(change) > 20,
            })
        
        results.sort(key=lambda x: x["change_pct"], reverse=True)
        return results
    
    def _detect_anomalies_spark(self, df) -> List[Dict[str, Any]]:
        """Detect cameras with unusual activity using Z-score via Spark."""
        # Aggregate counts per camera
        camera_counts = df.groupBy("camera_id").count()
        
        # Compute mean and stddev
        stats = camera_counts.agg(
            F.mean("count").alias("mean"),
            F.stddev("count").alias("std")
        ).collect()[0]
        
        mean_val = stats["mean"] or 0
        std_val = stats["std"] or 1
        
        if std_val == 0:
            std_val = 1
        
        # Get all camera counts
        counts = camera_counts.collect()
        
        results = []
        for row in counts:
            cam_id = row["camera_id"]
            count = row["count"]
            z = (count - mean_val) / std_val
            
            if z >= 2.5:
                severity = "critical"
                is_anomaly = True
            elif z >= 1.5:
                severity = "warning"
                is_anomaly = True
            else:
                severity = "normal"
                is_anomaly = False
            
            results.append({
                "camera_id": cam_id,
                "camera_name": CAMERA_NAMES.get(cam_id, cam_id),
                "current_count": count,
                "expected_count": round(mean_val, 1),
                "z_score": round(z, 2),
                "is_anomaly": is_anomaly,
                "severity": severity,
            })
        
        results.sort(key=lambda x: x["z_score"], reverse=True)
        return results
    
    def _generate_patrol_schedule_spark(self, df) -> List[Dict[str, Any]]:
        """Generate recommended patrol times using Spark aggregation."""
        # Group by camera and hour
        peak_times = df.groupBy("camera_id", "hour").count().collect()
        
        # Build dict: camera_id -> [(hour, count), ...]
        camera_hours = {}
        for row in peak_times:
            cam_id = row["camera_id"]
            if cam_id not in camera_hours:
                camera_hours[cam_id] = []
            camera_hours[cam_id].append((row["hour"], row["count"]))
        
        results = []
        for cam_id, hours in camera_hours.items():
            # Get top 3 hours
            top_hours = sorted(hours, key=lambda x: x[1], reverse=True)[:3]
            
            for hour, count in top_hours:
                if count >= 10:
                    priority = "high"
                elif count >= 5:
                    priority = "medium"
                else:
                    priority = "low"
                
                results.append({
                    "camera_id": cam_id,
                    "camera_name": CAMERA_NAMES.get(cam_id, cam_id),
                    "start_hour": hour,
                    "end_hour": (hour + 2) % 24,
                    "priority": priority,
                    "reason": f"Peak activity ({count} incidents at {hour:02d}:00)",
                    "expected_incidents": round(count / 7, 1),
                })
        
        priority_order = {"high": 0, "medium": 1, "low": 2}
        results.sort(key=lambda x: priority_order[x["priority"]])
        
        return results
    
    def _compute_camera_stats_spark(self, df) -> Dict[str, Any]:
        """Compute overall statistics per camera using Spark."""
        # Aggregate stats
        stats_df = df.groupBy("camera_id").agg(
            F.count("*").alias("total_events"),
            F.countDistinct("date").alias("num_days"),
            F.mean("confidence").alias("avg_confidence")
        ).collect()
        
        # Get mode for hour and day per camera
        hour_mode = df.groupBy("camera_id", "hour").count() \
            .orderBy(F.desc("count")).collect()
        
        day_mode = df.groupBy("camera_id", "day_of_week").count() \
            .orderBy(F.desc("count")).collect()
        
        # Build peak hour/day lookup
        peak_hours = {}
        peak_days = {}
        for row in hour_mode:
            if row["camera_id"] not in peak_hours:
                peak_hours[row["camera_id"]] = row["hour"]
        for row in day_mode:
            if row["camera_id"] not in peak_days:
                peak_days[row["camera_id"]] = row["day_of_week"]
        
        stats = {}
        for row in stats_df:
            cam_id = row["camera_id"]
            num_days = row["num_days"] or 1
            
            stats[cam_id] = {
                "camera_name": CAMERA_NAMES.get(cam_id, cam_id),
                "total_events": row["total_events"],
                "avg_daily": round(row["total_events"] / num_days, 1),
                "peak_hour": peak_hours.get(cam_id, 0),
                "peak_day": peak_days.get(cam_id, 0),
                "avg_confidence": round(row["avg_confidence"] or 0, 3),
            }
        
        return stats


# Singleton instance for API use
_insights_job: Optional[SparkInsightsJob] = None
_cached_results: Optional[Dict[str, Any]] = None
_cache_time: Optional[datetime] = None
CACHE_DURATION_HOURS = 24  # Results valid for 24 hours


def get_spark_insights(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get Spark insights, using cache if available.
    
    REQUIRES: PySpark and HDFS available.
    
    Args:
        force_refresh: If True, ignore cache and run job again
        
    Returns:
        Dict with trends, anomalies, and patrol schedule
        
    Raises:
        RuntimeError: If Spark/HDFS is not available
    """
    global _insights_job, _cached_results, _cache_time
    
    # Check cache
    if not force_refresh and _cached_results and _cache_time:
        age = (datetime.now() - _cache_time).total_seconds() / 3600
        if age < CACHE_DURATION_HOURS:
            logger.info(f"Returning cached Spark insights (age: {age:.1f}h)")
            return _cached_results
    
    # Run Spark job (will raise if Spark/HDFS not available)
    if _insights_job is None:
        _insights_job = SparkInsightsJob()
    
    _cached_results = _insights_job.run()
    _cache_time = datetime.now()
    
    return _cached_results
