"""
Spark Insights Job - Distributed Analytics with ML Models

REQUIRES PySpark and HDFS - No fallback.

Uses Apache Spark to:
1. Read violence events from HDFS data lake
2. Preprocess and aggregate data
3. Train ML models (K-means, FP-Growth, Random Forest)
4. Generate actionable insights

ML Algorithms Used:
- K-means Clustering: Discover event patterns
- FP-Growth: Association rules between conditions
- Random Forest: Risk level prediction
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


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
    PySpark job for processing HDFS event data with ML models.
    
    Pipeline:
    1. Spark reads CSV from HDFS
    2. Convert to ViolenceEvent format
    3. Train InsightsModel (K-means + FP-Growth + Random Forest)
    4. Generate insights and predictions
    
    REQUIRES:
    - PySpark installed
    - HDFS accessible
    - InsightsModel from ai_service
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
        self.model = None  # InsightsModel instance
        self.results: Dict[str, Any] = {}
        
    def _init_spark(self) -> SparkSession:
        """Initialize Spark session."""
        logger.info(f"Initializing Spark session (master: {self.spark_master})")
        
        self.spark = SparkSession.builder \
            .appName("ViolenceInsightsMLJob") \
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
        Run the full insights job with ML models.
        
        Returns:
            Dict with ML-based insights (patterns, rules, predictions)
        """
        logger.info("=" * 60)
        logger.info("SPARK ML INSIGHTS JOB STARTED")
        logger.info(f"  HDFS: hdfs://{self.hdfs_namenode}{self.hdfs_path}")
        logger.info(f"  Algorithms: K-means, FP-Growth, Random Forest")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Initialize Spark
            self._init_spark()
            
            # Step 2: Load data from HDFS using Spark
            spark_df = self._load_from_hdfs()
            total_count = spark_df.count()
            
            if total_count == 0:
                raise RuntimeError("No data found in HDFS")
            
            logger.info(f"Loaded {total_count} events from HDFS")
            
            # Step 3: Convert Spark DataFrame to ViolenceEvent objects
            events = self._convert_to_events(spark_df)
            logger.info(f"Converted {len(events)} events to ViolenceEvent format")
            
            # Step 4: Train ML models using InsightsModel
            ml_results = self._train_ml_models(events)
            
            # Step 5: Compute Spark-based aggregations (trends, anomalies)
            trends = self._compute_trends(spark_df)
            anomalies = self._detect_anomalies(spark_df)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            self.results = {
                "success": True,
                "computed_at": datetime.now().isoformat(),
                "processing_time_seconds": round(elapsed, 2),
                "engine": "Apache Spark + ML",
                "algorithms": ["K-means Clustering", "FP-Growth", "Random Forest"],
                "hdfs_path": f"hdfs://{self.hdfs_namenode}{self.hdfs_path}",
                "total_events": total_count,
                
                # ML Model Results
                "patterns": ml_results.get("patterns", []),
                "rules": ml_results.get("rules", []),
                "high_risk_conditions": ml_results.get("high_risk", []),
                "prediction_accuracy": ml_results.get("accuracy", 0),
                
                # Spark Aggregation Results
                "weekly_trends": trends.get("weekly", []),
                "monthly_trends": trends.get("monthly", []),
                "anomalies": anomalies,
            }
            
            logger.info(f"Spark ML insights job completed in {elapsed:.1f}s")
            logger.info(f"  Patterns (K-means): {len(self.results.get('patterns', []))}")
            logger.info(f"  Rules (FP-Growth): {len(self.results.get('rules', []))}")
            logger.info(f"  Prediction Accuracy: {self.results.get('prediction_accuracy', 0):.1%}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Spark ML insights job failed: {e}", exc_info=True)
            raise RuntimeError(f"Spark ML insights job failed: {e}")
        finally:
            if self.spark:
                self.spark.stop()
                self.spark = None
    
    def _load_from_hdfs(self):
        """Load event data from HDFS using Spark."""
        hdfs_full_path = f"hdfs://{self.hdfs_namenode}{self.hdfs_path}/*.csv"
        
        logger.info(f"Reading from HDFS: {hdfs_full_path}")
        
        try:
            df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(hdfs_full_path)
            
            # Filter to violence events only
            if "label" in df.columns:
                df = df.filter(F.col("label") == "violence")
            
            # Add datetime column
            if "timestamp" in df.columns:
                df = df.withColumn("datetime", F.from_unixtime(F.col("timestamp") / 1000))
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to read from HDFS: {e}")
    
    def _convert_to_events(self, spark_df) -> List:
        """Convert Spark DataFrame to list of ViolenceEvent objects."""
        from insights.core.schema import ViolenceEvent
        
        # Collect to driver (for ML model training)
        rows = spark_df.collect()
        
        events = []
        for row in rows:
            try:
                # Parse timestamp
                ts = row.get("timestamp")
                if ts:
                    dt = datetime.fromtimestamp(ts / 1000)
                else:
                    dt = datetime.now()
                
                # Get camera name
                camera_id = row.get("camera_id", "unknown")
                camera_name = row.get("camera_name", CAMERA_NAMES.get(camera_id, camera_id))
                
                # Get confidence
                conf = row.get("confidence", 0.8)
                if isinstance(conf, str):
                    conf = float(conf)
                
                event = ViolenceEvent(
                    camera_id=camera_id,
                    camera_name=camera_name,
                    timestamp=dt,
                    confidence=conf
                )
                events.append(event)
                
            except Exception as e:
                logger.warning(f"Failed to convert row: {e}")
                continue
        
        return events
    
    def _train_ml_models(self, events: List) -> Dict[str, Any]:
        """
        Analyze events using InsightsModel.
        
        Strategy:
        1. Load pre-trained model if available (to keep trained Random Forest).
        2. Re-fit K-means and FP-Growth on NEW HDFS data (to discover current patterns).
        3. Do NOT re-train Random Forest (use existing logic).
        """
        from insights import InsightsModel
        import pickle
        
        if len(events) < 50:
            logger.warning(f"Not enough events ({len(events)}) for analysis, need 50+")
            return {"patterns": [], "rules": [], "high_risk": [], "accuracy": 0}
        
        # 1. Try to load existing model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trained_model.pkl')
        try:
            if os.path.exists(model_path):
                logger.info(f"Loading pre-trained model from {model_path}")
                self.model = InsightsModel.load(model_path)
            else:
                logger.warning("No pre-trained model found, initializing new one")
                self.model = InsightsModel()
        except Exception as e:
            logger.error(f"Error loading model: {e}, initializing new one")
            self.model = InsightsModel()
        
        logger.info("Analyzing HDFS data...")
        logger.info("  - Updating Cluster Analysis (K-means) on new data")
        logger.info("  - Updating Association Rules (FP-Growth) on new data")
        
        # 2. Re-fit Unsupervised Models on CURRENT HDFS Data
        # We want patterns existing in the HDFS data, not old training data
        self.model.cluster_model.fit(events)
        self.model.association_model.fit(events)
        
        # 3. Random Forest: Keep pre-trained state (do not fit)
        # We assume the pre-trained RF is better or verified (ground truth trained)
        if not self.model.is_fitted:
             # Fallback if brand new model
             logger.warning("Model was not fitted, training RF on HDFS data as fallback")
             self.model.prediction_model.fit(events)
        
        self.model.events = events # Update event context
        self.model.is_fitted = True
        
        # Get results
        patterns = self.model.get_patterns()
        rules = self.model.get_rules(top_n=10)
        high_risk = self.model.get_high_risk_conditions(top_n=10)
        
        logger.info(f"Analysis complete:")
        logger.info(f"  - K-means found {len(patterns)} clusters in HDFS data")
        logger.info(f"  - FP-Growth found {len(rules)} rules in HDFS data")
        logger.info(f"  - Risk Predictor ready (Accuracy: {self.model.prediction_model.accuracy:.1%})")
        
        return {
            "patterns": patterns,
            "rules": rules,
            "high_risk": high_risk,
            "accuracy": self.model.prediction_model.accuracy,
        }
    
    def _compute_trends(self, df) -> Dict[str, List]:
        """Compute weekly and monthly trends using Spark."""
        now = datetime.now()
        
        # Weekly trends
        current_week_start = now - timedelta(days=now.weekday())
        last_week_start = current_week_start - timedelta(days=7)
        
        df = df.withColumn("datetime", F.from_unixtime(F.col("timestamp") / 1000))
        
        current_week = df.filter(F.col("datetime") >= current_week_start.strftime("%Y-%m-%d"))
        last_week = df.filter(
            (F.col("datetime") >= last_week_start.strftime("%Y-%m-%d")) &
            (F.col("datetime") < current_week_start.strftime("%Y-%m-%d"))
        )
        
        current_counts = {r["camera_id"]: r["count"] for r in current_week.groupBy("camera_id").count().collect()}
        last_counts = {r["camera_id"]: r["count"] for r in last_week.groupBy("camera_id").count().collect()}
        
        weekly = []
        for cam_id in set(current_counts.keys()) | set(last_counts.keys()):
            curr = current_counts.get(cam_id, 0)
            prev = last_counts.get(cam_id, 0)
            change = ((curr - prev) / prev * 100) if prev > 0 else (100 if curr > 0 else 0)
            
            weekly.append({
                "camera_id": cam_id,
                "camera_name": CAMERA_NAMES.get(cam_id, cam_id),
                "current_count": curr,
                "previous_count": prev,
                "change_pct": round(change, 1),
                "trend": "up" if change > 10 else "down" if change < -10 else "stable",
            })
        
        weekly.sort(key=lambda x: x["change_pct"], reverse=True)
        
        # Monthly trends (similar logic)
        current_month_start = now.replace(day=1)
        last_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
        
        current_month = df.filter(F.col("datetime") >= current_month_start.strftime("%Y-%m-%d"))
        last_month = df.filter(
            (F.col("datetime") >= last_month_start.strftime("%Y-%m-%d")) &
            (F.col("datetime") < current_month_start.strftime("%Y-%m-%d"))
        )
        
        current_m = {r["camera_id"]: r["count"] for r in current_month.groupBy("camera_id").count().collect()}
        last_m = {r["camera_id"]: r["count"] for r in last_month.groupBy("camera_id").count().collect()}
        
        monthly = []
        for cam_id in set(current_m.keys()) | set(last_m.keys()):
            curr = current_m.get(cam_id, 0)
            prev = last_m.get(cam_id, 0)
            change = ((curr - prev) / prev * 100) if prev > 0 else (100 if curr > 0 else 0)
            
            monthly.append({
                "camera_id": cam_id,
                "camera_name": CAMERA_NAMES.get(cam_id, cam_id),
                "current_count": curr,
                "previous_count": prev,
                "change_pct": round(change, 1),
                "trend": "up" if change > 10 else "down" if change < -10 else "stable",
            })
        
        monthly.sort(key=lambda x: x["change_pct"], reverse=True)
        
        return {"weekly": weekly, "monthly": monthly}
    
    def _detect_anomalies(self, df) -> List[Dict[str, Any]]:
        """Detect anomalies using Z-score via Spark aggregation."""
        camera_counts = df.groupBy("camera_id").count()
        
        stats = camera_counts.agg(
            F.mean("count").alias("mean"),
            F.stddev("count").alias("std")
        ).collect()[0]
        
        mean_val = stats["mean"] or 0
        std_val = stats["std"] or 1
        if std_val == 0:
            std_val = 1
        
        results = []
        for row in camera_counts.collect():
            cam_id = row["camera_id"]
            count = row["count"]
            z = (count - mean_val) / std_val
            
            results.append({
                "camera_id": cam_id,
                "camera_name": CAMERA_NAMES.get(cam_id, cam_id),
                "event_count": count,
                "z_score": round(z, 2),
                "is_anomaly": z >= 1.5,
                "severity": "critical" if z >= 2.5 else "warning" if z >= 1.5 else "normal",
            })
        
        results.sort(key=lambda x: x["z_score"], reverse=True)
        return results


# Singleton and caching
_insights_job: Optional[SparkInsightsJob] = None
_cached_results: Optional[Dict[str, Any]] = None
_cache_time: Optional[datetime] = None
CACHE_DURATION_HOURS = 24


def get_spark_insights(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get Spark ML insights, using cache if available.
    
    REQUIRES: PySpark, HDFS, and InsightsModel available.
    
    Returns:
        Dict with ML patterns, rules, predictions, and trends
    """
    global _insights_job, _cached_results, _cache_time
    
    # Check cache
    if not force_refresh and _cached_results and _cache_time:
        age = (datetime.now() - _cache_time).total_seconds() / 3600
        if age < CACHE_DURATION_HOURS:
            logger.info(f"Returning cached Spark ML insights (age: {age:.1f}h)")
            return _cached_results
    
    # Run Spark ML job
    if _insights_job is None:
        _insights_job = SparkInsightsJob()
    
    _cached_results = _insights_job.run()
    _cache_time = datetime.now()
    
    return _cached_results
