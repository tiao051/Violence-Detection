"""
Spark Insights Job - Camera Credibility Training Pipeline

Implements the Camera Intelligence system:
1. K-Means: Clustering camera behaviors (Noisy/Reliable/Selective)  
2. FP-Growth: Mining false alarm patterns from verified data
3. Random Forest: Predicting camera credibility tiers (HIGH/MEDIUM/LOW)

Requires verified data with human labels (true_positive/false_positive).
Artifacts are saved to: /app/ai_service/insights/data/
"""

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.clustering import KMeans
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

logger = logging.getLogger(__name__)

# Constants
ARTIFACTS_DIR = "/app/ai_service/insights/data"

# Camera Mapping
CAMERA_NAMES = {
    "cam1": "Luy Ban Bich Street",
    "cam2": "Au Co Junction", 
    "cam3": "Tan Ky Tan Quy Street",
    "cam4": "Tan Phu Market",
    "cam5": "Dam Sen Park",
}

class SparkInsightsJob:
    def __init__(self, hdfs_namenode: str = None, hdfs_path: str = "/analytics/raw", spark_master: str = None):
        self.hdfs_namenode = hdfs_namenode or os.getenv("HDFS_NAMENODE", "hdfs-namenode:9000")
        self.hdfs_path = hdfs_path
        self.spark_master = spark_master or os.getenv("SPARK_MASTER", "local[*]")
        self.spark = None
        
        # Ensure artifacts dir exists
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    def _init_spark(self):
        if self.spark: return self.spark
        
        logger.info(f"Initializing Spark Training Session ({self.spark_master})...")
        os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':/app:/app/ai_service'
        
        self.spark = SparkSession.builder \
            .appName("Violence_Camera_Credibility_Training") \
            .master(self.spark_master) \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.hadoop.fs.defaultFS", f"hdfs://{self.hdfs_namenode}") \
            .config("spark.executor.extraClassPath", "/app:/app/ai_service") \
            .config("spark.pyspark.python", "python3") \
            .getOrCreate()
            
        self.spark.sparkContext.setLogLevel("INFO")
        return self.spark

    def run(self) -> Dict[str, Any]:
        start_time = datetime.now()
        logger.info("STARTING CAMERA CREDIBILITY TRAINING PIPELINE")
        
        try:
            self._init_spark()
            
            # 1. Load & Preprocess Data
            df = self._load_data()
            df_processed = self._preprocess(df)
            df_processed.cache()
            count = df_processed.count()
            logger.info(f"Loaded {count} total samples")

            # Check verified data count
            verified_count = df_processed.filter(F.col("is_verified") == True).count()
            logger.info(f"Found {verified_count} verified samples")
            
            if verified_count < 100:
                logger.warning("Insufficient verified data. Need at least 100 verified samples.")
                logger.warning("Run generate_verified_scenarios.py to create synthetic training data")
                df_processed.unpersist()
                return {
                    "success": False,
                    "error": "Insufficient verified data",
                    "verified_count": verified_count,
                    "required": 100
                }
            
            # 2. Compute Camera Behavior Features
            logger.info("Computing camera behavior features from verified data...")
            camera_features = self._compute_camera_behavior_features(df_processed)
            
            # 3. Train K-Means Clustering
            logger.info("Training K-Means clustering on camera behaviors...")
            clusters = self._train_kmeans_behavior_clustering(camera_features)
            self._save_json(clusters, "camera_clusters.json")
            
            # 4. Mine False Alarm Patterns
            logger.info("Mining false alarm patterns with FP-Growth...")
            fp_patterns = self._mine_false_alarm_patterns(df_processed)
            self._save_json(fp_patterns, "false_alarm_patterns.json")
            
            # 5. Train Random Forest for Credibility Prediction
            logger.info("Training Random Forest for camera credibility prediction...")
            credibility_data, rf_metrics = self._train_credibility_rf(camera_features, clusters)
            self._save_json(credibility_data, "camera_credibility.json")
            
            # Keep trends/anomalies for backward compatibility
            trends = self._compute_trends(df_processed)
            anomalies = self._detect_anomalies(df_processed)
            
            # 6. Forecast Violence Trends (Random Forest Regressor)
            logger.info("Forecasting violence trends with Random Forest...")
            forecast = self._forecast_violence_rf(df_processed)
            
            # 7. Compute Peak Danger Heatmap
            logger.info("Computing peak danger heatmap...")
            heatmap = self._compute_heatmap(df_processed)
            
            # 8. Generate Strategic Recommendations
            logger.info("Synthesizing strategic recommendations...")
            strategies = self._generate_strategy(trends, heatmap, anomalies, forecast)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
            
            df_processed.unpersist()
            
            return {
                "success": True,
                "verified_count": verified_count,
                "clusters": clusters,
                "false_alarm_patterns": len(fp_patterns),
                "cameras_analyzed": len(credibility_data),
                "rf_metrics": rf_metrics,
                "training_time": elapsed,
                "trends": trends,
                "anomalies": anomalies,
                "forecast": forecast,
                "heatmap": heatmap,
                "strategies": strategies
            }
            
            # Save dashboard stats
            dashboard_stats = {
                "trends": trends, 
                "anomalies": anomalies,
                "forecast": forecast,
                "heatmap": heatmap,
                "strategies": strategies,
                "generated_at": datetime.now().isoformat()
            }
            self._save_json(dashboard_stats, "dashboard_stats.json")
            
            return {
                "success": True,
                "verified_count": verified_count,
                "clusters": clusters,
                "false_alarm_patterns": len(fp_patterns),
                "cameras_analyzed": len(credibility_data),
                "rf_metrics": rf_metrics,
                "training_time": elapsed,
                "dashboard_stats_saved": True
            }
            
        except Exception as e:
            logger.error(f"Training Pipeline Failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _load_data(self):
        """Load data from HDFS or local file."""
        # Try HDFS first
        try:
            path = f"hdfs://{self.hdfs_namenode}{self.hdfs_path}/*.csv"
            df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(path)
            logger.info(f"Loaded data from HDFS: {path}")
            return df
        except Exception as e:
            # Fallback to local file
            logger.warning(f"HDFS not available: {e}")
            local_path = "ai_service/tmp/verified_scenarios.csv"
            logger.info(f"Loading from local file: {local_path}")
            df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(local_path)
            return df

    def _preprocess(self, df: DataFrame):
        df = df.withColumn("confidence", F.col("confidence").cast("double")) \
               .withColumn("duration", F.col("duration").cast("double")) \
               .withColumn("is_verified", F.col("is_verified").cast("boolean"))

        # 1. Convert Timestamp
        df = df.withColumn("dt", F.from_unixtime(F.col("timestamp") / 1000).cast("timestamp"))
        
        # 2. Extract Time Features
        df = df.withColumn("hour", F.hour("dt")) \
               .withColumn("day_of_week", F.dayofweek("dt") - 1) \
               .withColumn("day_name", F.date_format("dt", "EEEE"))
        
        # 3. Remove invalid rows
        df = df.filter(F.col("hour").isNotNull() & F.col("confidence").isNotNull())

        # 4. Categorize period
        df = df.withColumn("period", 
            F.when((F.col("hour") < 6) | (F.col("hour") >= 22), "Night")
            .when(F.col("hour") >= 17, "Evening")
            .when(F.col("hour") >= 12, "Afternoon")
            .otherwise("Morning")
        )
        
        # 5. Weekend boolean
        df = df.withColumn("is_weekend", F.when(F.col("day_of_week").isin([0, 6]), 1).otherwise(0))
        
        # 6. Clamp duration
        df = df.withColumn("duration", F.when(F.col("duration") > 300, 300).otherwise(F.col("duration")))
        
        logger.info(f"After preprocessing: {df.count()} valid rows")
            
        return df

    def _save_json(self, data, filename):
        path = os.path.join(ARTIFACTS_DIR, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved artifact: {path}")

    # ==================== CAMERA CREDIBILITY TRAINING ====================
    
    def _compute_camera_behavior_features(self, df: DataFrame) -> DataFrame:
        """
        Compute behavior features per camera from VERIFIED data only.
        Returns DataFrame with camera-level aggregated features.
        """
        # Filter to verified alerts only
        df_verified = df.filter(F.col("is_verified") == True)
        
        # Aggregate per camera
        camera_features = df_verified.groupBy("camera_id").agg(
            # True/False positive rates
            (F.sum(F.when(F.col("verification_status") == "true_positive", 1).otherwise(0)) / F.count("*")).alias("true_positive_rate"),
            (F.sum(F.when(F.col("verification_status") == "false_positive", 1).otherwise(0)) / F.count("*")).alias("false_positive_rate"),
            
            # Confidence metrics
            F.avg("confidence").alias("avg_confidence"),
            F.stddev("confidence").alias("confidence_std"),
            
            # Duration metrics  
            F.avg("duration").alias("avg_duration"),
            F.stddev("duration").alias("duration_std"),
            
            # Temporal patterns
            (F.sum(F.when((F.col("hour") >= 22) | (F.col("hour") < 6), 1).otherwise(0)) / F.count("*")).alias("night_ratio"),
            (F.sum(F.when(F.col("is_weekend") == 1, 1).otherwise(0)) / F.count("*")).alias("weekend_ratio"),
            
            # Volume
            (F.count("*") / F.countDistinct(F.to_date("dt"))).alias("alerts_per_day"),
            F.count("*").alias("total_verified")
        )
        
        logger.info(f"Computed features for {camera_features.count()} cameras")
        return camera_features
    
    def _train_kmeans_behavior_clustering(self, camera_features: DataFrame) -> List[Dict]:
        """
        K-Means clustering on camera behavior features.
        Clusters cameras into behavioral groups.
        """
        feature_cols = [
            "true_positive_rate", "false_positive_rate",
            "avg_confidence", "confidence_std",
            "alerts_per_day", "night_ratio"
        ]
        
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        kmeans = KMeans(k=4, seed=42, featuresCol="scaled_features")
        
        # Build pipeline
        features_df = assembler.transform(camera_features)
        scaler_model = scaler.fit(features_df)
        scaled_df = scaler_model.transform(features_df)
        model = kmeans.fit(scaled_df)
        
        # Get predictions
        clustered = model.transform(scaled_df)
        
        # Interpret clusters
        clusters = []
        for cluster_id in range(4):
            cluster_df = clustered.filter(F.col("prediction") == cluster_id)
            
            if cluster_df.count() == 0:
                continue
                
            stats = cluster_df.agg(
                F.avg("true_positive_rate").alias("avg_tp_rate"),
                F.avg("false_positive_rate").alias("avg_fp_rate"),
                F.avg("avg_confidence").alias("avg_conf"),
                F.collect_list("camera_id").alias("cameras")
            ).collect()[0]
            
            # Name clusters based on characteristics
            tp_rate = stats["avg_tp_rate"]
            fp_rate = stats["avg_fp_rate"]
            
            if fp_rate > 0.3:
                name, base_cred = "Noisy Camera", 0.35
            elif tp_rate > 0.8 and fp_rate < 0.15:
                name, base_cred = "Reliable Camera", 0.85
            elif tp_rate > 0.7:
                name, base_cred = "Selective Camera", 0.75
            else:
                name, base_cred = "Overcautious Camera", 0.55
            
            clusters.append({
                "cluster_id": cluster_id,
                "name": name,
                "base_credibility": base_cred,
                "cameras": stats["cameras"],
                "avg_tp_rate": round(tp_rate, 2),
                "avg_fp_rate": round(fp_rate, 2),
                "avg_confidence": round(stats["avg_conf"], 2)
            })
        
        logger.info(f"Created {len(clusters)} clusters")
        return clusters
    
    def _mine_false_alarm_patterns(self, df: DataFrame) -> List[Dict]:
        """
        FP-Growth to mine patterns leading to false positives.
        Only uses VERIFIED data.
        """
        df_verified = df.filter(F.col("is_verified") == True)
        
        # Create items with verification status
        def make_items_with_verification(cam, hour, conf, dur, day, verified_status):
            items = []
            items.append(f"Cam_{cam}")
            
            # Bin hour into periods
            if hour < 6 or hour >= 22:
                items.append("Hour_night")
            elif hour < 12:
                items.append("Hour_morning")
            elif hour < 17:
                items.append("Hour_afternoon")
            else:
                items.append("Hour_evening")
            
            # Bin confidence
            if conf < 0.5:
                items.append("Conf_low")
            elif conf < 0.75:
                items.append("Conf_medium")
            else:
                items.append("Conf_high")
            
            # Bin duration
            if dur < 3:
                items.append("Dur_short")
            elif dur < 15:
                items.append("Dur_medium")
            else:
                items.append("Dur_long")
            
            items.append(f"Day_{day}")
            items.append(f"Verified_{verified_status}")
            
            return items
        
        make_items_udf = F.udf(make_items_with_verification, ArrayType(StringType()))
        
        df_items = df_verified.withColumn(
            "items",
            make_items_udf("camera_id", "hour", "confidence", "duration", "day_name", "verification_status")
        )
        
        # Mine patterns
        fp_growth = FPGrowth(itemsCol="items", minSupport=0.05, minConfidence=0.6)
        model = fp_growth.fit(df_items)
        
        # Extract FALSE POSITIVE patterns
        fp_rules = model.associationRules.filter(
            F.array_contains(F.col("consequent"), "Verified_false_positive")
        ).orderBy(F.col("confidence").desc()).limit(15).collect()
        
        patterns = []
        for rule in fp_rules:
            # Remove "Verified_false_positive" from antecedent if present
            antecedent = [item for item in rule["antecedent"] if not item.startswith("Verified_")]
            
            patterns.append({
                "pattern": antecedent,
                "outcome": "false_positive",
                "confidence": round(rule["confidence"], 2),
                "support": round(rule["support"], 2),
                "lift": round(rule["lift"], 2),
                "interpretation": f"Pattern {antecedent} often leads to false alarms",
                "action": "reduce_confidence"
            })
        
        logger.info(f"Found {len(patterns)} false alarm patterns")
        return patterns
    
    def _train_credibility_rf(self, camera_features: DataFrame, clusters: List[Dict]) -> tuple:
        """
        Train Random Forest to predict camera credibility tier.
        Uses verified data features + cluster assignment.
        
        Returns:
            (credibility_data, metrics): List of camera credibility scores and RF training metrics
        """
        # Create cluster lookup
        cluster_lookup = {}
        for cluster in clusters:
            for cam in cluster["cameras"]:
                cluster_lookup[cam] = {
                    "cluster_id": cluster["cluster_id"],
                    "cluster_name": cluster["name"],
                    "base_credibility": cluster["base_credibility"]
                }
        
        # Add cluster_id to camera_features
        @F.udf(returnType=StringType())
        def get_cluster_id_udf(cam_id):
            return str(cluster_lookup.get(cam_id, {}).get("cluster_id", -1))
        
        df_with_cluster = camera_features.withColumn(
            "cluster_id_str",
            get_cluster_id_udf(F.col("camera_id"))
        ).withColumn(
            "cluster_id",
            F.col("cluster_id_str").cast("int")
        )
        
        # Create credibility tier labels based on TP rate and FP rate
        def compute_tier_label(tp_rate, fp_rate):
            if tp_rate >= 0.80 and fp_rate < 0.15:
                return "HIGH"  # Highly reliable
            elif tp_rate >= 0.60:
                return "MEDIUM"  # Moderately reliable
            else:
                return "LOW"  # Needs attention
        
        tier_udf = F.udf(compute_tier_label, StringType())
        
        df_labeled = df_with_cluster.withColumn(
            "credibility_tier",
            tier_udf("true_positive_rate", "false_positive_rate")
        )
        
        # Prepare features for RF
        feature_cols = [
            "true_positive_rate",
            "false_positive_rate",
            "avg_confidence",
            "confidence_std",
            "avg_duration",
            "alerts_per_day",
            "night_ratio",
            "weekend_ratio",
            "cluster_id"
        ]
        
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_features = assembler.transform(df_labeled)
        
        # Index labels
        indexer = StringIndexer(inputCol="credibility_tier", outputCol="label")
        df_indexed = indexer.fit(df_features).transform(df_features)
        
        # Train Random Forest
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=50,
            maxDepth=5,
            seed=42
        )
        
        rf_model = rf.fit(df_indexed)
        predictions = rf_model.transform(df_indexed)
        
        # Evaluate
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        
        logger.info(f"RF Credibility Model Accuracy: {accuracy:.3f}")
        
        # Generate final credibility data
        credibility_data = []
        
        for row in df_labeled.collect():
            cam_id = row["camera_id"]
            tp_rate = row["true_positive_rate"]
            fp_rate = row["false_positive_rate"]
            tier = row["credibility_tier"]
            
            cluster_info = cluster_lookup.get(cam_id, {
                "cluster_id": -1,
                "cluster_name": "Unknown",
                "base_credibility": 0.5
            })
            
            # Compute final credibility score (weighted)
            # 70% from actual TP rate, 30% from cluster baseline
            credibility_score = (tp_rate * 0.7) + (cluster_info["base_credibility"] * 0.3)
            
            credibility_data.append({
                "camera_id": cam_id,
                "camera_name": CAMERA_NAMES.get(cam_id, cam_id),
                "credibility_score": round(credibility_score, 2),
                "credibility_tier": tier,
                "cluster": cluster_info["cluster_name"],
                "metrics": {
                    "true_positive_rate": round(tp_rate, 2),
                    "false_positive_rate": round(fp_rate, 2),
                    "total_verified": int(row["total_verified"]),
                    "avg_confidence": round(row["avg_confidence"], 2),
                    "avg_duration": round(row["avg_duration"], 1),
                    "alerts_per_day": round(row["alerts_per_day"], 1)
                },
                "recommendation": self._generate_recommendation(tier, fp_rate, credibility_score)
            })
        
        metrics = {
            "algorithm": "Random Forest",
            "accuracy": round(accuracy, 3),
            "num_trees": 50,
            "max_depth": 5,
            "features_used": feature_cols,
            "cameras_trained": len(credibility_data)
        }
        
        logger.info(f"Trained credibility RF for {len(credibility_data)} cameras")
        return credibility_data, metrics
    
    def _generate_recommendation(self, tier: str, fp_rate: float, score: float) -> str:
        """Generate human-readable recommendation based on camera credibility."""
        if tier == "HIGH":
            return f"Highly trustworthy (score: {score:.2f}) - prioritize alerts from this camera"
        elif tier == "LOW":
            if fp_rate > 0.3:
                return f"⚠️ High false alarm rate ({fp_rate*100:.0f}%) - needs recalibration or maintenance"
            else:
                return f"⚠️ Low credibility (score: {score:.2f}) - verify alerts manually"
        else:
            return f"Moderate reliability (score: {score:.2f}) - normal monitoring"

    # --- Legacy Helpers for Dashboard Trends ---
    def _compute_trends(self, df: DataFrame) -> Dict[str, List]:
        """Compute weekly and monthly trends using Spark SQL."""
        now = datetime.now()
        
        # Weekly trends
        current_week_start = now - timedelta(days=now.weekday())
        current_week_start = current_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        last_week_start = current_week_start - timedelta(days=7)
        
        # Monthly trends
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
        
        # Convert dates to strings
        cw_str = current_week_start.strftime("%Y-%m-%d")
        lw_str = last_week_start.strftime("%Y-%m-%d")
        cm_str = current_month_start.strftime("%Y-%m-%d")
        lm_str = last_month_start.strftime("%Y-%m-%d")
        
        def get_period_counts(start_date, end_date):
            return df.filter((F.col("dt") >= start_date) & (F.col("dt") < end_date)) \
                     .groupBy("camera_id").count().collect()
        
        # 1. Weekly
        curr_counts = {r["camera_id"]: r["count"] for r in get_period_counts(cw_str, (now + timedelta(days=1)).strftime("%Y-%m-%d"))}
        prev_counts = {r["camera_id"]: r["count"] for r in get_period_counts(lw_str, cw_str)}
        
        weekly = []
        for cam_id in set(curr_counts.keys()) | set(prev_counts.keys()):
            curr = curr_counts.get(cam_id, 0)
            prev = prev_counts.get(cam_id, 0)
            change = ((curr - prev) / prev) * 100 if prev > 0 else (100.0 if curr > 0 else 0.0)
                
            weekly.append({
                "camera_id": cam_id,
                "camera_name": CAMERA_NAMES.get(cam_id, cam_id),
                "current_count": curr,
                "previous_count": prev,
                "change_pct": round(change, 1),
                "trend": "up" if change > 10 else "down" if change < -10 else "stable"
            })
        weekly.sort(key=lambda x: x["change_pct"], reverse=True)
        
        # 2. Monthly
        curr_m_counts = {r["camera_id"]: r["count"] for r in get_period_counts(cm_str, (now + timedelta(days=1)).strftime("%Y-%m-%d"))}
        prev_m_counts = {r["camera_id"]: r["count"] for r in get_period_counts(lm_str, cm_str)}
        
        monthly = []
        for cam_id in set(curr_m_counts.keys()) | set(prev_m_counts.keys()):
            curr = curr_m_counts.get(cam_id, 0)
            prev = prev_m_counts.get(cam_id, 0)
            change = ((curr - prev) / prev) * 100 if prev > 0 else (100.0 if curr > 0 else 0.0)
                
            monthly.append({
                "camera_id": cam_id,
                "camera_name": CAMERA_NAMES.get(cam_id, cam_id),
                "current_count": curr,
                "previous_count": prev,
                "change_pct": round(change, 1),
                "trend": "up" if change > 10 else "down" if change < -10 else "stable"
            })
        monthly.sort(key=lambda x: x["change_pct"], reverse=True)
        
        return {"weekly": weekly, "monthly": monthly}

    def _detect_anomalies(self, df: DataFrame) -> List[Dict]:
        """Detect anomalies using Z-score via Spark aggregation."""
        camera_counts = df.groupBy("camera_id").count()
        
        stats = camera_counts.agg(
            F.mean("count").alias("mean"),
            F.stddev("count").alias("std")
        ).collect()[0]
        
        mean_val = stats["mean"] or 0
        std_val = stats["std"] or 1
        if std_val == 0: std_val = 1
        
        results = []
        rows = camera_counts.collect()
        
        for row in rows:
            cam_id = row["camera_id"]
            count = row["count"]
            z = (count - mean_val) / std_val
            
            results.append({
                "camera_id": cam_id,
                "camera_name": CAMERA_NAMES.get(cam_id, cam_id),
                "event_count": count,
                "z_score": round(z, 2),
                "is_anomaly": z >= 1.5,
                "severity": "critical" if z >= 2.5 else "warning" if z >= 1.5 else "normal"
            })
            
        results.sort(key=lambda x: x["z_score"], reverse=True)
        return results

    # ==================== ADVANCED ANALYTICS (FORECASTING & STRATEGY) ====================

    def _forecast_violence_rf(self, df: DataFrame) -> Dict[str, Any]:
        """
        Forecast next 7 days of violence events using Random Forest Regressor.
        Features: Day of week, Day of month, Is Weekend.
        """
        # 1. Aggregate daily counts
        daily_counts = df.groupBy(F.to_date("dt").alias("date")) \
                         .agg(F.count("*").alias("count")) \
                         .orderBy("date")
        
        # 2. Extract temporal features
        daily_features = daily_counts.withColumn("day_of_week", F.dayofweek("date") - 1) \
                                     .withColumn("day_of_month", F.dayofmonth("date")) \
                                     .withColumn("is_weekend", F.when(F.col("day_of_week").isin([0, 6]), 1).otherwise(0)) \
                                     .withColumn("date_idx", F.datediff(F.col("date"), F.to_date(F.lit("2024-01-01")))) # Monotonic trend
        
        # 3. Prepare ML data
        feature_cols = ["day_of_week", "day_of_month", "is_weekend", "date_idx"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        
        ml_df = assembler.transform(daily_features)
        
        # 4. Train Random Forest Regressor
        # Use more trees for stability
        rf = RandomForestRegressor(featuresCol="features", labelCol="count", numTrees=100, maxDepth=8, seed=42)
        model = rf.fit(ml_df)
        
        # 5. Predict next 7 days
        last_date_row = daily_features.orderBy(F.col("date").desc()).first()
        last_date = last_date_row["date"] if last_date_row else datetime.now().date()
        start_idx = last_date_row["date_idx"] if last_date_row else 0
        
        future_data = []
        for i in range(1, 8):
            next_date = last_date + timedelta(days=i)
            future_data.append({
                "date": next_date,
                "day_of_week": next_date.weekday(), # 0=Mon
                "day_of_month": next_date.day,
                "is_weekend": 1 if next_date.weekday() in [5, 6] else 0,
                "date_idx": start_idx + i
            })
            
        future_df = self.spark.createDataFrame(future_data)
        future_ml = assembler.transform(future_df)
        predictions = model.transform(future_ml).collect()
        
        # 6. Format results
        forecast_points = []
        total_predicted = 0
        for row in predictions:
            val = max(0, round(row["prediction"], 1)) # No negative violence
            forecast_points.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "predicted_count": val,
                "day": row["date"].strftime("%A")
            })
            total_predicted += val
            
        # Get historical last 7 days for comparison
        history_points = daily_counts.orderBy(F.col("date").desc()).limit(7).collect()
        history_cleanup = [{"date": r["date"].strftime("%Y-%m-%d"), "count": r["count"]} for r in reversed(history_points)]
        
        return {
            "forecast": forecast_points,
            "history": history_cleanup,
            "total_predicted_next_week": round(total_predicted, 1),
            "trend_direction": "increasing" if total_predicted > sum(h["count"] for h in history_cleanup) else "decreasing"
        }

    def _compute_heatmap(self, df: DataFrame) -> List[Dict]:
        """
        Compute 7x24 Heatmap (Day x Hour) of violence density.
        Only uses VERIFIED True Positive data for accuracy.
        """
        # Filter for verified or high confidence
        df_clean = df.filter((F.col("confidence") > 0.7) | (F.col("is_verified") == True))
        
        # Group by Day (0-6) and Hour (0-23)
        # Note: DayOfWeek in Spark is 1=Sunday, 2=Monday... we standardize to 0=Mon, 6=Sun
        heatmap_data = df_clean.withColumn("day_std", (F.dayofweek("dt") + 5) % 7) \
                               .groupBy("day_std", "hour") \
                               .count() \
                               .collect()
        
        # Initialize full grid
        grid = []
        counts = {(r["day_std"], r["hour"]): r["count"] for r in heatmap_data}
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        for d_idx, day_name in enumerate(days):
            day_hours = []
            for h in range(24):
                val = counts.get((d_idx, h), 0)
                day_hours.append({
                    "hour": h,
                    "count": val,
                    "level": "high" if val > 10 else "medium" if val > 5 else "low" if val > 0 else "none"
                })
            grid.append({
                "day": day_name,
                "hours": day_hours,
                "total_events": sum(h["count"] for h in day_hours)
            })
            
        return grid

    def _generate_strategy(self, trends: Dict, heatmap: List, anomalies: List, forecast: Dict) -> List[Dict]:
        """
        Strategic Recommendation Engine (Prescriptive Analytics).
        Synthesizes insights to tell the user WHAT to do.
        """
        strategies = []
        
        # 1. Forecast Strategy
        predicted_total = forecast.get("total_predicted_next_week", 0)
        direction = forecast.get("trend_direction", "stable")
        
        if direction == "increasing" and predicted_total > 50:
             strategies.append({
                "type": "deployment",
                "priority": "HIGH",
                "title": "Predicted Violence Surge",
                "message": f"Forecast models predict a violence increase (approx {predicted_total} events) next week. Review staffing schedules.",
                "action": "Increase Patrol Frequency"
            })
            
        # 2. Heatmap Strategy (Find peak day)
        max_day = max(heatmap, key=lambda x: x["total_events"])
        if max_day["total_events"] > 5:
            strategies.append({
                "type": "deployment",
                "priority": "MEDIUM",
                "title": f"High Risk Period: {max_day['day']}",
                "message": f"{max_day['day']} has the highest accumulated violence density. Focus monitoring efforts on this day.",
                "action": "Schedule Extra Shift"
            })
            
        # 3. Anomaly Strategy
        critical_anomalies = [a for a in anomalies if a["severity"] == "critical"]
        for anomaly in critical_anomalies:
            strategies.append({
                "type": "maintenance",
                "priority": "CRITICAL",
                "title": f"Anomaly on {anomaly['camera_name']}",
                "message": f"Camera is showing statistical deviation (Z-Score: {anomaly['z_score']}). May be a technical fault.",
                "action": "Technician Inspection Required"
            })
            
        # 4. Fallback if quiet
        if not strategies:
             strategies.append({
                "type": "info",
                "priority": "LOW",
                "title": "System Stable",
                "message": "No significant anomalies or risk spikes detected. Maintain standard procedure.",
                "action": "Standard Monitoring"
            })
            
        return strategies


# Singleton Management  
_insights_job = None

def get_insights_job_instance():
    global _insights_job
    if _insights_job is None:
        _insights_job = SparkInsightsJob()
    return _insights_job

def warmup_spark():
    try:
        get_insights_job_instance()._init_spark()
    except: pass


if __name__ == "__main__":
    job = SparkInsightsJob()
    res = job.run()
    print(json.dumps(res, indent=2, default=str))
