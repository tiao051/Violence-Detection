"""
Spark Insights Job - Offline Training Pipeline

Implements the "Logistics" phase:
1. K-Means: Profiling Cameras -> camera_profiles.json
2. FP-Growth: Mining High Risk Rules -> risk_rules.json
3. Random Forest: Training Severity Predictor -> hdfs:///models/severity_rf_model
4. Scikit-learn RF: Lightweight model for backend inference -> severity_rf_sklearn.pkl

Artifacts are saved to: /app/ai_service/insights/data/
"""

import os
import json
import logging
import shutil
import pickle
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Scikit-learn imports for lightweight model export
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.preprocessing import LabelEncoder, StandardScaler as SklearnScaler
import numpy as np

logger = logging.getLogger(__name__)

# Constants
ARTIFACTS_DIR = "/app/ai_service/insights/data"
HDFS_MODEL_PATH = "/models/severity_rf_model"

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
        self.spark: Optional[SparkSession] = None
        
        # Ensure artifacts dir exists
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    def _init_spark(self) -> SparkSession:
        if self.spark: return self.spark
        
        logger.info(f"Initializing Spark Training Session ({self.spark_master})...")
        os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':/app:/app/ai_service'
        
        self.spark = SparkSession.builder \
            .appName("Violence_Offline_Training") \
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
        logger.info("STARTING OFFLINE TRAINING PIPELINE")
        
        try:
            self._init_spark()
            
            # 1. Load & Preprocess Data
            df = self._load_data()
            df_processed = self._preprocess(df)
            df_processed.cache()
            count = df_processed.count()
            logger.info(f"Loaded {count} training samples for analysis.")

            results = {}

            # 2. MATCH 1: K-MEANS PROFILING
            # Goal: Group cameras/times into profiles (e.g. "High Risk Zone", "Safe Zone")
            logger.info("Training K-Means (Camera Profiling)...")
            profiles = self._train_kmeans_profiling(df_processed)
            self._save_json(profiles, "camera_profiles.json")
            results["profiles"] = profiles

            # 3. MATCH 2: FP-GROWTH RULES
            # Goal: Find rules causing HIGH severity
            logger.info("Mining FP-Growth Rules (High Severity)...")
            rules = self._mine_association_rules(df_processed)
            self._save_json(rules, "risk_rules.json")
            results["rules"] = rules

            # 4. MATCH 3: RANDOM FOREST MODEL
            # Goal: Train model to predict severity_level based on context
            logger.info("Training Random Forest (Severity Predictor)...")
            model_metrics = self._train_random_forest(df_processed)
            results["model_metrics"] = model_metrics
            
            # 5. Calculate Basic Stats (Trends) - Keeping this for Dashboard backward compatibility
            trends = self._compute_trends(df_processed)
            anomalies = self._detect_anomalies(df_processed)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"TRAINING COMPLETE in {elapsed:.1f}s. Artifacts saved to {ARTIFACTS_DIR}")
            
            # Release cached data
            df_processed.unpersist()
            
            return {
                "success": True,
                "training_time": elapsed,
                "profiles": profiles,
                "rules": rules,
                "model_metrics": model_metrics,
                "trends": trends, 
                "anomalies": anomalies 
            }
            
        except Exception as e:
            logger.error(f"Training Pipeline Failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            # Release cache if still exists
            try:
                df_processed.unpersist()
            except:
                pass

    def _load_data(self):
        path = f"hdfs://{self.hdfs_namenode}{self.hdfs_path}/*.csv"
        return self.spark.read.option("header", "true").option("inferSchema", "true").csv(path)

    def _preprocess(self, df: DataFrame):
        df = df.withColumn("confidence", F.col("confidence").cast("double")) \
               .withColumn("duration", F.col("duration").cast("double"))

        # 1. Convert Timestamp
        df = df.withColumn("dt", F.from_unixtime(F.col("timestamp") / 1000).cast("timestamp"))
        
        # 2. Extract Time Features
        df = df.withColumn("hour", F.hour("dt")) \
               .withColumn("day_of_week", F.dayofweek("dt") - 1) \
               .withColumn("day_name", F.date_format("dt", "EEEE"))
        
        # 3. Remove invalid rows - CRITICAL: filter before assigning defaults
        df = df.filter(F.col("hour").isNotNull() & F.col("confidence").isNotNull() & F.col("severity_level").isNotNull())

        # 4. Categorize period (Native Spark Logic)
        df = df.withColumn("period", 
            F.when((F.col("hour") < 6) | (F.col("hour") >= 22), "Night")
            .when(F.col("hour") >= 17, "Evening")
            .when(F.col("hour") >= 12, "Afternoon")
            .otherwise("Morning")
        )
        
        # 5. Weekend boolean
        df = df.withColumn("is_weekend", F.when(F.col("day_of_week").isin([0, 6]), 1).otherwise(0))
        
        # 6. Clamp duration to reasonable values (outlier handling)
        df = df.withColumn("duration", F.when(F.col("duration") > 300, 300).otherwise(F.col("duration")))
        
        # 7. Validate severity_level values and remove bad ones
        valid_severities = ["LOW", "MEDIUM", "HIGH"]
        df = df.filter(F.col("severity_level").isin(valid_severities))
        
        logger.info(f"After preprocessing: {df.count()} valid rows (removed invalid severity)")
            
        return df

    def _train_kmeans_profiling(self, df):
        # Feature Engineering: hour, traffic_volume? 
        # Here we simple cluster on (Hour, Confidence) to see patterns
        vec = VectorAssembler(inputCols=["hour", "confidence"], outputCol="features")
        
        kmeans = KMeans(k=3, seed=42)
        model = kmeans.fit(vec.transform(df))
        
        # Interpret Clusters
        centers = model.clusterCenters()
        profiles = []
        for i, center in enumerate(centers):
            hour = center[0]
            conf = center[1]
            # Semantic Naming
            if conf > 0.85: name = "Critical Risk"
            elif conf > 0.6: name = "Moderate Risk"
            else: name = "Low Risk / Noise"
            
            profiles.append({
                "cluster_id": i,
                "name": name,
                "avg_hour": round(hour, 1),
                "avg_confidence": round(conf, 2),
                "description": f"{name} around {int(hour)}h"
            })
        return profiles

    def _mine_association_rules(self, df):
        # Only mine patterns that lead to HIGH severity (The "Dangerous" patterns)
        # Or mine all and filter? Mining all is safer to find context.
        
        # Create Items Array: ["Zone_RED", "Day_Friday", "Time_Night", "Sev_HIGH"]
        # Use camera zone mapping if possible, else use name
        
        def make_items(cam, day, period, sev):
            items = []
            items.append(f"Cam_{cam}")
            items.append(f"Day_{day}")
            items.append(f"Time_{period}")
            items.append(f"Sev_{sev}")
            return items
            
        mkbox = F.udf(make_items, ArrayType(StringType()))
        df_trans = df.withColumn("items", mkbox("camera_id", "day_name", "period", "severity_level"))
        
        fp = FPGrowth(itemsCol="items", minSupport=0.05, minConfidence=0.5)
        model = fp.fit(df_trans)
        
        # Extract Rules where Consequent contains 'Sev_HIGH' using proper array check
        rules = model.associationRules.filter(
            F.array_contains(F.col("consequent"), "Sev_HIGH")
        ).orderBy(F.col("lift").desc()).limit(10).collect()
        
        results = []
        for r in rules:
            results.append({
                "if": r["antecedent"],
                "then": r["consequent"],
                "confidence": round(r["confidence"], 2),
                "lift": round(r["lift"], 2),
                "rule_text": f"IF {r['antecedent']} THEN {r['consequent']}"
            })
        return results

    def _train_random_forest(self, df):
        """
        Trains a Random Forest classifier with better hyperparameters.
        Includes train/test split and comprehensive evaluation metrics.
        """
        logger.info("Starting Random Forest training pipeline with evaluation...")

        # 1. Train/Test Split FIRST (before any fitting to avoid data leakage)
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        train_count = train_data.count()
        test_count = test_data.count()
        logger.info(f"Training samples: {train_count}, Test samples: {test_count}")
        
        # Check for minimum data
        if train_count < 50:
            logger.warning(f"Low training data ({train_count} samples). Model may not generalize well.")

        # 2. Define Pipeline Stages (indexers will fit only on train_data)
        label_indexer = StringIndexer(inputCol="severity_level", outputCol="severity_index", handleInvalid="keep")
        cam_indexer = StringIndexer(inputCol="camera_id", outputCol="cam_idx", handleInvalid="keep")
        
        # Use more features including confidence & duration (most important!)
        assembler = VectorAssembler(
            inputCols=["hour", "day_of_week", "cam_idx", "is_weekend", "confidence", "duration"], 
            outputCol="features_raw",
            handleInvalid="skip"
        )
        
        # Scale features - withMean=False for sparse vectors compatibility
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True)
        
        # Improved Random Forest hyperparameters
        rf = RandomForestClassifier(
            labelCol="severity_index", 
            featuresCol="features", 
            numTrees=150,           # Good for medium datasets
            maxDepth=10,            # Limit depth to avoid overfitting
            minInstancesPerNode=5,  # Require min samples in leaf
            minInfoGain=0.01,       # Stop splits with small gain
            seed=42,
            subsamplingRate=0.8,    # Use 80% samples per tree
            featureSubsetStrategy="sqrt"  # Use sqrt(n_features) per split
        )
        
        pipeline = Pipeline(stages=[label_indexer, cam_indexer, assembler, scaler, rf])

        # 3. Train Model (indexers fit ONLY on train_data now)
        model = pipeline.fit(train_data)
        
        # 4. Comprehensive Evaluation on Test Set
        predictions = model.transform(test_data)
        
        # Multiple metrics for better understanding
        metrics = {}
        for metric_name in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]:
            evaluator = MulticlassClassificationEvaluator(
                labelCol="severity_index", 
                predictionCol="prediction", 
                metricName=metric_name
            )
            metrics[metric_name] = round(evaluator.evaluate(predictions), 4)
        
        logger.info(f"Model Metrics: Accuracy={metrics['accuracy']}, F1={metrics['f1']}, "
                    f"Precision={metrics['weightedPrecision']}, Recall={metrics['weightedRecall']}")
        
        # 5. Save Strategy: Use Spark's native save with explicit file:// prefix
        # This forces local filesystem instead of HDFS
        final_path = os.path.join(ARTIFACTS_DIR, "severity_rf_model")
        
        try:
            logger.info(f"Saving Random Forest model to: {final_path}...")
            
            # Ensure ARTIFACTS_DIR exists
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            
            # Remove existing model directory if exists
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            
            # Use file:// prefix to force local filesystem (not HDFS)
            local_uri = f"file://{final_path}"
            model.write().overwrite().save(local_uri)
            
            # Verify model was saved
            if not os.path.exists(final_path):
                raise RuntimeError(f"Model save failed: {final_path} not created")
            
            logger.info(f"Pipeline model saved successfully: {final_path}")

        except Exception as e:
            logger.error(f"Critical error during model persistence: {e}", exc_info=True)
            if os.path.exists(final_path):
                shutil.rmtree(final_path, ignore_errors=True)
            raise 

        # 6. Export Metadata (Label Mapping & Indexer Info for inference)
        label_indexer_model = model.stages[0]  # First stage is StringIndexer
        cam_indexer_model = model.stages[1]     # Second stage is camera StringIndexer
        labels = list(label_indexer_model.labels)
        cameras = list(cam_indexer_model.labels)
        
        label_map_path = os.path.join(ARTIFACTS_DIR, "label_map.json")
        indexer_info_path = os.path.join(ARTIFACTS_DIR, "indexer_info.json")
        
        try:
            with open(label_map_path, "w") as f:
                json.dump(labels, f)
            logger.info(f"Label mapping saved: {labels}")
            
            with open(indexer_info_path, "w") as f:
                json.dump({
                    "severity_labels": labels,
                    "camera_ids": cameras,
                    "feature_columns": ["hour", "day_of_week", "camera", "is_weekend", "confidence", "duration"],
                    "model_type": "RandomForestClassifier",
                    "sklearn_model_file": "severity_rf_sklearn.pkl"  # Lightweight model for backend
                }, f, indent=2)
            logger.info(f"Indexer info saved: {len(cameras)} cameras")
        except IOError as e:
            logger.error(f"Failed to save metadata: {e}")

        # 7. Extract Feature Importance
        rf_model = model.stages[-1] 
        feature_names = ["hour", "day_of_week", "camera", "is_weekend", "confidence", "duration"]
        feature_importance = rf_model.featureImportances.toArray().tolist()
        
        importance_dict = {name: round(imp, 4) for name, imp in zip(feature_names, feature_importance)}
        logger.info(f"Feature Importance: {importance_dict}")
        
        # 8. Get label mapping for interpretation
        label_indexer_model = model.stages[0]
        labels = label_indexer_model.labels
        
        # 9. Train and export Scikit-learn model for backend inference (lightweight, no Spark needed)
        sklearn_metrics = self._train_sklearn_model(df, labels, cameras)
        logger.info(f"Sklearn model trained: accuracy={sklearn_metrics.get('accuracy', 'N/A')}")
        
        return {
            "algorithm": "Random Forest (Improved)",
            "hyperparameters": {
                "numTrees": 150,
                "maxDepth": 10,
                "minInstancesPerNode": 5,
                "subsamplingRate": 0.8,
                "featureSubsetStrategy": "sqrt"
            },
            "data_split": {
                "train_samples": train_count,
                "test_samples": test_count
            },
            "features": feature_names,
            "importance": importance_dict,
            "metrics": metrics,
            "sklearn_metrics": sklearn_metrics,
            "labels": labels,
            "status": "Success",
            "model_path": final_path
        }

    def _train_sklearn_model(self, df: DataFrame, severity_labels: List[str], camera_ids: List[str]) -> Dict[str, Any]:
        """
        Train a scikit-learn RandomForest model for lightweight backend inference.
        
        This model can be loaded in backend without Spark dependency.
        Features: [hour, day_of_week, camera_idx, is_weekend, confidence, duration]
        """
        logger.info("Training scikit-learn RandomForest for backend inference...")
        
        try:
            # 1. Collect data to pandas (small enough for memory)
            pandas_df = df.select(
                "hour", "day_of_week", "camera_id", "is_weekend", 
                "confidence", "duration", "severity_level"
            ).toPandas()
            
            logger.info(f"Collected {len(pandas_df)} samples for sklearn training")
            
            # 2. Encode categorical features
            # Camera ID -> numeric index (matching Spark indexer order)
            camera_to_idx = {cam: idx for idx, cam in enumerate(camera_ids)}
            pandas_df["camera_idx"] = pandas_df["camera_id"].map(
                lambda x: camera_to_idx.get(x, len(camera_ids))  # Unknown cameras get max index
            )
            
            # Severity label -> numeric index (matching Spark indexer order)
            severity_to_idx = {sev: idx for idx, sev in enumerate(severity_labels)}
            pandas_df["severity_idx"] = pandas_df["severity_level"].map(
                lambda x: severity_to_idx.get(x, 1)  # Default to MEDIUM (index 1) if unknown
            )
            
            # 3. Prepare features and labels
            feature_cols = ["hour", "day_of_week", "camera_idx", "is_weekend", "confidence", "duration"]
            X = pandas_df[feature_cols].values.astype(np.float32)
            y = pandas_df["severity_idx"].values
            
            # 4. Train/test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 5. Scale features
            scaler = SklearnScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 6. Train Random Forest (matching Spark hyperparameters)
            clf = SklearnRF(
                n_estimators=150,
                max_depth=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1  # Use all cores
            )
            clf.fit(X_train_scaled, y_train)
            
            # 7. Evaluate
            from sklearn.metrics import accuracy_score, f1_score, classification_report
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"Sklearn model: accuracy={accuracy:.4f}, f1={f1:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=severity_labels)}")
            
            # 8. Save model bundle (model + scaler + encodings)
            model_bundle = {
                "model": clf,
                "scaler": scaler,
                "severity_labels": severity_labels,
                "camera_ids": camera_ids,
                "camera_to_idx": camera_to_idx,
                "severity_to_idx": severity_to_idx,
                "feature_columns": feature_cols,
                "feature_importance": dict(zip(feature_cols, clf.feature_importances_.tolist())),
                "metrics": {
                    "accuracy": round(accuracy, 4),
                    "f1": round(f1, 4)
                }
            }
            
            sklearn_model_path = os.path.join(ARTIFACTS_DIR, "severity_rf_sklearn.pkl")
            with open(sklearn_model_path, "wb") as f:
                pickle.dump(model_bundle, f)
            
            logger.info(f"Sklearn model saved to: {sklearn_model_path}")
            
            return {
                "accuracy": round(accuracy, 4),
                "f1": round(f1, 4),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "model_path": sklearn_model_path
            }
            
        except Exception as e:
            logger.error(f"Failed to train sklearn model: {e}", exc_info=True)
            return {"error": str(e)}

    def _save_json(self, data, filename):
        path = os.path.join(ARTIFACTS_DIR, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved artifact: {path}")

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
        
        # Convert dates to strings for filter
        cw_str = current_week_start.strftime("%Y-%m-%d")
        lw_str = last_week_start.strftime("%Y-%m-%d")
        cm_str = current_month_start.strftime("%Y-%m-%d")
        lm_str = last_month_start.strftime("%Y-%m-%d")
        
        # --- Helper for trend aggregation ---
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
            if prev > 0:
                change = ((curr - prev) / prev) * 100
            else:
                change = 100.0 if curr > 0 else 0.0
                
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
            if prev > 0:
                change = ((curr - prev) / prev) * 100
            else:
                change = 100.0 if curr > 0 else 0.0
                
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


# Singleton Management
_insights_job: Optional[SparkInsightsJob] = None
_cached_results: Optional[Dict[str, Any]] = None

def get_insights_job_instance() -> SparkInsightsJob:
    global _insights_job
    if _insights_job is None:
        _insights_job = SparkInsightsJob()
    return _insights_job

def warmup_spark():
    try:
        get_insights_job_instance()._init_spark()
    except: pass

def get_spark_insights(force_refresh: bool = False) -> Dict[str, Any]:
    """
    API Access Point: Reads pre-trained metrics from filesystem.
    Does NOT trigger training (which is heavy).
    """
    results = {
        "success": True, 
        "profiles": [], 
        "rules": [], 
        "trends": {"weekly": [], "monthly": []}, 
        "anomalies": []
    }
    
    try:
        # Load Artifacts
        path_profiles = os.path.join(ARTIFACTS_DIR, "camera_profiles.json")
        if os.path.exists(path_profiles):
            with open(path_profiles) as f: results["profiles"] = json.load(f)
            
        path_rules = os.path.join(ARTIFACTS_DIR, "risk_rules.json")
        if os.path.exists(path_rules):
            with open(path_rules) as f: results["rules"] = json.load(f)
            
        # Note: Trends are dynamic or saved? Ideally saved.
        # For this turn, we return empty stats or we could save stats.json in run() too.
        
        return results
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        return {"success": False, "error": "Insights not available. Run offline training."}

if __name__ == "__main__":
    job = SparkInsightsJob()
    res = job.run()
    print(json.dumps(res, indent=2, default=str))
