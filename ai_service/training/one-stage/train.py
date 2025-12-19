"""
Training script for Insights Model (One-Stage Analysis).

This script trains the unified insights model (Clustering + Association Rules + Risk Prediction)
using data stored in HDFS.

Pipeline:
1. Initialize Spark Session
2. Load violence events from HDFS (Parquet/CSV)
3. Convert to ViolenceEvent objects
4. Train InsightsModel
5. Save model locally and to HDFS
"""

import os
import logging
import argparse
from datetime import datetime
from typing import List
import joblib

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp

from ai_service.insights.core.model import InsightsModel
from ai_service.insights.core.schema import ViolenceEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def get_spark_session(app_name: str = "ViolenceInsightsTraining") -> SparkSession:
    """Create or get SparkSession."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.session.timeZone", "Asia/Ho_Chi_Minh") \
        .getOrCreate()


def load_events_from_hdfs(spark: SparkSession, hdfs_path: str) -> List[ViolenceEvent]:
    """
    Read events from HDFS and convert to ViolenceEvent objects.
    
    Args:
        spark: SparkSession
        hdfs_path: Path to data in HDFS (e.g., hdfs://namenode:9000/data/events)
        
    Returns:
        List of ViolenceEvent objects
    """
    logger.info(f"Reading data from HDFS: {hdfs_path}")
    
    # Read Parquet or CSV (auto-detect or try both)
    try:
        df = spark.read.parquet(hdfs_path)
    except Exception:
        logger.warning("Failed to read as Parquet, trying CSV...")
        df = spark.read.option("header", "true").csv(hdfs_path)

    # Ensure timestamp is correct type
    if dict(df.dtypes)['timestamp'] == 'string':
        df = df.withColumn("timestamp", to_timestamp(col("timestamp")))

    # Collect to driver (InsightsModel runs locally on collected data)
    # Note: This assumes data fits in memory. For massive datasets, 
    # we would need to implement distributed training logic in InsightsModel.
    rows = df.collect()
    logger.info(f"Loaded {len(rows)} events from HDFS")
    
    events = []
    for row in rows:
        try:
            event = ViolenceEvent(
                user_id=row.user_id if 'user_id' in row else "unknown",
                camera_id=row.camera_id,
                camera_name=row.camera_name if 'camera_name' in row else row.camera_id,
                camera_description=row.camera_description if 'camera_description' in row else "",
                timestamp=row.timestamp,
                confidence=float(row.confidence),
                video_url=row.video_url if 'video_url' in row else "",
                thumbnail_url=row.thumbnail_url if 'thumbnail_url' in row else "",
                label=row.label if 'label' in row else "violence"
            )
            events.append(event)
        except Exception as e:
            logger.warning(f"Skipping invalid row: {e}")
            continue
            
    return events


def train_and_save(
    hdfs_input_path: str,
    model_output_path: str,
    hdfs_model_path: str = None
):
    """
    Main training pipeline.
    """
    spark = get_spark_session()
    
    try:
        # 1. Load Data
        events = load_events_from_hdfs(spark, hdfs_input_path)
        
        if not events:
            logger.error("No events loaded. Exiting.")
            return
        
        # 2. Initialize Model
        model = InsightsModel(
            n_clusters=3,
            min_support=0.05,
            min_confidence=0.5,
            n_estimators=100
        )
        
        # 3. Train
        logger.info("Starting model training...")
        model.fit(events)
        logger.info("Training complete.")
        
        # 4. Save Locally
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        logger.info(f"Model saved locally to: {model_output_path}")
        
        # 5. Save to HDFS (Optional)
        if hdfs_model_path:
            logger.info(f"Saving model to HDFS: {hdfs_model_path}")
            # Use Hadoop FileSystem API via Spark context to copy file
            Path = spark._gateway.jvm.org.apache.hadoop.fs.Path
            fs = spark._jsc.hadoopConfiguration()
            fs = Path(hdfs_model_path).getFileSystem(fs)
            
            fs.copyFromLocalFile(
                False, # delSrc
                True,  # overwrite
                Path(model_output_path),
                Path(hdfs_model_path)
            )
            logger.info("Model uploaded to HDFS successfully.")
            
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Insights Model on HDFS Data")
    parser.add_argument("--input", required=True, help="HDFS input path (parquet/csv)")
    parser.add_argument("--output", default="trained_model.pkl", help="Local output path")
    parser.add_argument("--hdfs-output", help="HDFS output path for model (optional)")
    
    args = parser.parse_args()
    
    train_and_save(args.input, args.output, args.hdfs_output)
