"""
Spark Streaming Job for Real-time Violence Detection.

Consumes frames from Kafka, distributes inference across Spark cluster,
and publishes results back to Kafka + Redis.

Architecture:
- Spark Streaming consumes micro-batches from Kafka
- Each batch is distributed across worker nodes
- Results aggregated and published to downstream systems
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, struct, to_json
from pyspark.sql.types import StructType, StructField, StringType, BinaryType, DoubleType

logger = logging.getLogger(__name__)


class SparkStreamingJob:
    """
    Spark Streaming job for violence detection.
    
    Reads frames from Kafka topic, processes them using distributed inference,
    and writes results back to Kafka + Redis.
    """
    
    def __init__(
        self,
        kafka_servers: str = "localhost:9092",
        input_topic: str = "frames",
        output_topic: str = "inference-results",
        batch_interval_sec: int = 2,
        n_workers: int = 4,
        model_path: str = None,
        device: str = "cpu",
        redis_url: str = None,
        alert_confidence_threshold: float = 0.85,
        checkpoint_dir: str = "/tmp/spark-checkpoint",
    ):
        """
        Initialize Spark Streaming job.
        
        Args:
            kafka_servers: Kafka broker address (host:port,host:port,...)
            input_topic: Kafka topic to consume frames from
            output_topic: Kafka topic to publish results to
            batch_interval_sec: Micro-batch interval in seconds
            n_workers: Number of worker nodes
            model_path: Path to trained violence detection model
            device: 'cpu' or 'cuda'
            redis_url: Redis URL for alert publishing
            alert_confidence_threshold: Confidence threshold for alerts
            checkpoint_dir: Directory for Spark checkpoint state
        """
        if not HAS_SPARK:
            raise ImportError("PySpark is required. Install with: pip install pyspark")
        
        self.kafka_servers = kafka_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.batch_interval_sec = batch_interval_sec
        self.n_workers = n_workers
        self.model_path = model_path
        self.device = device
        self.redis_url = redis_url
        self.alert_confidence_threshold = alert_confidence_threshold
        self.checkpoint_dir = checkpoint_dir
        
        # Runtime state
        self.spark: Optional[SparkSession] = None
        self.is_running = False
        self.start_time = None
        
        # Metrics
        self.metrics = {
            "batches_processed": 0,
            "frames_processed": 0,
            "alerts_sent": 0,
        }
    
    def start(self) -> None:
        """Start the Spark Streaming job."""
        logger.info("Starting Spark Streaming job")
        
        try:
            # Create Spark session
            self.spark = SparkSession.builder \
                .appName("ViolenceDetectionStreaming") \
                .master(f"local[{self.n_workers}]") \
                .config("spark.streaming.kafka.maxRatePerPartition", "100") \
                .config("spark.sql.streaming.minBatchesToRetain", "2") \
                .config("spark.sql.streaming.checkpointLocation", self.checkpoint_dir) \
                .getOrCreate()
            
            self.spark.sparkContext.setLogLevel("WARN")
            
            # Setup streaming pipeline
            self._setup_streaming_pipeline()
            
            self.is_running = True
            self.start_time = datetime.now()
            
            logger.info("Spark Streaming job started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Spark Streaming job: {e}", exc_info=True)
            raise
    
    def stop(self) -> None:
        """Stop the Spark Streaming job."""
        logger.info("Stopping Spark Streaming job")
        
        if self.spark:
            self.spark.stop()
        
        self.is_running = False
        
        logger.info("Spark Streaming job stopped")
    
    def _setup_streaming_pipeline(self) -> None:
        """Setup the streaming pipeline: Kafka -> Process -> Kafka + Redis."""
        
        # Schema for incoming frame messages from Kafka
        frame_schema = StructType([
            StructField("camera_id", StringType(), True),
            StructField("frame_id", StringType(), True),
            StructField("frame_seq", BinaryType(), True),  # Encoded frame data
            StructField("timestamp", DoubleType(), True),
        ])
        
        # Read from Kafka
        df_frames = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_servers) \
            .option("subscribe", self.input_topic) \
            .option("startingOffsets", "latest") \
            .option("maxOffsetsPerTrigger", 100) \
            .load()
        
        logger.info(f"Subscribed to Kafka topic: {self.input_topic}")
        
        # Parse frame messages
        df_parsed = df_frames.select(
            from_json(col("value").cast("string"), frame_schema).alias("data")
        ).select("data.*")
        
        # Process frames through inference
        df_results = self._add_inference_udf(df_parsed)
        
        # Filter and prepare alert messages
        df_alerts = df_results.filter(
            col("confidence") >= self.alert_confidence_threshold
        ).select(
            to_json(struct(
                col("frame_id"),
                col("camera_id"),
                col("timestamp"),
                col("confidence"),
                col("processing_time_ms"),
            )).alias("value")
        )
        
        # Write results to output Kafka topic
        query = df_alerts.writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_servers) \
            .option("topic", self.output_topic) \
            .option("checkpointLocation", f"{self.checkpoint_dir}/results") \
            .outputMode("append") \
            .start()
        
        logger.info(f"Writing results to Kafka topic: {self.output_topic}")
        
        # Optional: Write to console for debugging
        console_query = df_results.writeStream \
            .format("console") \
            .option("truncate", False) \
            .option("checkpointLocation", f"{self.checkpoint_dir}/console") \
            .start()
        
        # Await termination
        self.spark.streams.awaitAnyTermination()
    
    def _add_inference_udf(self, df):
        """
        Add inference UDF to dataframe.
        
        This is a placeholder - in production, would use PySpark UDF
        or Spark MLlib model serving.
        
        Args:
            df: Input dataframe with frames
            
        Returns:
            Dataframe with inference results
        """
        # In production: Define UDF that calls model inference
        # For now, return df with mock results
        
        from pyspark.sql.functions import lit, rand, when
        
        df_results = df.select(
            col("frame_id"),
            col("camera_id"),
            col("timestamp"),
            # Mock inference results (in production: call actual model)
            when(rand() > 0.9, 1).otherwise(0).alias("is_violence"),
            (rand() * 0.5 + 0.5).alias("confidence"),
            lit(25.0).alias("processing_time_ms"),
        )
        
        return df_results
    
    def get_status(self) -> Dict[str, Any]:
        """Get streaming job status."""
        return {
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time else 0
            ),
            "metrics": self.metrics,
        }


__all__ = ["SparkStreamingJob"]
