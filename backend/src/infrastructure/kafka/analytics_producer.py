"""
Kafka Analytics Producer

Reads nonviolence events from HDFS and produces to Kafka topic
for analytics model inference.

This creates a pipeline:
HDFS (nonviolence_events.csv) -> Kafka (analytics-events topic) -> InsightsModel

Usage:
    python kafka_analytics_producer.py
    
Prerequisites:
    - Kafka must be running
    - HDFS must have nonviolence_events.csv uploaded
"""

import os
import json
import subprocess
import tempfile
from typing import List, Dict, Any
from kafka import KafkaProducer
from kafka.errors import KafkaError
import pandas as pd


class AnalyticsProducer:
    """
    Kafka producer for analytics events.
    
    Reads events from HDFS and publishes to Kafka topic.
    """
    
    TOPIC = "analytics-events"
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        """
        Initialize the analytics producer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
        """
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        
    def connect(self) -> bool:
        """Connect to Kafka broker."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
            )
            print(f"Connected to Kafka at {self.bootstrap_servers}")
            return True
        except KafkaError as e:
            print(f"Failed to connect to Kafka: {e}")
            return False
    
    def download_from_hdfs(self, hdfs_path: str) -> str:
        """
        Download file from HDFS to local temp file.
        
        Args:
            hdfs_path: Path in HDFS
            
        Returns:
            Path to local temp file
        """
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            local_path = tmp.name
        
        container_path = "/tmp/downloaded.csv"
        
        try:
            # Download from HDFS to container
            get_cmd = [
                "docker", "exec", "hdfs-namenode",
                "hdfs", "dfs", "-get", hdfs_path, container_path
            ]
            print(f"Downloading from HDFS: {hdfs_path}")
            subprocess.run(get_cmd, check=True)
            
            # Copy from container to local
            copy_cmd = [
                "docker", "cp",
                f"hdfs-namenode:{container_path}",
                local_path
            ]
            subprocess.run(copy_cmd, check=True)
            
            print(f"Downloaded to: {local_path}")
            return local_path
            
        except subprocess.CalledProcessError as e:
            print(f"Error downloading from HDFS: {e}")
            return ""
    
    def publish_events(self, events: List[Dict[str, Any]]) -> int:
        """
        Publish events to Kafka topic.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Number of events published successfully
        """
        if not self.producer:
            print("Producer not connected")
            return 0
        
        success_count = 0
        
        for event in events:
            try:
                # Use cameraId as partition key
                key = event.get("cameraId", "unknown")
                
                future = self.producer.send(
                    self.TOPIC,
                    key=key,
                    value=event
                )
                
                # Wait for confirmation
                future.get(timeout=10)
                success_count += 1
                
            except KafkaError as e:
                print(f"Error publishing event: {e}")
        
        # Flush all messages
        self.producer.flush()
        
        return success_count
    
    def process_from_hdfs(self, hdfs_path: str = "/analytics/nonviolence_events.csv") -> int:
        """
        Download CSV from HDFS and publish all events to Kafka.
        
        Args:
            hdfs_path: Path to CSV in HDFS
            
        Returns:
            Number of events published
        """
        # Download from HDFS
        local_path = self.download_from_hdfs(hdfs_path)
        if not local_path:
            return 0
        
        try:
            # Read CSV
            df = pd.read_csv(local_path)
            print(f"Read {len(df)} events from CSV")
            
            # Convert to list of dicts
            events = df.to_dict(orient='records')
            
            # Publish to Kafka
            print(f"Publishing {len(events)} events to Kafka topic '{self.TOPIC}'...")
            published = self.publish_events(events)
            
            print(f"Published {published}/{len(events)} events successfully")
            return published
            
        finally:
            # Cleanup
            if os.path.exists(local_path):
                os.remove(local_path)
    
    def close(self):
        """Close the producer connection."""
        if self.producer:
            self.producer.close()
            print("Producer closed")


def main():
    print("=" * 60)
    print("Kafka Analytics Producer")
    print("=" * 60)
    
    # Get Kafka bootstrap servers from environment or use default
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    
    producer = AnalyticsProducer(bootstrap_servers)
    
    if not producer.connect():
        print("Failed to connect to Kafka. Make sure Kafka is running.")
        return
    
    try:
        # Process events from HDFS
        published = producer.process_from_hdfs()
        
        if published > 0:
            print(f"\n✅ Successfully published {published} events to Kafka")
        else:
            print("\n❌ No events published")
            
    finally:
        producer.close()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
