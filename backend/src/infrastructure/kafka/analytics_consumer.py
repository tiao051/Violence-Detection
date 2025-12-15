"""
HDFS Result Consumer

Consumes inference results (violence and non-violence) from Kafka
and uploads them to HDFS for analytics.

Flow: Kafka (inference-results) -> Buffer -> HDFS (CSV)
"""

import os
import json
import time
import asyncio
import pandas as pd
from typing import List, Dict, Any
from aiokafka import AIOKafkaConsumer
import requests

class AnalyticsConsumer:
    def __init__(self, 
                 bootstrap_servers: str = None,
                 topic: str = None,
                 hdfs_url: str = None,
                 batch_size: int = 5000,
                 batch_timeout: int = 300):
        self.bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        if not self.bootstrap_servers:
            raise ValueError("KAFKA_BOOTSTRAP_SERVERS is required")

        self.topic = topic or os.getenv("KAFKA_RESULT_TOPIC")
        if not self.topic:
            raise ValueError("KAFKA_RESULT_TOPIC is required")

        self.hdfs_url = hdfs_url or os.getenv("HDFS_NAMENODE_URL")
        if not self.hdfs_url:
            raise ValueError("HDFS_NAMENODE_URL is required")

        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        self.buffer: List[Dict[str, Any]] = []
        self.last_flush_time = time.time()
        self.consumer = None
        self.is_running = False

    async def start(self):
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id="hdfs-archiver",
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        await self.consumer.start()
        self.is_running = True
        print(f"Started HDFS Consumer for topic {self.topic}")
        
        try:
            while self.is_running:
                try:
                    # Wait for message with timeout to allow periodic flush
                    msg = await asyncio.wait_for(self.consumer.getone(), timeout=1.0)
                    self.buffer.append(msg.value)
                except asyncio.TimeoutError:
                    pass
                
                if len(self.buffer) >= self.batch_size or \
                   (time.time() - self.last_flush_time > self.batch_timeout and self.buffer):
                    if await self.flush_to_hdfs():
                        await self.consumer.commit()
                    
        except Exception as e:
            print(f"Error consuming: {e}")
        finally:
            await self.stop()

    async def stop(self):
        self.is_running = False
        if self.consumer:
            await self.consumer.stop()
            
    async def flush_to_hdfs(self) -> bool:
        if not self.buffer:
            return True
            
        try:
            df = pd.DataFrame(self.buffer)
            timestamp = int(time.time())
            filename = f"events_{timestamp}.csv"
            hdfs_path = f"/analytics/raw/{filename}"
            
            csv_content = df.to_csv(index=False).encode('utf-8')
            
            # Upload to HDFS
            if self._upload_to_hdfs(csv_content, hdfs_path):
                print(f"Uploaded {len(self.buffer)} events to {hdfs_path}")
                self.buffer = []
                self.last_flush_time = time.time()
                return True
            else:
                print("Failed to upload to HDFS, keeping buffer")
                return False
                
        except Exception as e:
            print(f"Error flushing to HDFS: {e}")
            return False

    def _upload_to_hdfs(self, content: bytes, hdfs_path: str) -> bool:
        try:
            # Create parent directory if needed
            parent_dir = os.path.dirname(hdfs_path)
            requests.put(f"{self.hdfs_url}/webhdfs/v1{parent_dir}?op=MKDIRS&permission=755")
            
            # Create file
            url = f"{self.hdfs_url}/webhdfs/v1{hdfs_path}?op=CREATE&overwrite=true"
            resp = requests.put(url, allow_redirects=False)
            
            if resp.status_code == 307:
                redirect_url = resp.headers['Location']
                resp = requests.put(redirect_url, data=content)
                return resp.status_code == 201
                
            return False
        except Exception as e:
            print(f"HDFS upload error: {e}")
            return False

if __name__ == "__main__":
    consumer = AnalyticsConsumer()
    asyncio.run(consumer.start())
