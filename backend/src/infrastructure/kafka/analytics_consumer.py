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
import logging
from typing import List, Dict, Any
from aiokafka import AIOKafkaConsumer
import requests
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hdfs_upload.log'),
        logging.StreamHandler()
    ]
)

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
        
        # Stats tracking
        self.total_messages = 0
        self.total_flushes = 0
        self.total_uploaded = 0
        self.upload_errors = 0
        
        logger.info("=" * 80)
        logger.info("HDFS Analytics Consumer initialized")
        logger.info(f"  Kafka: {self.bootstrap_servers}")
        logger.info(f"  Topic: {self.topic}")
        logger.info(f"  HDFS: {self.hdfs_url}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Batch timeout: {self.batch_timeout}s")
        logger.info("=" * 80)

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
        logger.info(f"Connected to Kafka topic: {self.topic}")
        
        try:
            while self.is_running:
                try:
                    # Wait for message with timeout to allow periodic flush
                    msg = await asyncio.wait_for(self.consumer.getone(), timeout=1.0)
                    self.buffer.append(msg.value)
                    self.total_messages += 1
                    
                    # Log every 100 messages
                    if self.total_messages % 100 == 0:
                        logger.info(f"Consumed {self.total_messages} messages (buffer size: {len(self.buffer)})")
                        
                except asyncio.TimeoutError:
                    pass
                
                if len(self.buffer) >= self.batch_size or \
                   (time.time() - self.last_flush_time > self.batch_timeout and self.buffer):
                    await self._flush_to_hdfs_with_logging()
                    
        except Exception as e:
            logger.error(f"Error consuming from Kafka: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        self.is_running = False
        if self.consumer:
            await self.consumer.stop()
        logger.info("Analytics Consumer stopped")
            
    async def _flush_to_hdfs_with_logging(self) -> bool:
        """Flush buffer to HDFS with detailed logging."""
        if not self.buffer:
            return True
        
        buffer_size = len(self.buffer)
        timestamp = int(time.time())
        
        try:
            # Create DataFrame
            df = pd.DataFrame(self.buffer)
            csv_content = df.to_csv(index=False).encode('utf-8')
            
            # Prepare HDFS path
            filename = f"events_{timestamp}.csv"
            hdfs_path = f"/analytics/raw/{filename}"
            
            # Count violence vs non-violence
            violence_count = (df['label'] == 'violence').sum() if 'label' in df.columns else 0
            nonviolence_count = (df['label'] == 'nonviolence').sum() if 'label' in df.columns else 0
            
            logger.info("-" * 80)
            logger.info(f"HDFS FLUSH ATTEMPT #{self.total_flushes + 1}")
            logger.info(f"  Timestamp: {datetime.now().isoformat()}")
            logger.info(f"  Buffer size: {buffer_size} messages")
            logger.info(f"  Data size: {len(csv_content):,} bytes ({len(csv_content) / 1024 / 1024:.2f} MB)")
            logger.info(f"  Violence events: {violence_count}")
            logger.info(f"  Non-violence events: {nonviolence_count}")
            logger.info(f"  Target HDFS path: {hdfs_path}")
            logger.info(f"  CSV preview (first 2 rows):")
            preview = df.head(2).to_string()
            for line in preview.split('\n'):
                logger.info(f"    {line}")
            
            # Upload to HDFS
            if self._upload_to_hdfs(csv_content, hdfs_path):
                self.total_flushes += 1
                self.total_uploaded += buffer_size
                logger.info(f"  ✓ Successfully uploaded!")
                logger.info(f"  Total messages uploaded so far: {self.total_uploaded:,}")
                logger.info("-" * 80)
                
                self.buffer = []
                self.last_flush_time = time.time()
                
                # Commit offset after successful upload
                try:
                    await self.consumer.commit()
                except:
                    pass
                
                return True
            else:
                self.upload_errors += 1
                logger.warning(f"  ✗ Upload failed, keeping buffer")
                logger.warning(f"  Failed uploads so far: {self.upload_errors}")
                logger.info("-" * 80)
                return False
                
        except Exception as e:
            self.upload_errors += 1
            logger.error(f"  ✗ Error flushing to HDFS: {e}", exc_info=True)
            logger.info("-" * 80)
            return False
            
    async def flush_to_hdfs(self) -> bool:
        """Legacy method for compatibility."""
        return await self._flush_to_hdfs_with_logging()

    def _upload_to_hdfs(self, content: bytes, hdfs_path: str) -> bool:
        try:
            # Create parent directory if needed
            parent_dir = os.path.dirname(hdfs_path)
            logger.debug(f"Creating HDFS directory: {parent_dir}")
            
            mkdir_resp = requests.put(
                f"{self.hdfs_url}/webhdfs/v1{parent_dir}?op=MKDIRS&permission=755"
            )
            
            if mkdir_resp.status_code not in [200, 201, 409]:
                logger.warning(f"Directory creation returned {mkdir_resp.status_code}")
            
            # Create file
            url = f"{self.hdfs_url}/webhdfs/v1{hdfs_path}?op=CREATE&overwrite=true"
            logger.debug(f"Initiating HDFS upload to {hdfs_path}")
            
            resp = requests.put(url, allow_redirects=False)
            
            if resp.status_code == 307:
                redirect_url = resp.headers.get('Location')
                if redirect_url:
                    logger.debug(f"Following redirect to datanode: {redirect_url}")
                    resp = requests.put(redirect_url, data=content)
                    
                    if resp.status_code == 201:
                        logger.debug(f"DataNode confirmed write (201)")
                        return True
                    else:
                        logger.error(f"DataNode write failed ({resp.status_code})")
                        return False
            
            logger.error(f"Unexpected status code from NameNode: {resp.status_code}")
            return False
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to HDFS: {e}")
            logger.error(f"Is HDFS running at {self.hdfs_url}?")
            return False
        except Exception as e:
            logger.error(f"HDFS upload error: {e}", exc_info=True)
            return False

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Starting HDFS Analytics Consumer")
    logger.info("=" * 80)
    
    consumer = AnalyticsConsumer()
    try:
        asyncio.run(consumer.start())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    finally:
        logger.info("=" * 80)
        logger.info("FINAL STATISTICS:")
        logger.info(f"  Total messages consumed: {consumer.total_messages:,}")
        logger.info(f"  Total flushes to HDFS: {consumer.total_flushes:,}")
        logger.info(f"  Total messages uploaded: {consumer.total_uploaded:,}")
        logger.info(f"  Upload errors: {consumer.upload_errors}")
        logger.info("=" * 80)
