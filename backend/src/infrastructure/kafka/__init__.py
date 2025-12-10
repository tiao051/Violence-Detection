"""Kafka infrastructure module."""
from .producer import get_kafka_producer, KafkaFrameProducer

__all__ = ['get_kafka_producer', 'KafkaFrameProducer']