import { useState, useRef, useCallback } from 'react'
import { Detection, DetectionMetadata } from '../types/detection'

export const useDetectionLog = () => {
  const [detections, setDetections] = useState<Detection[]>([])
  const detectionCounterRef = useRef<number>(0)

  const addDetection = useCallback((metadata: DetectionMetadata) => {
    const detection: Detection = {
      id: `detection-${detectionCounterRef.current++}`,
      timestamp: metadata.timestamp || new Date().toISOString(),
      violence: metadata.violence,
      camera: metadata.camera_id || 'Unknown',
      confidence: metadata.confidence,
    }

    setDetections((prev) => [detection, ...prev.slice(0, 19)]) // Keep last 20
  }, [])

  return { detections, addDetection }
}
