import React, { FC, useRef, useState, useEffect } from 'react'
import WebRTCService from '../../services/WebRTCService'
import { MetadataBuffer } from '../../utils/overlay'
import StatusBadge from '../StatusBadge'
import DetectionLog from '../DetectionLog'
import { useDetectionLog } from '../../hooks/useDetectionLog'
import { VideoDisplayProps, DetectionMetadata } from '../../types/detection'
import './VideoDisplay.css'

/**
 * VideoDisplay Component - Ultra Minimalist Dashboard
 * Displays video stream, status, and detection log
 */
const VideoDisplay: FC<VideoDisplayProps> = ({
  signalingUrl = 'ws://localhost:8000/ws/threats',
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [connected, setConnected] = useState<boolean>(false)
  const [metadata, setMetadata] = useState<DetectionMetadata | null>(null)
  const { detections, addDetection } = useDetectionLog()
  const metadataBufferRef = useRef<MetadataBuffer>(new MetadataBuffer())

  useEffect(() => {
    const webrtcService = WebRTCService.getInstance(signalingUrl)

    const handleConnectionChange = (connected: boolean) => {
      setConnected(connected)
      console.log(`WebSocket connection: ${connected ? 'established' : 'closed'}`)
    }

    const handleMetadata = (newMetadata: DetectionMetadata) => {
      metadataBufferRef.current.add(newMetadata)
      setMetadata(newMetadata)
      addDetection(newMetadata)
    }

    webrtcService.onConnectionChange(handleConnectionChange)
    webrtcService.onMetadata(handleMetadata)

    if (videoRef.current) {
      webrtcService.connect(videoRef.current).catch((error) => {
        console.error('Failed to connect to WebSocket service:', error)
      })
    }

    const service = webrtcService

    return () => {
      service.disconnect()
    }
  }, [signalingUrl])

  return React.createElement(
    'div',
    { className: 'video-display' },
    // Video Container
    React.createElement(
      'div',
      { className: 'video-container' },
      React.createElement('video', {
        ref: videoRef,
        autoPlay: true,
        muted: true,
        playsInline: true,
        className: 'video-element',
      }),
      React.createElement('canvas', { ref: canvasRef, className: 'canvas-overlay' }),

      // Status Indicator (top right)
      React.createElement(
        'div',
        { className: 'video-status-indicator' },
        React.createElement(
          'div',
          { className: `status-dot ${connected ? 'connected' : 'disconnected'}` },
          React.createElement('span', null, connected ? '●' : '○')
        ),
        React.createElement(
          'span',
          { className: 'status-text text-xs text-secondary' },
          connected ? 'Connected' : 'Disconnected'
        )
      )
    ),

    // Metadata Section
    metadata &&
      React.createElement(
        'div',
        { className: 'metadata-section' },
        React.createElement(StatusBadge, {
          violence: metadata.violence,
          confidence: metadata.confidence,
        })
      ),

    // Detection Log Section
    React.createElement(DetectionLog, { detections })
  )
}

export default VideoDisplay
