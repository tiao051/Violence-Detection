import React, { FC, useRef, useState, useEffect } from 'react'
import WebRTCService from '../services/WebRTCService'
import { MetadataBuffer } from '../utils/overlay'
import './VideoDisplay.css'

interface VideoDisplayProps {
  signalingUrl?: string
}

interface DetectionMetadata {
  frame_seq: number
  timestamp: string
  violence: boolean
  confidence: number
}

/**
 * VideoDisplay Component - Main component for video streaming and overlay
 * Handles WebRTC connection, video display, and real-time metadata visualization
 */
const VideoDisplay: FC<VideoDisplayProps> = ({
  signalingUrl = 'ws://localhost:8000/ws/threats',
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [connected, setConnected] = useState<boolean>(false)
  const [metadata, setMetadata] = useState<DetectionMetadata | null>(null)
  const [fps, setFps] = useState<number>(0)
  const metadataBufferRef = useRef<MetadataBuffer>(new MetadataBuffer())

  useEffect(() => {
    // Initialize WebRTC Service
    const webrtcService = WebRTCService.getInstance(signalingUrl)

    // Set up event listeners
    const handleConnectionChange = (connected: boolean) => {
      setConnected(connected)
      console.log(`WebRTC connection: ${connected ? 'established' : 'closed'}`)
    }

    const handleMetadata = (newMetadata: DetectionMetadata) => {
      // Add to buffer
      metadataBufferRef.current.add(newMetadata)

      // Update state with latest metadata
      setMetadata(newMetadata)

      // Update FPS counter
      const buffer = metadataBufferRef.current
      if (buffer.getSize() % 10 === 0) {
        setFps(Math.round(buffer.calculateFPS()))
      }
    }

    webrtcService.onConnectionChange(handleConnectionChange)
    webrtcService.onMetadata(handleMetadata)

    // Connect to video stream
    if (videoRef.current) {
      webrtcService.connect(videoRef.current).catch((error) => {
        console.error('Failed to connect to WebRTC service:', error)
      })
    }

    // Store service reference for cleanup
    const service = webrtcService

    // Cleanup
    return () => {
      service.disconnect()
    }
  }, [signalingUrl])

  const connectionStatusClass = `connection-status ${connected ? 'connected' : 'disconnected'}`
  const confidenceBarClass = `confidence-bar ${metadata?.violence ? 'danger' : 'safe'}`
  const violenceText = metadata?.violence ? '🚨 YES' : '✓ NO'
  const confidencePercent = metadata ? (metadata.confidence * 100).toFixed(2) : '0'
  const violenceMetadataClass = `value ${metadata?.violence ? 'violence-detected' : ''}`

  return React.createElement(
    'div',
    { className: 'video-display-container' },
    React.createElement(
      'div',
      { className: 'video-wrapper' },
      React.createElement('video', {
        ref: videoRef,
        autoPlay: true,
        muted: true,
        playsInline: true,
        className: 'video-stream',
      }),
      React.createElement('canvas', { ref: canvasRef, className: 'overlay-canvas' }),
      React.createElement(
        'div',
        { className: connectionStatusClass },
        React.createElement('div', { className: 'status-dot' }),
        React.createElement('span', null, connected ? 'Connected' : 'Disconnected')
      ),
      fps > 0 && React.createElement('div', { className: 'fps-counter' }, `FPS: ${fps}`)
    ),
    metadata &&
      React.createElement(
        'div',
        { className: 'detection-panel' },
        React.createElement('h3', null, 'Detection Information'),
        React.createElement(
          'div',
          { className: 'metadata-grid' },
          React.createElement(
            'div',
            { className: 'metadata-item' },
            React.createElement('span', { className: 'label' }, 'Frame:'),
            React.createElement('span', { className: 'value' }, metadata.frame_seq)
          ),
          React.createElement(
            'div',
            { className: 'metadata-item' },
            React.createElement('span', { className: 'label' }, 'Timestamp:'),
            React.createElement('span', { className: 'value' }, metadata.timestamp)
          ),
          React.createElement(
            'div',
            { className: 'metadata-item' },
            React.createElement('span', { className: 'label' }, 'Violence:'),
            React.createElement('span', { className: violenceMetadataClass }, violenceText)
          ),
          React.createElement(
            'div',
            { className: 'metadata-item' },
            React.createElement('span', { className: 'label' }, 'Confidence:'),
            React.createElement('span', { className: 'value' }, `${confidencePercent}%`)
          )
        ),
        React.createElement(
          'div',
          { className: 'confidence-bar-container' },
          React.createElement('div', {
            className: confidenceBarClass,
            style: { width: `${metadata.confidence * 100}%` },
          })
        )
      ),
    !connected &&
      React.createElement(
        'div',
        { className: 'connection-message' },
        React.createElement('p', null, 'Connecting to WebRTC stream...')
      )
  )
}

export default VideoDisplay
