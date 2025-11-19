/**
 * Simplified WebSocket Service for Threat Detection
 * Connects to backend WebSocket endpoint to receive real-time detection metadata
 */

interface DetectionMetadata {
  frame_seq: number
  timestamp: string
  violence: boolean
  confidence: number
  camera_id?: string
}

type OnConnectionChangeCallback = (connected: boolean) => void
type OnMetadataCallback = (metadata: DetectionMetadata) => void

class WebRTCService {
  private static instance: WebRTCService | null = null
  private signalingUrl: string
  private signalingSocket: WebSocket | null = null
  private connected: boolean = false
  private reconnectAttempts: number = 0
  private maxReconnectAttempts: number = 5
  private reconnectDelay: number = 2000

  private onConnectionChangeCallbacks: OnConnectionChangeCallback[] = []
  private onMetadataCallbacks: OnMetadataCallback[] = []

  private constructor(signalingUrl: string) {
    this.signalingUrl = signalingUrl
  }

  /**
   * Get singleton instance
   */
  public static getInstance(signalingUrl: string = 'ws://localhost:8000/ws/threats'): WebRTCService {
    if (!WebRTCService.instance) {
      WebRTCService.instance = new WebRTCService(signalingUrl)
    }
    return WebRTCService.instance
  }

  /**
   * Register callback for connection state changes
   */
  public onConnectionChange(callback: OnConnectionChangeCallback): void {
    this.onConnectionChangeCallbacks.push(callback)
  }

  /**
   * Register callback for metadata reception
   */
  public onMetadata(callback: OnMetadataCallback): void {
    this.onMetadataCallbacks.push(callback)
  }

  /**
   * Emit connection state change
   */
  private emitConnectionChange(connected: boolean): void {
    this.connected = connected
    this.onConnectionChangeCallbacks.forEach((callback) => callback(connected))
  }

  /**
   * Emit metadata received
   */
  private emitMetadata(metadata: DetectionMetadata): void {
    this.onMetadataCallbacks.forEach((callback) => callback(metadata))
  }

  /**
   * Connect to WebSocket threat detection endpoint
   */
  public async connect(_videoElement: HTMLVideoElement): Promise<void> {
    try {
      await this.connectWebSocket()
      console.log('WebRTC service connected')
    } catch (error) {
      console.error('Error connecting to threat detection service:', error)
      this.emitConnectionChange(false)
      throw error
    }
  }

  /**
   * Connect to WebSocket endpoint
   */
  private connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log(`Connecting to WebSocket: ${this.signalingUrl}`)
        this.signalingSocket = new WebSocket(this.signalingUrl)

        this.signalingSocket.onopen = () => {
          console.log('Connected to threat detection WebSocket')
          this.emitConnectionChange(true)
          this.reconnectAttempts = 0
          
          // Send ping to server to keep connection alive
          if (this.signalingSocket && this.signalingSocket.readyState === WebSocket.OPEN) {
            this.signalingSocket.send(JSON.stringify({ type: 'ping' }))
          }
          
          resolve()
        }

        this.signalingSocket.onmessage = (event: MessageEvent) => {
          try {
            const data = JSON.parse(event.data)

            // Handle detection metadata
            if (data.type === 'detection') {
              const metadata: DetectionMetadata = {
                frame_seq: data.frame_seq,
                timestamp: data.timestamp,
                violence: data.violence,
                confidence: data.confidence,
                camera_id: data.camera_id,
              }
              this.emitMetadata(metadata)
            } else if (data.type === 'pong') {
              console.log('Pong received')
            } else if (data.type === 'error') {
              console.error('Server error:', data.message)
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error)
          }
        }

        this.signalingSocket.onerror = (event: Event) => {
          console.error('WebSocket error:', event)
          this.emitConnectionChange(false)
          reject(new Error('WebSocket connection failed'))
        }

        this.signalingSocket.onclose = () => {
          console.log('WebSocket closed')
          this.emitConnectionChange(false)
          this.attemptReconnect()
        }
      } catch (error) {
        console.error('Error creating WebSocket:', error)
        reject(error)
      }
    })
  }

  /**
   * Attempt to reconnect with exponential backoff
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`)

      setTimeout(() => {
        this.connectWebSocket().catch((error) => {
          console.error('Reconnection failed:', error)
        })
      }, delay)
    } else {
      console.error('Max reconnection attempts reached')
      this.emitConnectionChange(false)
    }
  }

  /**
   * Disconnect from WebSocket
   */
  public disconnect(): void {
    console.log('Disconnecting from threat detection service')

    if (this.signalingSocket) {
      this.signalingSocket.close()
      this.signalingSocket = null
    }

    this.emitConnectionChange(false)
  }

  /**
   * Get connection status
   */
  public isConnected(): boolean {
    return this.connected
  }

  /**
   * Get WebSocket state
   */
  public getConnectionState(): string | null {
    if (!this.signalingSocket) {
      return null
    }
    
    switch (this.signalingSocket.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting'
      case WebSocket.OPEN:
        return 'connected'
      case WebSocket.CLOSING:
        return 'closing'
      case WebSocket.CLOSED:
        return 'closed'
      default:
        return 'unknown'
    }
  }
}

export default WebRTCService
