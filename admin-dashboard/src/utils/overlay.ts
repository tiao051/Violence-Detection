/**
 * Overlay utilities for canvas drawing and metadata management
 */

interface DetectionMetadata {
  frame_seq: number
  timestamp: string
  violence: boolean
  confidence: number
}

interface Point {
  x: number
  y: number
}

interface Size {
  width: number
  height: number
}

/**
 * MetadataBuffer - Circular buffer for storing recent metadata
 * Useful for calculating FPS and maintaining a sliding window of data
 */
export class MetadataBuffer {
  private buffer: DetectionMetadata[] = []
  private maxSize: number = 300 // Keep last 300 frames (~10 seconds at 30fps)
  private timestamps: number[] = []

  /**
   * Add metadata to buffer
   */
  public add(metadata: DetectionMetadata): void {
    this.buffer.push(metadata)
    this.timestamps.push(Date.now())

    // Maintain max size
    if (this.buffer.length > this.maxSize) {
      this.buffer.shift()
      this.timestamps.shift()
    }
  }

  /**
   * Get buffer size
   */
  public getSize(): number {
    return this.buffer.length
  }

  /**
   * Get latest metadata
   */
  public getLatest(): DetectionMetadata | null {
    return this.buffer.length > 0 ? this.buffer[this.buffer.length - 1] : null
  }

  /**
   * Get metadata at index
   */
  public get(index: number): DetectionMetadata | null {
    return index >= 0 && index < this.buffer.length ? this.buffer[index] : null
  }

  /**
   * Calculate FPS based on timestamps
   */
  public calculateFPS(): number {
    if (this.timestamps.length < 2) return 0

    const timeDiff = this.timestamps[this.timestamps.length - 1] - this.timestamps[0]
    const frameCount = this.timestamps.length - 1

    // FPS = frames / time in seconds
    return frameCount / (timeDiff / 1000)
  }

  /**
   * Get violence detection count in buffer
   */
  public getViolenceCount(): number {
    return this.buffer.filter((m) => m.violence).length
  }

  /**
   * Clear buffer
   */
  public clear(): void {
    this.buffer = []
    this.timestamps = []
  }

  /**
   * Get all metadata
   */
  public getAll(): DetectionMetadata[] {
    return [...this.buffer]
  }
}

/**
 * Format metadata for display
 */
export function formatMetadata(metadata: DetectionMetadata): string {
  return `
    Frame: ${metadata.frame_seq}
    Time: ${metadata.timestamp}
    Violence: ${metadata.violence ? 'YES' : 'NO'}
    Confidence: ${(metadata.confidence * 100).toFixed(2)}%
  `.trim()
}

/**
 * Draw metadata on canvas
 */
export function drawMetadata(
  ctx: CanvasRenderingContext2D,
  metadata: DetectionMetadata,
  position: Point = { x: 10, y: 30 }
): void {
  const fontSize = 14
  const lineHeight = 20
  const padding = 10

  ctx.font = `${fontSize}px monospace`
  ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)'
  ctx.lineWidth = 3

  const lines = [
    `Frame: ${metadata.frame_seq}`,
    `Timestamp: ${metadata.timestamp}`,
    `Violence: ${metadata.violence ? '🚨 YES' : '✓ NO'}`,
    `Confidence: ${(metadata.confidence * 100).toFixed(2)}%`,
  ]

  // Calculate background size
  const textWidths = lines.map((line) => ctx.measureText(line).width)
  const maxWidth = Math.max(...textWidths)
  const bgWidth = maxWidth + padding * 2
  const bgHeight = lines.length * lineHeight + padding * 2

  // Draw background
  ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
  ctx.fillRect(position.x - 5, position.y - 20, bgWidth + 10, bgHeight + 5)

  // Draw text
  ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
  lines.forEach((line, index) => {
    ctx.fillText(line, position.x + padding, position.y + index * lineHeight)
  })
}

/**
 * Draw violence indicator
 */
export function drawViolenceIndicator(
  ctx: CanvasRenderingContext2D,
  violence: boolean,
  size: Size,
  position: Point = { x: 10, y: 10 }
): void {
  const radius = size.width / 2
  const color = violence ? '#FF0000' : '#00FF00'
  const shadowColor = violence ? 'rgba(255, 0, 0, 0.3)' : 'rgba(0, 255, 0, 0.3)'

  // Draw shadow/glow
  ctx.fillStyle = shadowColor
  ctx.beginPath()
  ctx.arc(position.x + radius, position.y + radius, radius + 5, 0, Math.PI * 2)
  ctx.fill()

  // Draw indicator circle
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.arc(position.x + radius, position.y + radius, radius, 0, Math.PI * 2)
  ctx.fill()

  // Draw border
  ctx.strokeStyle = violence ? '#AA0000' : '#00AA00'
  ctx.lineWidth = 2
  ctx.stroke()
}

/**
 * Draw confidence bar
 */
export function drawConfidenceBar(
  ctx: CanvasRenderingContext2D,
  confidence: number,
  size: Size,
  position: Point
): void {
  const barHeight = 20
  const padding = 5

  // Draw background
  ctx.fillStyle = 'rgba(100, 100, 100, 0.7)'
  ctx.fillRect(position.x, position.y, size.width, barHeight)

  // Draw filled portion
  const fillWidth = size.width * Math.min(confidence, 1)
  const barColor = confidence > 0.7 ? '#FF4444' : confidence > 0.4 ? '#FFAA00' : '#44FF44'
  ctx.fillStyle = barColor
  ctx.fillRect(position.x, position.y, fillWidth, barHeight)

  // Draw border
  ctx.strokeStyle = '#FFFFFF'
  ctx.lineWidth = 1
  ctx.strokeRect(position.x, position.y, size.width, barHeight)

  // Draw percentage text
  ctx.fillStyle = '#FFFFFF'
  ctx.font = '12px monospace'
  ctx.textAlign = 'center'
  ctx.fillText(`${(confidence * 100).toFixed(0)}%`, position.x + size.width / 2, position.y + 14)
}

/**
 * Draw bounding box
 */
export function drawBoundingBox(
  ctx: CanvasRenderingContext2D,
  box: {
    x: number
    y: number
    width: number
    height: number
  },
  color: string = '#00FF00',
  lineWidth: number = 2,
  label?: string
): void {
  // Draw box
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.strokeRect(box.x, box.y, box.width, box.height)

  // Draw label if provided
  if (label) {
    ctx.fillStyle = color
    ctx.font = '14px monospace'
    ctx.fillText(label, box.x, box.y - 5)
  }
}

/**
 * Draw FPS counter
 */
export function drawFPS(
  ctx: CanvasRenderingContext2D,
  fps: number,
  position: Point = { x: 10, y: 10 }
): void {
  ctx.fillStyle = '#00FF00'
  ctx.font = 'bold 16px monospace'
  ctx.fillText(`FPS: ${fps.toFixed(1)}`, position.x, position.y + 16)
}

/**
 * Clear canvas
 */
export function clearCanvas(ctx: CanvasRenderingContext2D, size: Size): void {
  ctx.clearRect(0, 0, size.width, size.height)
}

/**
 * Draw network status
 */
export function drawNetworkStatus(
  ctx: CanvasRenderingContext2D,
  connected: boolean,
  position: Point = { x: 10, y: 10 }
): void {
  const color = connected ? '#00FF00' : '#FF0000'
  const status = connected ? 'CONNECTED' : 'DISCONNECTED'

  ctx.fillStyle = color
  ctx.font = 'bold 12px monospace'
  ctx.fillText(`● ${status}`, position.x, position.y + 12)
}

/**
 * Draw detection statistics
 */
export function drawStatistics(
  ctx: CanvasRenderingContext2D,
  stats: {
    totalFrames: number
    violenceFrames: number
    avgConfidence: number
  },
  position: Point = { x: 10, y: 50 }
): void {
  const lines = [
    `Total: ${stats.totalFrames}`,
    `Violence: ${stats.violenceFrames}`,
    `Avg Conf: ${(stats.avgConfidence * 100).toFixed(1)}%`,
  ]

  ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
  ctx.font = '12px monospace'

  lines.forEach((line, index) => {
    ctx.fillText(line, position.x, position.y + index * 16)
  })
}
