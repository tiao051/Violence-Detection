export interface DetectionMetadata {
  frame_seq: number
  timestamp: string
  violence: boolean
  confidence: number
  camera_id?: string
}

export interface Detection {
  id: string
  timestamp: string
  violence: boolean
  camera: string
  confidence: number
}

export interface VideoDisplayProps {
  signalingUrl?: string
}

export interface HeaderProps {
  onSettingsClick?: () => void
  onHelpClick?: () => void
}

export interface StatusBadgeProps {
  violence: boolean
  confidence: number
}

export interface DetectionLogProps {
  detections: Detection[]
}
