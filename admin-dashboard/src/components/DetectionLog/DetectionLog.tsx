import React, { FC } from 'react'
import { DetectionLogProps } from '../../types/detection'
import './DetectionLog.css'

const DetectionLog: FC<DetectionLogProps> = ({ detections }) => {
  const renderIndicator = (violence: boolean) => {
    return violence ? '●' : '○'
  }

  const formatTime = (timestamp: string) => {
    try {
      const date = new Date(timestamp)
      return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      })
    } catch {
      return timestamp
    }
  }

  return React.createElement(
    'div',
    { className: 'detection-log' },
    React.createElement('h3', { className: 'detection-log-title' }, 'Recent Detections'),
    React.createElement(
      'div',
      { className: 'detection-log-content' },
      detections.length === 0
        ? React.createElement(
            'div',
            { className: 'detection-log-empty' },
            'No detections yet'
          )
        : React.createElement(
            'div',
            { className: 'detection-log-list' },
            detections.map((detection) =>
              React.createElement(
                'div',
                { key: detection.id, className: 'detection-log-row' },
                React.createElement(
                  'span',
                  { className: 'detection-log-time text-secondary text-xs' },
                  formatTime(detection.timestamp)
                ),
                React.createElement(
                  'span',
                  {
                    className: `detection-log-status text-sm ${
                      detection.violence ? 'text-danger text-bold' : 'text-success'
                    }`,
                  },
                  detection.violence ? 'VIOLENCE' : 'non-violence'
                ),
                React.createElement(
                  'span',
                  { className: 'detection-log-camera text-secondary text-xs' },
                  detection.camera
                ),
                React.createElement(
                  'span',
                  { className: 'detection-log-confidence text-secondary text-xs' },
                  `${(detection.confidence * 100).toFixed(1)}%`
                ),
                React.createElement(
                  'span',
                  {
                    className: `detection-log-indicator ${
                      detection.violence ? 'danger' : 'success'
                    }`,
                  },
                  renderIndicator(detection.violence)
                )
              )
            )
          )
    )
  )
}

export default DetectionLog
