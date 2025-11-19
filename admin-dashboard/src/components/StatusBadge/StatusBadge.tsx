import React, { FC } from 'react'
import { StatusBadgeProps } from '../../types/detection'
import './StatusBadge.css'

const StatusBadge: FC<StatusBadgeProps> = ({ violence, confidence }) => {
  const statusClass = `status-badge ${violence ? 'danger' : 'success'}`
  const statusText = violence ? '🔴 VIOLENCE' : '✓ NON-VIOLENCE'
  const confidencePercent = (confidence * 100).toFixed(1)

  return React.createElement(
    'div',
    { className: statusClass },
    React.createElement('div', { className: 'status-badge-content' }, statusText),
    React.createElement(
      'div',
      { className: 'confidence-text' },
      `${confidencePercent}%`
    )
  )
}

export default StatusBadge
