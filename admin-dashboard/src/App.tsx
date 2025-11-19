import React, { FC, useState } from 'react'
import VideoDisplay from './components/VideoDisplay'
import './App.css'

/**
 * App Component - Root component of the application
 * Manages backend configuration and renders main layout
 */
const App: FC = () => {
  const [backendUrl] = useState<string>('ws://localhost:8000')

  return React.createElement(
    'div',
    { className: 'app' },
    React.createElement(
      'header',
      { className: 'app-header' },
      React.createElement('h1', null, '🎥 Violence Detection Dashboard'),
      React.createElement(
        'div',
        { className: 'backend-status' },
        React.createElement('span', { className: 'status-badge' }, `Backend: ${backendUrl}`)
      )
    ),
    React.createElement(
      'main',
      { className: 'app-main' },
      React.createElement(VideoDisplay, { signalingUrl: `${backendUrl}/ws/threats` })
    ),
    React.createElement(
      'footer',
      { className: 'app-footer' },
      React.createElement(
        'p',
        null,
        'Real-time violence detection using WebRTC streaming + AI inference (RemoNet 3-stage model)'
      )
    )
  )
}

export default App
