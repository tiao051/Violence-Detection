import React, { FC, useState } from 'react'
import Header from './components/Header'
import VideoDisplay from './components/VideoDisplay'
import './App.css'


const App: FC = () => {
  const [backendUrl] = useState<string>('ws://localhost:8000')

  return React.createElement(
    'div',
    { className: 'app' },
    React.createElement(Header, {
      onSettingsClick: () => console.log('Settings clicked'),
      onHelpClick: () => console.log('Help clicked'),
    }),
    React.createElement(
      'main',
      { className: 'app-main' },
      React.createElement(VideoDisplay, { signalingUrl: `${backendUrl}/ws/threats` })
    )
  )
}

export default App
