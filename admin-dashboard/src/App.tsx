import React from 'react'
import Header from './components/Header'
import { VideoDashboard } from './components/VideoDashboard'
import './App.css'

console.log('🚀 App loaded - WebRTC only version')

const App = () => {
  return (
    <div className="app">
      <Header />
      <main className="app-main">
        <VideoDashboard />
      </main>
    </div>
  )
}

export default App
