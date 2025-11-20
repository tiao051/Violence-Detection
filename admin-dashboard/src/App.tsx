import React from 'react'
import Header from './components/Header'
import { VideoDashboard } from './components/VideoDashboard'
import { ThemeProvider } from './contexts'
import './App.css'


const App = () => {
  return (
    <ThemeProvider>
      <div className="app">
        <Header />
        <main className="app-main">
          <VideoDashboard />
        </main>
      </div>
    </ThemeProvider>
  )
}

export default App
