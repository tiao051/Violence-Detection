import React, { useState } from 'react'
import Header from './components/Header'
import Sidebar from './components/Sidebar/Sidebar'
import { VideoDashboard } from './components/VideoDashboard'
import { ThemeProvider } from './contexts'
import './App.css'

const App = () => {
  const [activeTab, setActiveTab] = useState('live');

  const renderContent = () => {
    switch (activeTab) {
      case 'live':
        return <VideoDashboard />;
      case 'history':
        return <div className="placeholder-view"><h2>📜 Alert History</h2><p>Coming soon...</p></div>;
      case 'analytics':
        return <div className="placeholder-view"><h2>📈 Analytics</h2><p>Coming soon...</p></div>;
      case 'settings':
        return <div className="placeholder-view"><h2>⚙️ Settings</h2><p>Coming soon...</p></div>;
      default:
        return <VideoDashboard />;
    }
  };

  return (
    <ThemeProvider>
      <div className="app-container">
        <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
        <div className="main-content">
          <Header />
          <main className="content-area">
            {renderContent()}
          </main>
        </div>
      </div>
    </ThemeProvider>
  )
}

export default App
