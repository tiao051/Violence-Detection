import React, { useState } from 'react'
import Header from './components/Header'
import Sidebar from './components/Sidebar/Sidebar'
import { VideoDashboard } from './components/VideoDashboard'
import AlertHistory from './components/AlertHistory/AlertHistory'
import { ThemeProvider, AlertProvider } from './contexts'
import './App.css'

const App = () => {
  const [activeTab, setActiveTab] = useState('live');

  // Render other tabs (not live)
  const renderOtherContent = () => {
    switch (activeTab) {
      case 'history':
        return <AlertHistory />;
      case 'analytics':
        return <div className="placeholder-view"><h2>📈 Analytics</h2><p>Coming soon...</p></div>;
      case 'settings':
        return <div className="placeholder-view"><h2>⚙️ Settings</h2><p>Coming soon...</p></div>;
      default:
        return null;
    }
  };

  return (
    <ThemeProvider>
      <AlertProvider>
        <div className="app-container">
          <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
          <div className="main-content">
            <Header />
            <main className="content-area">
              {/* Keep-Alive: VideoDashboard stays mounted, just hidden */}
              <div style={{ display: activeTab === 'live' ? 'block' : 'none', height: '100%' }}>
                <VideoDashboard />
              </div>
              {/* Other tabs render normally */}
              {activeTab !== 'live' && renderOtherContent()}
            </main>
          </div>
        </div>
      </AlertProvider>
    </ThemeProvider>
  )
}

export default App
