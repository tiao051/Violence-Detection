import React, { useState } from 'react'
import Header from './components/Header'
import Sidebar from './components/Sidebar/Sidebar'
import { VideoDashboard } from './components/VideoDashboard'
import AlertHistory from './components/AlertHistory/AlertHistory'
import AnalyticsDashboard from './components/Analytics/AnalyticsDashboard'

import MapDashboard from './components/MapDashboard'
import UserManagement from './components/UserManagement/UserManagement'
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
        return <AnalyticsDashboard />;
      case 'users':
        return <UserManagement />;
      case 'map':
        return <MapDashboard />;
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

              {/* Keep-Alive: MapDashboard stays mounted, just hidden */}
              <div style={{ display: activeTab === 'map' ? 'block' : 'none', height: '100%' }}>
                <MapDashboard isVisible={activeTab === 'map'} />
              </div>

              {/* Other tabs render normally (unmount on switch) */}
              {activeTab !== 'live' && activeTab !== 'map' && renderOtherContent()}
            </main>
          </div>
        </div>
      </AlertProvider>
    </ThemeProvider>
  )
}

export default App
