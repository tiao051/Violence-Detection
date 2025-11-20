import React from 'react';
import './Header.css';

const Header: React.FC = () => {
  return (
    <header className="app-header">
      <div className="header-content">
        <h1 className="header-title">Violence Detection System</h1>
        <p className="header-subtitle">Real-time violence detection monitoring</p>
      </div>
      <div className="header-status">
        <div className="status-indicator">
          <div className="status-dot active"></div>
          <span className="status-text">System Active</span>
        </div>
      </div>
    </header>
  );
};

export default Header;