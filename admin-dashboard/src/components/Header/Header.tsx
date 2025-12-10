import React from 'react';
import { useTheme } from '../../contexts';
import './Header.css';

const Header: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="app-header">
      <div className="header-content">
        <h1 className="header-title">Violence Detection System</h1>
        <p className="header-subtitle">Real-time violence detection monitoring</p>
      </div>
      <div className="header-actions">
        <div className="status-indicator">
          <div className="status-dot active"></div>
          <span className="status-text">System Active</span>
        </div>
        <button
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? '☀️' : '🌙'}
        </button>
      </div>
    </header>
  );
};

export default Header;