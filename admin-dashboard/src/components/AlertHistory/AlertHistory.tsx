import React, { useState, useMemo } from 'react';
import { useAlerts, Alert } from '../../contexts';
import './AlertHistory.css';

const AlertHistory: React.FC = () => {
  const { alerts, clearAlerts, markAsReviewed } = useAlerts();
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);

  // Filter States
  const [statusFilter, setStatusFilter] = useState('all');
  const [timeFilter, setTimeFilter] = useState('all');
  const [cameraFilter, setCameraFilter] = useState('all');

  // Filter Logic
  const filteredAlerts = useMemo(() => {
    return alerts.filter(alert => {
      // Status Filter
      if (statusFilter === 'new' && alert.is_reviewed) return false;
      if (statusFilter === 'reviewed' && !alert.is_reviewed) return false;

      // Camera Filter
      if (cameraFilter !== 'all' && alert.camera_id !== cameraFilter) return false;

      // Time Filter
      const now = Date.now();
      if (timeFilter === '1h' && now - alert.timestamp > 3600000) return false;
      if (timeFilter === '24h' && now - alert.timestamp > 86400000) return false;
      if (timeFilter === '7d' && now - alert.timestamp > 604800000) return false;

      return true;
    });
  }, [alerts, statusFilter, timeFilter, cameraFilter]);

  const formatDate = (ts: number) => {
    return new Date(ts).toLocaleString('vi-VN', {
      timeZone: 'Asia/Ho_Chi_Minh',
      month: '2-digit', day: '2-digit', year: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      hour12: false
    });
  };

  const handleRowClick = (alert: Alert) => {
    markAsReviewed(alert.id);
    setSelectedAlert(alert);
  };

  const checkVideoStatus = async () => {
    if (!selectedAlert) return;
    
    try {
      // Use lookup endpoint with camera_id and timestamp
      // This bridges the gap between Frontend ID and Backend ID
      const url = `http://localhost:8000/api/events/lookup?camera_id=${selectedAlert.camera_id}&timestamp=${selectedAlert.timestamp}`;
      console.log('Calling lookup endpoint:', url);
      
      const response = await fetch(url);
      
      if (response.ok) {
        const data = await response.json();
        console.log('Lookup response:', data);
        if (data.video_url) {
          const updatedAlert = { ...selectedAlert, video_url: data.video_url };
          setSelectedAlert(updatedAlert);
        }
      }
    } catch (e) {
      console.error('Error checking video status:', e);
    }
  };

  return (
    <div className="alert-history-container">
      <div className="history-header">
        <div className="header-title-group">
          <h2>Alert History</h2>
          <span className="alert-count">{filteredAlerts.length} Events Found</span>
        </div>
        <button className="clear-btn" onClick={clearAlerts} disabled={alerts.length === 0}>
          Clear History
        </button>
      </div>

      {/* Filter Bar */}
      <div className="filter-bar">
        <div className="filter-group">
          <label>Status:</label>
          <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
            <option value="all">All Status</option>
            <option value="new">New</option>
            <option value="reviewed">Reviewed</option>
          </select>
        </div>
        
        <div className="filter-group">
          <label>Time:</label>
          <select value={timeFilter} onChange={e => setTimeFilter(e.target.value)}>
            <option value="all">All Time</option>
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Camera:</label>
          <select value={cameraFilter} onChange={e => setCameraFilter(e.target.value)}>
            <option value="all">All Cameras</option>
            <option value="cam1">Camera 1</option>
            <option value="cam2">Camera 2</option>
            <option value="cam3">Camera 3</option>
            <option value="cam4">Camera 4</option>
          </select>
        </div>
      </div>

      <div className="history-content">
        {filteredAlerts.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📜</div>
            <h3>No Alerts Found</h3>
            <p>Try adjusting your filters or wait for new events.</p>
          </div>
        ) : (
          <div className="table-container">
            <table className="alert-table">
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Time</th>
                  <th>Camera</th>
                  <th>Confidence</th>
                  <th>Snapshot</th>
                </tr>
              </thead>
              <tbody>
                {filteredAlerts.map((alert) => (
                  <tr 
                    key={alert.id} 
                    onClick={() => handleRowClick(alert)}
                    className={selectedAlert?.id === alert.id ? 'selected' : ''}
                  >
                    <td>
                      <span className={`status-badge ${alert.is_reviewed ? 'reviewed' : 'new'}`}>
                        {alert.is_reviewed ? 'Reviewed' : 'New'}
                      </span>
                    </td>
                    <td className="time-cell">{formatDate(alert.timestamp)}</td>
                    <td className="camera-cell">{alert.camera_id}</td>
                    <td>
                      <div className="confidence-bar-wrapper">
                        <div 
                          className="confidence-bar" 
                          style={{ width: `${alert.violence_score * 100}%` }}
                        />
                        <span>{(alert.violence_score * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td>
                      {alert.image_base64 && (
                        <img 
                          src={alert.image_base64.startsWith('data:') ? alert.image_base64 : `data:image/jpeg;base64,${alert.image_base64}`}
                          alt="Snapshot" 
                          className="table-thumb"
                        />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Detail View Panel */}
        {selectedAlert && (
          <div className="alert-detail-panel">
            <div className="detail-header">
              <h3>Event Details</h3>
              <button className="close-btn" onClick={() => setSelectedAlert(null)}>×</button>
            </div>
            <div className="detail-content">
              <div className="detail-image-container">
                {selectedAlert.video_url ? (
                  <div className="video-wrapper">
                    <video 
                      controls 
                      autoPlay 
                      src={`http://localhost:8000${selectedAlert.video_url}`}
                      className="detail-video"
                    />
                  </div>
                ) : selectedAlert.image_base64 ? (
                  <div className="no-video-yet">
                    <img 
                      src={selectedAlert.image_base64.startsWith('data:') ? selectedAlert.image_base64 : `data:image/jpeg;base64,${selectedAlert.image_base64}`}
                      alt="Snapshot" 
                    />
                    <div className="video-pending">
                      <p>📹 Video processing...</p>
                      <button onClick={checkVideoStatus} className="refresh-btn">Check Again</button>
                    </div>
                  </div>
                ) : (
                  <div className="no-image">No Evidence Available</div>
                )}
              </div>
              <div className="detail-info">
                <div className="info-row">
                  <label>Time:</label>
                  <span>{formatDate(selectedAlert.timestamp)}</span>
                </div>
                <div className="info-row">
                  <label>Camera:</label>
                  <span>{selectedAlert.camera_id}</span>
                </div>
                <div className="info-row">
                  <label>Confidence:</label>
                  <span className="danger-text">{(selectedAlert.violence_score * 100).toFixed(1)}%</span>
                </div>
                <div className="info-row">
                  <label>ID:</label>
                  <span className="mono-text">{selectedAlert.id}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AlertHistory;
