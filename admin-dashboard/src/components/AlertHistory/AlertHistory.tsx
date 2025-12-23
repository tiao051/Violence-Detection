import React, { useState, useMemo, useEffect } from "react";
import { useAlerts, Alert } from "../../contexts";
import Swal from 'sweetalert2';
import "./AlertHistory.css";

const AlertHistory: React.FC = () => {
  const { alerts, clearAlerts, markAsReviewed } = useAlerts();
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [cameraCredibility, setCameraCredibility] = useState<Record<string, any>>({});
  const [isReporting, setIsReporting] = useState(false);

  // Fetch camera credibility on mount
  useEffect(() => {
    const fetchCredibility = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/credibility/camera-intelligence');
        if (response.ok) {
          const data = await response.json();
          const credibilityMap: Record<string, any> = {};
          data.cameras.forEach((cam: any) => {
            credibilityMap[cam.camera_id] = cam;
          });
          setCameraCredibility(credibilityMap);
        }
      } catch (e) {
        console.error("Failed to fetch camera credibility:", e);
      }
    };
    fetchCredibility();
  }, []);

  // Auto-update selectedAlert when alerts array changes (e.g., event_completed received)
  useEffect(() => {
    if (selectedAlert) {
      const updatedAlert = alerts.find(a => a.id === selectedAlert.id);
      if (updatedAlert && (updatedAlert.status !== selectedAlert.status || updatedAlert.video_url !== selectedAlert.video_url)) {
        setSelectedAlert(updatedAlert);
      }
    }
  }, [alerts, selectedAlert]);

  // Filter States
  const [statusFilter, setStatusFilter] = useState("all");
  const [timeFilter, setTimeFilter] = useState("all");
  const [cameraFilter, setCameraFilter] = useState("all");

  // Filter Logic
  const filteredAlerts = useMemo(() => {
    return alerts.filter((alert) => {
      // Status Filter
      if (statusFilter === "new" && alert.is_reviewed) return false;
      if (statusFilter === "reviewed" && !alert.is_reviewed) return false;

      // Camera Filter
      if (cameraFilter !== "all" && alert.camera_id !== cameraFilter)
        return false;

      // Time Filter
      const now = Date.now();
      if (timeFilter === "1h" && now - alert.timestamp > 3600000) return false;
      if (timeFilter === "24h" && now - alert.timestamp > 86400000)
        return false;
      if (timeFilter === "7d" && now - alert.timestamp > 604800000)
        return false;

      return true;
    });
  }, [alerts, statusFilter, timeFilter, cameraFilter]);

  const formatDate = (ts: number) => {
    // Backend provides timestamp in seconds (Unix time)
    // Convert to milliseconds for JavaScript Date
    const timestampMs = ts * 1000;
    return new Date(timestampMs).toLocaleString("vi-VN", {
      timeZone: "Asia/Ho_Chi_Minh",
      month: "2-digit",
      day: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  };

  const handleRowClick = (alert: Alert) => {
    markAsReviewed(alert.id);
    setSelectedAlert(alert);
  };

  const handleReportFalseAlarm = async () => {
    if (!selectedAlert) return;
    setIsReporting(true);
    try {
      const response = await fetch(`http://localhost:8000/api/credibility/mark-false-alarm/${selectedAlert.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          alert_id: selectedAlert.id,
          camera_id: selectedAlert.camera_id,
          reason: "User reported via dashboard"
        })
      });

      if (response.ok) {
        Swal.fire({
          icon: 'success',
          title: 'Report Sent',
          text: 'Alert reported as False Alarm. Thank you for your feedback!',
          timer: 2500,
          showConfirmButton: false,
          background: '#1e293b',
          color: '#fff'
        });
        setSelectedAlert(null); // Close panel
      } else {
        Swal.fire({
          icon: 'error',
          title: 'Failed',
          text: 'Failed to report false alarm.',
          background: '#1e293b',
          color: '#fff'
        });
      }
    } catch (e) {
      console.error("Error reporting false alarm:", e);
      Swal.fire({
        icon: 'error',
        title: 'Error',
        text: 'System error while reporting.',
        background: '#1e293b',
        color: '#fff'
      });
    } finally {
      setIsReporting(false);
    }
  };

  const checkVideoStatus = async () => {
    if (!selectedAlert) return;

    try {
      // Use lookup endpoint with camera_id and timestamp
      // Frontend timestamp is in seconds, but backend expects milliseconds
      const timestampMs = selectedAlert.timestamp * 1000;
      const url = `http://localhost:8000/api/events/lookup?camera_id=${selectedAlert.camera_id}&timestamp=${timestampMs}`;
      console.log("Calling lookup endpoint:", url);

      const response = await fetch(url);

      if (response.ok) {
        const data = await response.json();
        console.log("Lookup response:", data);
        if (data.video_url) {
          const updatedAlert = { ...selectedAlert, video_url: data.video_url };
          setSelectedAlert(updatedAlert);
        }
      }
    } catch (e) {
      console.error("Error checking video status:", e);
    }
  };

  const getCredibilityBadge = (cameraId: string) => {
    const cred = cameraCredibility[cameraId];
    if (!cred) return null;

    let tierClass = "medium";
    if (cred.credibility_tier === "HIGH") tierClass = "high";
    if (cred.credibility_tier === "LOW") tierClass = "low";

    return (
      <span className={`credibility-badge credibility-${tierClass}`} title={`Score: ${cred.credibility_score} | ${cred.recommendation}`}>
        {cred.credibility_tier} TRUST
      </span>
    );
  };

  return (
    <div className="alert-history-container">
      <div className="history-header">
        <div className="header-title-group">
          <h2>Alert History</h2>
          <span className="alert-count">
            {filteredAlerts.length} Events Found
          </span>
        </div>
        <button
          className="clear-btn"
          onClick={clearAlerts}
          disabled={alerts.length === 0}
        >
          Clear History
        </button>
      </div>

      {/* Filter Bar */}
      <div className="filter-bar">
        <div className="filter-group">
          <label>Status:</label>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="all">All Status</option>
            <option value="new">New</option>
            <option value="reviewed">Reviewed</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Time:</label>
          <select
            value={timeFilter}
            onChange={(e) => setTimeFilter(e.target.value)}
          >
            <option value="all">All Time</option>
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Camera:</label>
          <select
            value={cameraFilter}
            onChange={(e) => setCameraFilter(e.target.value)}
          >
            <option value="all">All Cameras</option>
            <option value="cam1">Le Trong Tan Intersection</option>
            <option value="cam2">Cong Hoa Intersection</option>
            <option value="cam3">Au Co Junction</option>
            <option value="cam4">Hoa Binh Intersection</option>
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
                    className={`${selectedAlert?.id === alert.id ? 'selected' : ''}`}
                  >
                    <td>
                      <span className={`status-badge ${alert.is_reviewed ? 'reviewed' : 'new'}`}>
                        {alert.is_reviewed ? 'Reviewed' : 'New'}
                      </span>
                    </td>
                    <td className="time-cell">{formatDate(alert.timestamp)}</td>
                    <td className="camera-cell">
                      {alert.camera_id}
                      {getCredibilityBadge(alert.camera_id) && (
                        <span style={{ marginLeft: '6px', fontSize: '10px' }}>{getCredibilityBadge(alert.camera_id)}</span>
                      )}
                    </td>
                    <td>
                      <div className="confidence-bar-wrapper">
                        <div
                          className="confidence-bar"
                          style={{ width: `${(alert.raw_violence_score || alert.violence_score) * 100}%` }}
                        />
                        <span>{((alert.raw_violence_score || alert.violence_score) * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td>
                      {alert.image_base64 ? (
                        <img
                          src={
                            alert.image_base64.startsWith("data:")
                              ? alert.image_base64
                              : `data:image/jpeg;base64,${alert.image_base64}`
                          }
                          alt="Snapshot"
                          className="table-thumb"
                        />
                      ) : (
                        <div className="no-snapshot-thumb" title="No snapshot available">
                          <span>📷</span>
                        </div>
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
              <button
                className="close-btn"
                onClick={() => setSelectedAlert(null)}
              >
                ×
              </button>
            </div>
            <div className="detail-content">
              <div className="detail-image-container">
                {selectedAlert.video_url ? (
                  <div className="video-wrapper">
                    <video
                      controls
                      autoPlay
                      src={
                        selectedAlert.video_url.startsWith("http")
                          ? selectedAlert.video_url
                          : `http://localhost:8000${selectedAlert.video_url}`
                      }
                      className="detail-video"
                      onError={(e) => {
                        console.error(
                          "Video failed to load:",
                          selectedAlert.video_url
                        );
                        // Hide video, show fallback image instead
                        const target = e.target as HTMLVideoElement;
                        target.style.display = "none";
                        const parent = target.parentElement;
                        if (parent && selectedAlert.image_base64) {
                          parent.innerHTML = `
                            <img src="${selectedAlert.image_base64.startsWith("data:")
                              ? selectedAlert.image_base64
                              : "data:image/jpeg;base64," +
                              selectedAlert.image_base64
                            }" alt="Snapshot" style="max-width: 100%; border-radius: 8px;" />
                            <div style="text-align: center; color: #f87171; margin-top: 8px;">⚠️ Video format not supported. Old events may have corrupted videos.</div>
                          `;
                        }
                      }}
                    />
                  </div>
                ) : selectedAlert.image_base64 ? (
                  <div className="no-video-yet">
                    <img
                      src={
                        selectedAlert.image_base64.startsWith("data:")
                          ? selectedAlert.image_base64
                          : `data:image/jpeg;base64,${selectedAlert.image_base64}`
                      }
                      alt="Snapshot"
                    />
                    <div className="video-pending">
                      {selectedAlert.status === "active" ? (
                        <p>🔴 Event in progress...</p>
                      ) : (
                        <>
                          <p>📹 Video processing...</p>
                          <button
                            onClick={checkVideoStatus}
                            className="refresh-btn"
                          >
                            Check Again
                          </button>
                        </>
                      )}
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
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span>{selectedAlert.camera_id}</span>
                    {getCredibilityBadge(selectedAlert.camera_id)}
                  </div>
                </div>
                <div className="info-row">
                  <label>Confidence:</label>
                  <span className="danger-text">
                    {((selectedAlert.raw_violence_score || selectedAlert.violence_score) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="info-row">
                  <label>Status:</label>
                  <span
                    className={`event-status ${selectedAlert.status || "completed"
                      }`}
                  >
                    {selectedAlert.status === "active" ? "Active" : "Completed"}
                  </span>
                </div>
                <div className="info-row">
                  <label>ID:</label>
                  <span className="mono-text">{selectedAlert.id}</span>
                </div>

                <div className="detail-actions" style={{ marginTop: '24px', padding: '16px', borderTop: '1px solid var(--border-color)' }}>
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px' }}>Is this a false alarm?</p>
                  <button
                    className="report-false-btn"
                    onClick={handleReportFalseAlarm}
                    disabled={isReporting}
                    style={{
                      width: '100%',
                      padding: '8px',
                      background: 'rgba(239, 68, 68, 0.1)',
                      color: 'var(--error-color)',
                      border: '1px solid var(--error-color)',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '13px',
                      fontWeight: 500,
                      transition: 'all 0.2s'
                    }}
                  >
                    {isReporting ? "Reporting..." : "🚫 Report False Alarm"}
                  </button>
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
