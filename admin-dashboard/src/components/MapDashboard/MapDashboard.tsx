import React, { useEffect, useRef, useState, useCallback } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./MapDashboard.css";

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

interface CameraLocation {
  cameraId: string;
  cameraName: string;
  cameraNameEn: string;
  cameraDescription: string;
  lat: number;
  lng: number;
  classification: "hotspot" | "warning" | "safe";
  riskLevel: string;
  violenceRatio: number;
  avgConfidence: number;
  hotspotScore: number;
  totalEvents: number;
  violenceEvents: number;
  zScore: number;
}

interface HotspotApiResponse {
  success: boolean;
  algorithm: string;
  description: string;
  weights: {
    violence_ratio: number;
    avg_confidence: number;
    z_score: number;
  };
  thresholds: {
    hotspot: number;
    warning: number;
  };
  total_cameras: number;
  hotspots: number;
  warnings: number;
  safe_zones: number;
  total_events: number;
  total_violence_events: number;
  cameras: Array<{
    camera_id: string;
    camera_name: string;
    camera_name_en: string;
    camera_description: string;
    lat: number;
    lng: number;
    total_events: number;
    violence_events: number;
    violence_ratio: number;
    avg_confidence: number;
    z_score: number;
    hotspot_score: number;
    classification: string;
    risk_level: string;
  }>;
}

interface MapStats {
  totalCameras: number;
  hotspots: number;
  warnings: number;
  safeZones: number;
  totalEvents: number;
  violenceEvents: number;
  algorithm: string;
}

// ============================================================================
// DEFAULT COORDINATES (Fallback if API doesn't return coordinates)
// ============================================================================

const CAMERA_COORDINATES: Record<string, { lat: number; lng: number }> = {
  cam1: { lat: 10.7912, lng: 106.6294 },
  cam2: { lat: 10.8024, lng: 106.6401 },
  cam3: { lat: 10.7935, lng: 106.6512 },
  cam4: { lat: 10.8103, lng: 106.6287 },
  cam5: { lat: 10.7856, lng: 106.6523 },
};

// Map center (Tan Phu District center)
const MAP_CENTER: [number, number] = [10.7950, 106.6400];
const DEFAULT_ZOOM = 14;
const COVERAGE_RADIUS = 200; // meters

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Fetch hotspot analysis from API
 */
const fetchHotspotData = async (): Promise<HotspotApiResponse | null> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/analytics/hotspots`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch hotspot data:", error);
    return null;
  }
};

/**
 * Transform API response to CameraLocation array
 */
const transformApiResponse = (data: HotspotApiResponse): CameraLocation[] => {
  return data.cameras.map((cam) => ({
    cameraId: cam.camera_id,
    cameraName: cam.camera_name,
    cameraNameEn: cam.camera_name_en || cam.camera_name,
    cameraDescription: cam.camera_description || "",
    lat: cam.lat || CAMERA_COORDINATES[cam.camera_id]?.lat || 10.7950,
    lng: cam.lng || CAMERA_COORDINATES[cam.camera_id]?.lng || 106.6400,
    classification: cam.classification as "hotspot" | "warning" | "safe",
    riskLevel: cam.risk_level,
    violenceRatio: cam.violence_ratio,
    avgConfidence: cam.avg_confidence,
    hotspotScore: cam.hotspot_score,
    totalEvents: cam.total_events,
    violenceEvents: cam.violence_events,
    zScore: cam.z_score,
  }));
};

/**
 * Get color based on classification
 */
const getClassificationColor = (classification: string): string => {
  switch (classification) {
    case "hotspot":
      return "#ef4444"; // Red
    case "warning":
      return "#f59e0b"; // Orange
    case "safe":
      return "#22c55e"; // Green
    default:
      return "#6b7280"; // Gray
  }
};

/**
 * Create custom marker icon based on classification
 */
const createMarkerIcon = (classification: string): L.DivIcon => {
  const color = getClassificationColor(classification);
  const iconHtml = `
    <div class="custom-marker ${classification}" style="background-color: ${color};">
      <svg viewBox="0 0 24 24" fill="white" width="16" height="16">
        <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
      </svg>
    </div>
  `;
  return L.divIcon({
    html: iconHtml,
    className: "custom-marker-container",
    iconSize: [32, 40],
    iconAnchor: [16, 40],
    popupAnchor: [0, -40],
  });
};

/**
 * Get status label for display
 */
const getStatusLabel = (classification: string): string => {
  switch (classification) {
    case "hotspot":
      return "Hotspot";
    case "warning":
      return "Warning";
    case "safe":
      return "Safe Zone";
    default:
      return "Unknown";
  }
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

const MapDashboard: React.FC = () => {
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const markersRef = useRef<Map<string, L.Marker>>(new Map());
  const circlesRef = useRef<Map<string, L.Circle>>(new Map());

  const [cameras, setCameras] = useState<CameraLocation[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [stats, setStats] = useState<MapStats | null>(null);
  const [showCoverage, setShowCoverage] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  /**
   * Fetch data from API on mount
   */
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);

      const data = await fetchHotspotData();
      
      if (data && data.success) {
        const transformed = transformApiResponse(data);
        setCameras(transformed);
        setStats({
          totalCameras: data.total_cameras,
          hotspots: data.hotspots,
          warnings: data.warnings,
          safeZones: data.safe_zones,
          totalEvents: data.total_events,
          violenceEvents: data.total_violence_events,
          algorithm: data.algorithm,
        });
      } else {
        setError("Failed to load hotspot data. Please try again later.");
      }
      
      setLoading(false);
    };

    loadData();
  }, []);

  /**
   * Initialize and render the Leaflet map
   */
  const renderMap = useCallback(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    // Initialize map
    const map = L.map(mapContainerRef.current, {
      center: MAP_CENTER,
      zoom: DEFAULT_ZOOM,
      zoomControl: false,
    });

    // Add OpenStreetMap tiles
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
    }).addTo(map);

    // Add zoom control to bottom-right
    L.control.zoom({ position: "bottomright" }).addTo(map);

    // Add scale control
    L.control.scale({ position: "bottomleft", metric: true }).addTo(map);

    mapRef.current = map;

    // Render markers after map is initialized
    renderMarkers(map, cameras);
  }, [cameras]);

  /**
   * Render markers and coverage circles on the map
   */
  const renderMarkers = (map: L.Map, cameraList: CameraLocation[]) => {
    // Clear existing markers and circles
    markersRef.current.forEach((marker) => marker.remove());
    circlesRef.current.forEach((circle) => circle.remove());
    markersRef.current.clear();
    circlesRef.current.clear();

    cameraList.forEach((camera) => {
      // Create coverage circle (200m radius)
      const circleColor = getClassificationColor(camera.classification);
      const circle = L.circle([camera.lat, camera.lng], {
        radius: COVERAGE_RADIUS,
        color: circleColor,
        fillColor: circleColor,
        fillOpacity: 0.15,
        weight: 2,
        opacity: 0.5,
      }).addTo(map);

      circlesRef.current.set(camera.cameraId, circle);

      // Create marker
      const marker = L.marker([camera.lat, camera.lng], {
        icon: createMarkerIcon(camera.classification),
      }).addTo(map);

      // Create popup content with detailed statistics
      const popupContent = `
        <div class="map-popup">
          <h3 class="popup-title">${camera.cameraNameEn}</h3>
          <p class="popup-description">${camera.cameraName}</p>
          <div class="popup-stats">
            <div class="popup-stat">
              <span class="stat-label">Status:</span>
              <span class="stat-value ${camera.classification}">${getStatusLabel(camera.classification)}</span>
            </div>
            <div class="popup-stat">
              <span class="stat-label">Hotspot Score:</span>
              <span class="stat-value">${(camera.hotspotScore * 100).toFixed(1)}%</span>
            </div>
            <div class="popup-stat">
              <span class="stat-label">Violence Ratio:</span>
              <span class="stat-value">${(camera.violenceRatio * 100).toFixed(1)}%</span>
            </div>
            <div class="popup-stat">
              <span class="stat-label">Avg Confidence:</span>
              <span class="stat-value">${(camera.avgConfidence * 100).toFixed(1)}%</span>
            </div>
            <div class="popup-stat">
              <span class="stat-label">Z-Score:</span>
              <span class="stat-value">${camera.zScore.toFixed(2)}</span>
            </div>
            <div class="popup-stat">
              <span class="stat-label">Total Events:</span>
              <span class="stat-value">${camera.totalEvents}</span>
            </div>
            <div class="popup-stat">
              <span class="stat-label">Violence Events:</span>
              <span class="stat-value">${camera.violenceEvents}</span>
            </div>
            <div class="popup-stat">
              <span class="stat-label">Coverage:</span>
              <span class="stat-value">${COVERAGE_RADIUS}m radius</span>
            </div>
          </div>
        </div>
      `;

      marker.bindPopup(popupContent, {
        maxWidth: 300,
        className: "custom-popup",
      });

      // Handle marker click
      marker.on("click", () => {
        setSelectedCamera(camera.cameraId);
      });

      markersRef.current.set(camera.cameraId, marker);
    });
  };

  // Update markers when cameras data changes
  useEffect(() => {
    if (mapRef.current && cameras.length > 0) {
      renderMarkers(mapRef.current, cameras);
    }
  }, [cameras]);

  /**
   * Pan map to selected camera
   */
  const panToCamera = (cameraId: string) => {
    const camera = cameras.find((c) => c.cameraId === cameraId);
    if (!camera || !mapRef.current) return;

    setSelectedCamera(cameraId);
    mapRef.current.flyTo([camera.lat, camera.lng], 16, {
      duration: 0.8,
    });

    // Open popup
    const marker = markersRef.current.get(cameraId);
    if (marker) {
      marker.openPopup();
    }
  };

  /**
   * Toggle coverage circles visibility
   */
  const toggleCoverage = () => {
    setShowCoverage(!showCoverage);
    circlesRef.current.forEach((circle) => {
      if (showCoverage) {
        circle.setStyle({ opacity: 0, fillOpacity: 0 });
      } else {
        circle.setStyle({ opacity: 0.5, fillOpacity: 0.15 });
      }
    });
  };

  /**
   * Reset map view to default
   */
  const resetView = () => {
    if (mapRef.current) {
      mapRef.current.flyTo(MAP_CENTER, DEFAULT_ZOOM, { duration: 0.5 });
      setSelectedCamera(null);
    }
  };

  // Initialize map on mount
  useEffect(() => {
    renderMap();

    // Cleanup on unmount
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [renderMap]);

  return (
    <div className="map-dashboard">
      {/* Sidebar */}
      <aside className="map-sidebar">
        <div className="sidebar-header">
          <h2>Hotspot Analysis</h2>
          <p className="sidebar-subtitle">Statistical Weighted Scoring</p>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Analyzing hotspots...</p>
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div className="error-container">
            <p className="error-message">{error}</p>
            <button onClick={() => window.location.reload()}>Retry</button>
          </div>
        )}

        {/* Stats Summary */}
        {!loading && !error && stats && (
          <>
            <div className="stats-summary">
              <div className="stat-item">
                <span className="stat-number">{stats.totalCameras}</span>
                <span className="stat-label">Cameras</span>
              </div>
              <div className="stat-item hotspot">
                <span className="stat-number">{stats.hotspots}</span>
                <span className="stat-label">Hotspots</span>
              </div>
              <div className="stat-item warning">
                <span className="stat-number">{stats.warnings}</span>
                <span className="stat-label">Warnings</span>
              </div>
              <div className="stat-item safe">
                <span className="stat-number">{stats.safeZones}</span>
                <span className="stat-label">Safe Zones</span>
              </div>
            </div>

            {/* Algorithm Info */}
            <div className="algorithm-info">
              <span className="algorithm-badge">{stats.algorithm}</span>
              <div className="event-stats">
                <span>Total Events: {stats.totalEvents}</span>
                <span>Violence: {stats.violenceEvents}</span>
              </div>
            </div>

            {/* Controls */}
            <div className="sidebar-controls">
              <button
                className={`control-btn ${showCoverage ? "active" : ""}`}
                onClick={toggleCoverage}
              >
                {showCoverage ? "Hide Coverage" : "Show Coverage"}
              </button>
              <button className="control-btn" onClick={resetView}>
                Reset View
              </button>
            </div>

            {/* Camera List */}
            <div className="camera-list">
              <h3>All Cameras</h3>
              {cameras.map((camera) => (
                <div
                  key={camera.cameraId}
                  className={`camera-item ${camera.classification} ${selectedCamera === camera.cameraId ? "selected" : ""}`}
                  onClick={() => panToCamera(camera.cameraId)}
                >
                  <div className="camera-indicator">
                    <span
                      className={`status-dot ${camera.classification}`}
                      title={getStatusLabel(camera.classification)}
                    />
                  </div>
                  <div className="camera-info">
                    <span className="camera-name">{camera.cameraNameEn}</span>
                    <span className="camera-desc">
                      Score: {(camera.hotspotScore * 100).toFixed(0)}% | 
                      Violence: {(camera.violenceRatio * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className={`camera-classification ${camera.classification}`}>
                    {getStatusLabel(camera.classification)}
                  </div>
                </div>
              ))}
            </div>

            {/* Legend */}
            <div className="map-legend">
              <h3>Legend</h3>
              <div className="legend-item">
                <span className="legend-color hotspot" />
                <span>Hotspot (Score ≥ 60%)</span>
              </div>
              <div className="legend-item">
                <span className="legend-color warning" />
                <span>Warning (Score 40-60%)</span>
              </div>
              <div className="legend-item">
                <span className="legend-color safe" />
                <span>Safe Zone (Score &lt; 40%)</span>
              </div>
              <div className="legend-item">
                <span className="legend-circle" />
                <span>200m Coverage Area</span>
              </div>
            </div>
          </>
        )}
      </aside>

      {/* Map Container */}
      <div className="map-container" ref={mapContainerRef} />
    </div>
  );
};

export default MapDashboard;
