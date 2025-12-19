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
  cameraDescription: string;
  lat: number;
  lng: number;
  label: "violence" | "nonviolence";
  confidence: number;
  eventCount: number;
}

interface MapStats {
  totalCameras: number;
  hotspots: number;
  safeZones: number;
  avgConfidence: number;
}

// ============================================================================
// HARDCODED COORDINATES (Real locations in Tan Phu/Tan Binh, Ho Chi Minh City)
// ============================================================================

const CAMERA_COORDINATES: Record<string, { lat: number; lng: number }> = {
  cam1: { lat: 10.8056, lng: 106.6196 }, // Ngã tư Lê Trọng Tấn (Lê Trọng Tấn x Tân Kỳ Tân Quý)
  cam2: { lat: 10.8017, lng: 106.6322 }, // Ngã tư Cộng Hòa (Tân Kỳ Tân Quý x Cộng Hòa)
  cam3: { lat: 10.7983, lng: 106.6397 }, // Ngã ba Âu Cơ (Lũy Bán Bích x Âu Cơ)
  cam4: { lat: 10.7756, lng: 106.6353 }, // Ngã tư Hòa Bình
  cam5: { lat: 10.8112, lng: 106.6283 }, // Ngã tư Tân Sơn Nhì (Tân Sơn Nhì x Tây Thạnh)
};

// Map center (Tan Phu District center)
const MAP_CENTER: [number, number] = [10.7950, 106.6320];
const DEFAULT_ZOOM = 14;
const COVERAGE_RADIUS = 200; // meters

// ============================================================================
// MOCK DATA (Simulating aggregated data from CSV)
// In production, this would come from an API endpoint
// ============================================================================

const MOCK_CAMERA_DATA: CameraLocation[] = [
  {
    cameraId: "cam1",
    cameraName: "Ngã tư Lê Trọng Tấn",
    cameraDescription: "Lê Trọng Tấn giao Tân Kỳ Tân Quý",
    lat: CAMERA_COORDINATES.cam1.lat,
    lng: CAMERA_COORDINATES.cam1.lng,
    label: "nonviolence",
    confidence: 0.92,
    eventCount: 156,
  },
  {
    cameraId: "cam2",
    cameraName: "Ngã tư Cộng Hòa",
    cameraDescription: "Tân Kỳ Tân Quý giao Cộng Hòa",
    lat: CAMERA_COORDINATES.cam2.lat,
    lng: CAMERA_COORDINATES.cam2.lng,
    label: "violence",
    confidence: 0.78,
    eventCount: 234,
  },
  {
    cameraId: "cam3",
    cameraName: "Ngã ba Âu Cơ",
    cameraDescription: "Lũy Bán Bích giao Âu Cơ",
    lat: CAMERA_COORDINATES.cam3.lat,
    lng: CAMERA_COORDINATES.cam3.lng,
    label: "violence",
    confidence: 0.85,
    eventCount: 312,
  },
  {
    cameraId: "cam4",
    cameraName: "Ngã tư Hòa Bình",
    cameraDescription: "Ngã tư Hòa Bình - Lạc Long Quân",
    lat: CAMERA_COORDINATES.cam4.lat,
    lng: CAMERA_COORDINATES.cam4.lng,
    label: "nonviolence",
    confidence: 0.88,
    eventCount: 98,
  },
  {
    cameraId: "cam5",
    cameraName: "Ngã tư Tân Sơn Nhì",
    cameraDescription: "Tân Sơn Nhì giao Tây Thạnh",
    lat: CAMERA_COORDINATES.cam5.lat,
    lng: CAMERA_COORDINATES.cam5.lng,
    label: "violence",
    confidence: 0.91,
    eventCount: 187,
  },
];

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Analyze camera data to generate statistics
 */
const analyzeData = (cameras: CameraLocation[]): MapStats => {
  const hotspots = cameras.filter((c) => c.label === "violence").length;
  const safeZones = cameras.filter((c) => c.label === "nonviolence").length;
  const avgConfidence =
    cameras.reduce((sum, c) => sum + c.confidence, 0) / cameras.length;

  return {
    totalCameras: cameras.length,
    hotspots,
    safeZones,
    avgConfidence: Math.round(avgConfidence * 100),
  };
};

/**
 * Create custom marker icon based on label
 */
const createMarkerIcon = (label: "violence" | "nonviolence"): L.DivIcon => {
  const color = label === "violence" ? "#ef4444" : "#22c55e";
  const iconHtml = `
    <div class="custom-marker ${label}" style="background-color: ${color};">
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

// ============================================================================
// MAIN COMPONENT
// ============================================================================

const MapDashboard: React.FC = () => {
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const markersRef = useRef<Map<string, L.Marker>>(new Map());
  const circlesRef = useRef<Map<string, L.Circle>>(new Map());

  const [cameras] = useState<CameraLocation[]>(MOCK_CAMERA_DATA);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [stats, setStats] = useState<MapStats | null>(null);
  const [showCoverage, setShowCoverage] = useState(true);

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
      const circleColor = camera.label === "violence" ? "#ef4444" : "#22c55e";
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
        icon: createMarkerIcon(camera.label),
      }).addTo(map);

      // Create popup content
      const popupContent = `
        <div class="map-popup">
          <h3 class="popup-title">${camera.cameraName}</h3>
          <p class="popup-description">${camera.cameraDescription}</p>
          <div class="popup-stats">
            <div class="popup-stat">
              <span class="stat-label">Status:</span>
              <span class="stat-value ${camera.label}">${camera.label === "violence" ? "⚠️ Hotspot" : "✓ Safe Zone"}</span>
            </div>
            <div class="popup-stat">
              <span class="stat-label">Confidence:</span>
              <span class="stat-value">${Math.round(camera.confidence * 100)}%</span>
            </div>
            <div class="popup-stat">
              <span class="stat-label">Events:</span>
              <span class="stat-value">${camera.eventCount}</span>
            </div>
            <div class="popup-stat">
              <span class="stat-label">Coverage:</span>
              <span class="stat-value">${COVERAGE_RADIUS}m radius</span>
            </div>
          </div>
          <!-- TODO: Integrate School API here -->
          <!-- Add button to check nearby schools/sensitive locations -->
          <button class="popup-action-btn" onclick="console.log('Check nearby schools for ${camera.cameraId}')">
            Check Nearby Schools
          </button>
        </div>
      `;

      marker.bindPopup(popupContent, {
        maxWidth: 300,
        className: "custom-popup",
      });

      // Handle marker click
      marker.on("click", () => {
        setSelectedCamera(camera.cameraId);
        // TODO: Integrate School API here
        // Call API to get nearby schools and calculate distances
      });

      markersRef.current.set(camera.cameraId, marker);
    });
  };

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

  // Calculate stats on data change
  useEffect(() => {
    setStats(analyzeData(cameras));
  }, [cameras]);

  return (
    <div className="map-dashboard">
      {/* Sidebar */}
      <aside className="map-sidebar">
        <div className="sidebar-header">
          <h2>Camera Locations</h2>
          <p className="sidebar-subtitle">Hotspot Analysis Dashboard</p>
        </div>

        {/* Stats Summary */}
        {stats && (
          <div className="stats-summary">
            <div className="stat-item">
              <span className="stat-number">{stats.totalCameras}</span>
              <span className="stat-label">Cameras</span>
            </div>
            <div className="stat-item hotspot">
              <span className="stat-number">{stats.hotspots}</span>
              <span className="stat-label">Hotspots</span>
            </div>
            <div className="stat-item safe">
              <span className="stat-number">{stats.safeZones}</span>
              <span className="stat-label">Safe Zones</span>
            </div>
          </div>
        )}

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
              className={`camera-item ${camera.label} ${selectedCamera === camera.cameraId ? "selected" : ""}`}
              onClick={() => panToCamera(camera.cameraId)}
            >
              <div className="camera-indicator">
                <span
                  className={`status-dot ${camera.label}`}
                  title={camera.label === "violence" ? "Hotspot" : "Safe Zone"}
                />
              </div>
              <div className="camera-info">
                <span className="camera-name">{camera.cameraName}</span>
                <span className="camera-desc">{camera.cameraDescription}</span>
              </div>
              <div className="camera-confidence">
                {Math.round(camera.confidence * 100)}%
              </div>
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="map-legend">
          <h3>Legend</h3>
          <div className="legend-item">
            <span className="legend-color violence" />
            <span>Hotspot (Violence detected)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color safe" />
            <span>Safe Zone</span>
          </div>
          <div className="legend-item">
            <span className="legend-circle" />
            <span>200m Coverage Area</span>
          </div>
        </div>
      </aside>

      {/* Map Container */}
      <div className="map-container" ref={mapContainerRef} />
    </div>
  );
};

export default MapDashboard;
