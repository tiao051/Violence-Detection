import React, { useEffect, useRef, useState, useCallback } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import * as turf from "@turf/turf";
import "./MapDashboard.css";

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

interface School {
  id: string;
  name: string;
  lat: number;
  lng: number;
  type: string;
  distance?: number;
}

interface SuggestedLocation {
  lat: number;
  lng: number;
  reason: string;
  betweenCameras: [string, string];
  gapDistance: number;
  score?: number;
  scoreBreakdown?: {
    eventDensity: number;
    gapDistance: number;
    schoolProximity: number;
    criticalCluster: number;
    nightActivity: number;
  };
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

interface MapDashboardProps {
  isVisible?: boolean;
}

const CAMERA_COORDINATES: Record<string, { lat: number; lng: number }> = {
  cam1: { lat: 10.766761, lng: 106.641934 },  // Ngã tư Hòa Bình
  cam2: { lat: 10.804096, lng: 106.637517 },  // Ngã tư Cộng Hòa
  cam3: { lat: 10.801625, lng: 106.636815 },  // Ngã ba Âu Cơ
  cam4: { lat: 10.803639, lng: 106.632466 },  // Ngã ba Lê Trọng Tấn
  cam5: { lat: 10.794952, lng: 106.629491 },  // Ngã tư Tân Sơn Nhì
};

// ... (rest of imports and helper functions)



const MAP_CENTER: [number, number] = [10.7950, 106.6400];
const DEFAULT_ZOOM = 14;
const COVERAGE_RADIUS = 200;
const SCHOOL_SEARCH_RADIUS = 1000;
const VIOLENCE_NEAR_SCHOOL_THRESHOLD = 0.3;
const GAP_ANALYSIS_THRESHOLD = 0.8;

// Weighted Scoring Weights
const WEIGHTS = {
  eventDensity: 0.30,
  gapDistance: 0.25,
  schoolProximity: 0.20,
  criticalCluster: 0.15,
  nightActivity: 0.10,
};

// Camera Placement API Response
interface CameraPlacementApiResponse {
  success: boolean;
  events: Array<{
    camera_id: string;
    lat: number;
    lng: number;
    hour: number;
    is_night: boolean;
    confidence: number;
    is_violence: boolean;
    cluster: number;
  }>;
  violence_events_count: number;
  cameras: Array<{
    camera_id: string;
    lat: number;
    lng: number;
    name: string;
  }>;
  risk_clusters: Array<{
    cluster_id: number;
    name: string;
    avg_hour: number;
    avg_confidence: number;
    description: string;
  }>;
  risk_rules: Array<{
    if: string[];
    then: string[];
    confidence: number;
    lift: number;
    rule_text: string;
  }>;
  statistics: {
    total_events: number;
    violence_events: number;
    night_violence_ratio: number;
    critical_cluster_ratio: number;
  };
}

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const fetchNearbySchools = async (lat: number, lng: number, radiusMeters: number = SCHOOL_SEARCH_RADIUS): Promise<School[]> => {
  const query = `
    [out:json][timeout:25];
    (
      node["amenity"="school"](around:${radiusMeters},${lat},${lng});
      node["amenity"="college"](around:${radiusMeters},${lat},${lng});
      node["amenity"="university"](around:${radiusMeters},${lat},${lng});
      node["amenity"="kindergarten"](around:${radiusMeters},${lat},${lng});
      way["amenity"="school"](around:${radiusMeters},${lat},${lng});
      way["amenity"="college"](around:${radiusMeters},${lat},${lng});
      way["amenity"="university"](around:${radiusMeters},${lat},${lng});
    );
    out center;
  `;

  try {
    const response = await fetch("https://overpass-api.de/api/interpreter", {
      method: "POST",
      body: `data=${encodeURIComponent(query)}`,
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
    });

    if (!response.ok) {
      throw new Error(`Overpass API error: ${response.status}`);
    }

    const data = await response.json();

    return data.elements.map((el: any) => ({
      id: el.id.toString(),
      name: el.tags?.name || el.tags?.["name:vi"] || "Unknown School",
      lat: el.lat || el.center?.lat,
      lng: el.lon || el.center?.lon,
      type: el.tags?.amenity || "school",
    })).filter((s: School) => s.lat && s.lng);
  } catch (error) {
    console.error("Failed to fetch schools:", error);
    return [];
  }
};

const calculateDistance = (lat1: number, lng1: number, lat2: number, lng2: number): number => {
  const from = turf.point([lng1, lat1]);
  const to = turf.point([lng2, lat2]);
  return turf.distance(from, to, { units: "kilometers" });
};

const findNearestSchool = (camera: CameraLocation, schools: School[]): School | null => {
  if (schools.length === 0) return null;

  let nearest: School | null = null;
  let minDistance = Infinity;

  schools.forEach((school) => {
    const distance = calculateDistance(camera.lat, camera.lng, school.lat, school.lng);
    if (distance < minDistance) {
      minDistance = distance;
      nearest = { ...school, distance };
    }
  });

  return nearest;
};

/**
 * Advanced Camera Placement Algorithm using Voronoi + Weighted Scoring
 * 
 * Score = 0.30 × EventDensity + 0.25 × GapDistance + 0.20 × SchoolProximity
 *       + 0.15 × CriticalCluster + 0.10 × NightActivity
 */
const analyzeOptimalPlacements = (
  cameras: CameraLocation[],
  placementData: CameraPlacementApiResponse | null,
  schools: School[]
): SuggestedLocation[] => {
  if (!placementData || !placementData.events || placementData.events.length === 0) {
    // Fallback to simple gap analysis
    return analyzeGapsSimple(cameras);
  }

  const suggestions: SuggestedLocation[] = [];
  const violenceEvents = placementData.events.filter(e => e.is_violence);

  if (violenceEvents.length === 0) {
    return analyzeGapsSimple(cameras);
  }

  // Create candidate points using Voronoi centroids and midpoints
  const candidatePoints: Array<{ lat: number; lng: number; nearestCameras: string[] }> = [];

  // Method 1: Midpoints between cameras (gap analysis)
  for (let i = 0; i < cameras.length; i++) {
    for (let j = i + 1; j < cameras.length; j++) {
      const cam1 = cameras[i];
      const cam2 = cameras[j];
      const distance = calculateDistance(cam1.lat, cam1.lng, cam2.lat, cam2.lng);

      if (distance > GAP_ANALYSIS_THRESHOLD * 0.5) { // Lower threshold for more candidates
        const midLat = (cam1.lat + cam2.lat) / 2;
        const midLng = (cam1.lng + cam2.lng) / 2;
        candidatePoints.push({
          lat: midLat,
          lng: midLng,
          nearestCameras: [cam1.cameraNameEn, cam2.cameraNameEn],
        });
      }
    }
  }

  // Method 2: Event cluster centroids using simple grid-based clustering
  const gridSize = 0.003; // ~300m grid cells
  const eventClusters: Map<string, { events: typeof violenceEvents; lat: number; lng: number }> = new Map();

  violenceEvents.forEach(event => {
    const gridKey = `${Math.floor(event.lat / gridSize)}_${Math.floor(event.lng / gridSize)}`;
    if (!eventClusters.has(gridKey)) {
      eventClusters.set(gridKey, { events: [], lat: 0, lng: 0 });
    }
    const cluster = eventClusters.get(gridKey)!;
    cluster.events.push(event);
    cluster.lat += event.lat;
    cluster.lng += event.lng;
  });

  eventClusters.forEach((cluster, _key) => {
    const centroidLat = cluster.lat / cluster.events.length;
    const centroidLng = cluster.lng / cluster.events.length;

    // Check if not too close to existing camera
    const nearestCamDist = Math.min(...cameras.map(c =>
      calculateDistance(centroidLat, centroidLng, c.lat, c.lng)
    ));

    if (nearestCamDist > 0.15) { // At least 150m from nearest camera
      const nearestCams = cameras
        .map(c => ({ name: c.cameraNameEn, dist: calculateDistance(centroidLat, centroidLng, c.lat, c.lng) }))
        .sort((a, b) => a.dist - b.dist)
        .slice(0, 2);

      candidatePoints.push({
        lat: centroidLat,
        lng: centroidLng,
        nearestCameras: [nearestCams[0]?.name || 'Unknown', nearestCams[1]?.name || 'Unknown'],
      });
    }
  });

  // Calculate score for each candidate point
  const maxEvents = Math.max(...Array.from(eventClusters.values()).map(c => c.events.length), 1);
  const maxDistance = Math.max(...candidatePoints.map(p =>
    Math.min(...cameras.map(c => calculateDistance(p.lat, p.lng, c.lat, c.lng)))
  ), 0.001);

  candidatePoints.forEach(point => {
    // 1. Event Density (30%)
    const nearbyEvents = violenceEvents.filter(e =>
      calculateDistance(point.lat, point.lng, e.lat, e.lng) < 0.3 // 300m radius
    );
    const eventDensityScore = nearbyEvents.length / maxEvents;

    // 2. Gap Distance (25%)
    const distToNearestCam = Math.min(...cameras.map(c =>
      calculateDistance(point.lat, point.lng, c.lat, c.lng)
    ));
    const gapDistanceScore = Math.min(distToNearestCam / maxDistance, 1);

    // 3. School Proximity (20%)
    let schoolProximityScore = 0;
    if (schools.length > 0) {
      const nearestSchoolDist = Math.min(...schools.map(s =>
        calculateDistance(point.lat, point.lng, s.lat, s.lng)
      ));
      schoolProximityScore = nearestSchoolDist < 0.3 ? 1 : 0; // Near school = bonus
    }

    // 4. Critical Cluster (15%) - based on cluster 1 from camera_profiles.json
    const criticalEvents = nearbyEvents.filter(e => e.cluster === 1);
    const criticalClusterScore = nearbyEvents.length > 0
      ? criticalEvents.length / nearbyEvents.length
      : 0;

    // 5. Night Activity (10%) - based on risk_rules.json (Night → HIGH)
    const nightEvents = nearbyEvents.filter(e => e.is_night);
    const nightActivityScore = nearbyEvents.length > 0
      ? nightEvents.length / nearbyEvents.length
      : 0;

    // Calculate total weighted score
    const totalScore =
      WEIGHTS.eventDensity * eventDensityScore +
      WEIGHTS.gapDistance * gapDistanceScore +
      WEIGHTS.schoolProximity * schoolProximityScore +
      WEIGHTS.criticalCluster * criticalClusterScore +
      WEIGHTS.nightActivity * nightActivityScore;

    // Only suggest if score is meaningful
    if (totalScore > 0.25) {
      const reasons: string[] = [];
      if (eventDensityScore > 0.3) reasons.push(`${nearbyEvents.length} violence events nearby`);
      if (gapDistanceScore > 0.5) reasons.push(`${(distToNearestCam * 1000).toFixed(0)}m coverage gap`);
      if (schoolProximityScore > 0) reasons.push('Near educational facility');
      if (criticalClusterScore > 0.3) reasons.push('High-risk time pattern detected');
      if (nightActivityScore > 0.3) reasons.push('Night activity hotspot');

      suggestions.push({
        lat: point.lat,
        lng: point.lng,
        reason: reasons.length > 0 ? reasons.join(' • ') : 'Coverage optimization',
        betweenCameras: point.nearestCameras as [string, string],
        gapDistance: distToNearestCam,
        score: totalScore,
        scoreBreakdown: {
          eventDensity: eventDensityScore,
          gapDistance: gapDistanceScore,
          schoolProximity: schoolProximityScore,
          criticalCluster: criticalClusterScore,
          nightActivity: nightActivityScore,
        },
      });
    }
  });

  // Sort by score descending and take top 5
  return suggestions
    .sort((a, b) => (b.score || 0) - (a.score || 0))
    .slice(0, 5);
};

// Simple gap analysis fallback (original algorithm)
const analyzeGapsSimple = (cameras: CameraLocation[]): SuggestedLocation[] => {
  const riskyCameras = cameras.filter((c) => c.classification === "hotspot" || c.classification === "warning");
  const suggestions: SuggestedLocation[] = [];

  for (let i = 0; i < riskyCameras.length; i++) {
    for (let j = i + 1; j < riskyCameras.length; j++) {
      const cam1 = riskyCameras[i];
      const cam2 = riskyCameras[j];
      const distance = calculateDistance(cam1.lat, cam1.lng, cam2.lat, cam2.lng);

      if (distance > GAP_ANALYSIS_THRESHOLD) {
        const point1 = turf.point([cam1.lng, cam1.lat]);
        const point2 = turf.point([cam2.lng, cam2.lat]);
        const midpoint = turf.midpoint(point1, point2);
        const [lng, lat] = midpoint.geometry.coordinates;

        suggestions.push({
          lat,
          lng,
          reason: `Gap of ${distance.toFixed(2)}km between risk zones`,
          betweenCameras: [cam1.cameraNameEn, cam2.cameraNameEn],
          gapDistance: distance,
        });
      }
    }
  }

  return suggestions;
};

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

const getClassificationColor = (classification: string): string => {
  switch (classification) {
    case "hotspot":
      return "#ef4444";
    case "warning":
      return "#f59e0b";
    case "safe":
      return "#22c55e";
    default:
      return "#6b7280";
  }
};

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

const createSchoolIcon = (): L.DivIcon => {
  const iconHtml = `
    <div class="school-marker">
      <svg viewBox="0 0 24 24" fill="#fbbf24" width="20" height="20">
        <path d="M12 3L1 9l4 2.18v6L12 21l7-3.82v-6l2-1.09V17h2V9L12 3zm6.82 6L12 12.72 5.18 9 12 5.28 18.82 9zM17 15.99l-5 2.73-5-2.73v-3.72L12 15l5-2.73v3.72z"/>
      </svg>
    </div>
  `;
  return L.divIcon({
    html: iconHtml,
    className: "school-marker-container",
    iconSize: [28, 28],
    iconAnchor: [14, 14],
    popupAnchor: [0, -14],
  });
};

const createSuggestionIcon = (): L.DivIcon => {
  const iconHtml = `
    <div class="suggestion-marker">
      <svg viewBox="0 0 24 24" fill="white" width="18" height="18">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/>
      </svg>
    </div>
  `;
  return L.divIcon({
    html: iconHtml,
    className: "suggestion-marker-container",
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -16],
  });
};

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

const MapDashboard: React.FC<MapDashboardProps> = ({ isVisible = true }) => {
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const markersRef = useRef<Map<string, L.Marker>>(new Map());
  const circlesRef = useRef<Map<string, L.Circle>>(new Map());
  const schoolMarkersRef = useRef<L.Marker[]>([]);
  const schoolLinesRef = useRef<L.Polyline[]>([]);
  const suggestionMarkersRef = useRef<L.Marker[]>([]);

  const [cameras, setCameras] = useState<CameraLocation[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [stats, setStats] = useState<MapStats | null>(null);
  const [showCoverage, setShowCoverage] = useState(true);
  const [showSchools, setShowSchools] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingSchools, setLoadingSchools] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [schools, setSchools] = useState<School[]>([]);
  const [nearestSchool, setNearestSchool] = useState<School | null>(null);
  const [suggestions, setSuggestions] = useState<SuggestedLocation[]>([]);
  const [alerts, setAlerts] = useState<string[]>([]);

  useEffect(() => {
    if (isVisible && mapRef.current) {
      setTimeout(() => {
        mapRef.current?.invalidateSize();
      }, 100);
    }
  }, [isVisible]);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);

      // Fetch hotspot data for camera display
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
          algorithm: 'Voronoi + Weighted Scoring',
        });

        // Fetch camera placement data for advanced suggestions
        let placementData: CameraPlacementApiResponse | null = null;
        try {
          const placementResponse = await fetch(`${API_BASE_URL}/api/analytics/camera-placement`);
          if (placementResponse.ok) {
            placementData = await placementResponse.json();
          }
        } catch (err) {
          console.warn('Camera placement API not available, using simple gap analysis');
        }

        // Fetch nearby schools for scoring
        const nearbySchools = await fetchNearbySchools(MAP_CENTER[0], MAP_CENTER[1], 2000);
        setSchools(nearbySchools);

        // Use advanced algorithm with weighted scoring
        const optimalPlacements = analyzeOptimalPlacements(transformed, placementData, nearbySchools);
        setSuggestions(optimalPlacements);
      } else {
        setError("Failed to load hotspot data. Please try again later.");
      }

      setLoading(false);
    };

    loadData();
  }, []);

  const renderMap = useCallback(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: MAP_CENTER,
      zoom: DEFAULT_ZOOM,
      zoomControl: false,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
    }).addTo(map);

    L.control.zoom({ position: "bottomright" }).addTo(map);
    L.control.scale({ position: "bottomleft", metric: true }).addTo(map);

    mapRef.current = map;
    renderMarkers(map, cameras);
  }, [cameras]);

  const renderMarkers = (map: L.Map, cameraList: CameraLocation[]) => {
    markersRef.current.forEach((marker) => marker.remove());
    circlesRef.current.forEach((circle) => circle.remove());
    markersRef.current.clear();
    circlesRef.current.clear();

    cameraList.forEach((camera) => {
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

      const marker = L.marker([camera.lat, camera.lng], {
        icon: createMarkerIcon(camera.classification),
      }).addTo(map);

      const popupContent = `
        <div class="map-popup">
          <h3 class="popup-title">${camera.cameraNameEn}</h3>
          <p class="popup-description">${camera.cameraDescription || camera.cameraNameEn}</p>
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
              <span class="stat-label">Total Events:</span>
              <span class="stat-value">${camera.totalEvents}</span>
            </div>
          </div>
          <div class="popup-hint">Click to find nearby schools</div>
        </div>
      `;

      marker.bindPopup(popupContent, {
        maxWidth: 320,
        className: "custom-popup",
      });

      marker.on("click", async () => {
        setSelectedCamera(camera.cameraId);
        handleCameraClick(camera);
      });

      markersRef.current.set(camera.cameraId, marker);
    });
  };

  const handleCameraClick = async (camera: CameraLocation) => {
    if (!mapRef.current) return;

    setLoadingSchools(true);
    clearSchoolMarkers();

    const nearbySchools = await fetchNearbySchools(camera.lat, camera.lng);
    setSchools(nearbySchools);

    if (nearbySchools.length > 0) {
      const nearest = findNearestSchool(camera, nearbySchools);
      setNearestSchool(nearest);

      const newAlerts: string[] = [];
      if (
        camera.classification === "hotspot" &&
        nearest &&
        nearest.distance !== undefined &&
        nearest.distance < VIOLENCE_NEAR_SCHOOL_THRESHOLD
      ) {
        newAlerts.push(
          `⚠️ ALERT: Violence hotspot "${camera.cameraNameEn}" is only ${(nearest.distance * 1000).toFixed(0)}m from "${nearest.name}"!`
        );
      }
      setAlerts(newAlerts);

      if (showSchools) {
        renderSchoolMarkers(camera, nearbySchools, nearest);
      }

      updateCameraPopup(camera, nearest);
    } else {
      setNearestSchool(null);
      setAlerts([]);
    }

    setLoadingSchools(false);
  };

  const renderSchoolMarkers = (
    camera: CameraLocation,
    schoolList: School[],
    nearest: School | null
  ) => {
    if (!mapRef.current) return;

    schoolList.forEach((school) => {
      const marker = L.marker([school.lat, school.lng], {
        icon: createSchoolIcon(),
      }).addTo(mapRef.current!);

      const distance = calculateDistance(camera.lat, camera.lng, school.lat, school.lng);
      marker.bindPopup(`
        <div class="school-popup">
          <h4>🎓 ${school.name}</h4>
          <p>Type: ${school.type}</p>
          <p>Distance: ${(distance * 1000).toFixed(0)}m from camera</p>
        </div>
      `);

      schoolMarkersRef.current.push(marker);
    });

    if (nearest) {
      const line = L.polyline(
        [
          [camera.lat, camera.lng],
          [nearest.lat, nearest.lng],
        ],
        {
          color: camera.classification === "hotspot" ? "#ef4444" : "#3b82f6",
          weight: 3,
          opacity: 0.7,
          dashArray: "10, 10",
        }
      ).addTo(mapRef.current);

      schoolLinesRef.current.push(line);
    }
  };

  const updateCameraPopup = (camera: CameraLocation, nearest: School | null) => {
    const marker = markersRef.current.get(camera.cameraId);
    if (!marker) return;

    let schoolInfo = "";
    let alertHtml = "";

    if (nearest && nearest.distance !== undefined) {
      const distanceM = (nearest.distance * 1000).toFixed(0);
      schoolInfo = `
        <div class="popup-school-info">
          <span class="school-label">Nearest School:</span>
          <span class="school-name">${nearest.name}</span>
          <span class="school-distance">${distanceM}m away</span>
        </div>
      `;

      if (camera.classification === "hotspot" && nearest.distance < VIOLENCE_NEAR_SCHOOL_THRESHOLD) {
        alertHtml = `
          <div class="popup-alert">
            ⚠️ ALERT: Violence detected near school!
          </div>
        `;
      }
    }

    const popupContent = `
      <div class="map-popup">
        <h3 class="popup-title">${camera.cameraNameEn}</h3>
        <p class="popup-description">${camera.cameraName}</p>
        ${alertHtml}
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
        </div>
        ${schoolInfo}
      </div>
    `;

    marker.setPopupContent(popupContent);
  };

  const clearSchoolMarkers = () => {
    schoolMarkersRef.current.forEach((m) => m.remove());
    schoolMarkersRef.current = [];
    schoolLinesRef.current.forEach((l) => l.remove());
    schoolLinesRef.current = [];
  };

  const renderSuggestionMarkers = () => {
    if (!mapRef.current) return;

    clearSuggestionMarkers();

    suggestions.forEach((suggestion, index) => {
      const marker = L.marker([suggestion.lat, suggestion.lng], {
        icon: createSuggestionIcon(),
      }).addTo(mapRef.current!);

      marker.bindPopup(`
        <div class="suggestion-popup">
          <h4>📍 Suggested Camera Location #${index + 1}</h4>
          ${suggestion.score ? `<div class="score-badge">Score: ${(suggestion.score * 100).toFixed(1)}%</div>` : ''}
          <p class="suggestion-reason">${suggestion.reason}</p>
          <p><strong>Between:</strong> ${suggestion.betweenCameras.join(" ↔ ")}</p>
          ${suggestion.scoreBreakdown ? `
            <div class="score-breakdown">
              <div class="score-item">
                <span>Event Density</span>
                <div class="score-bar"><div style="width: ${suggestion.scoreBreakdown.eventDensity * 100}%"></div></div>
                <span>${(suggestion.scoreBreakdown.eventDensity * 100).toFixed(0)}%</span>
              </div>
              <div class="score-item">
                <span>Coverage Gap</span>
                <div class="score-bar"><div style="width: ${suggestion.scoreBreakdown.gapDistance * 100}%"></div></div>
                <span>${(suggestion.scoreBreakdown.gapDistance * 100).toFixed(0)}%</span>
              </div>
              <div class="score-item">
                <span>Near School</span>
                <div class="score-bar"><div style="width: ${suggestion.scoreBreakdown.schoolProximity * 100}%"></div></div>
                <span>${suggestion.scoreBreakdown.schoolProximity > 0 ? 'Yes' : 'No'}</span>
              </div>
              <div class="score-item">
                <span>Critical Risk</span>
                <div class="score-bar critical"><div style="width: ${suggestion.scoreBreakdown.criticalCluster * 100}%"></div></div>
                <span>${(suggestion.scoreBreakdown.criticalCluster * 100).toFixed(0)}%</span>
              </div>
              <div class="score-item">
                <span>Night Activity</span>
                <div class="score-bar night"><div style="width: ${suggestion.scoreBreakdown.nightActivity * 100}%"></div></div>
                <span>${(suggestion.scoreBreakdown.nightActivity * 100).toFixed(0)}%</span>
              </div>
            </div>
          ` : ''}
          <p class="install-hint">Install a camera here to optimize coverage.</p>
        </div>
      `);

      suggestionMarkersRef.current.push(marker);
    });
  };

  const clearSuggestionMarkers = () => {
    suggestionMarkersRef.current.forEach((m) => m.remove());
    suggestionMarkersRef.current = [];
  };

  const toggleSchools = async () => {
    const newState = !showSchools;
    setShowSchools(newState);

    if (!newState) {
      clearSchoolMarkers();
      return;
    }

    if (!selectedCamera) {
      setLoadingSchools(true);
      const centerSchools = await fetchNearbySchools(MAP_CENTER[0], MAP_CENTER[1], 2000);
      setSchools(centerSchools);

      if (centerSchools.length > 0 && mapRef.current) {
        const virtualCamera: CameraLocation = {
          cameraId: 'center',
          cameraName: 'Map Center',
          cameraNameEn: 'Map Center',
          cameraDescription: '',
          lat: MAP_CENTER[0],
          lng: MAP_CENTER[1],
          classification: 'safe',
          riskLevel: 'LOW',
          violenceRatio: 0,
          avgConfidence: 0,
          hotspotScore: 0,
          totalEvents: 0,
          violenceEvents: 0,
          zScore: 0
        };
        renderSchoolMarkers(virtualCamera, centerSchools, null);
      }
      setLoadingSchools(false);
      return;
    }

    const camera = cameras.find((c) => c.cameraId === selectedCamera);
    if (camera && schools.length > 0) {
      renderSchoolMarkers(camera, schools, nearestSchool);
    } else if (camera) {
      setLoadingSchools(true);
      const nearbySchools = await fetchNearbySchools(camera.lat, camera.lng);
      setSchools(nearbySchools);
      if (nearbySchools.length > 0) {
        const nearest = findNearestSchool(camera, nearbySchools);
        setNearestSchool(nearest);
        renderSchoolMarkers(camera, nearbySchools, nearest);
      }
      setLoadingSchools(false);
    }
  };

  const toggleSuggestions = () => {
    const newState = !showSuggestions;
    setShowSuggestions(newState);

    if (newState) {
      renderSuggestionMarkers();
    } else {
      clearSuggestionMarkers();
    }
  };

  useEffect(() => {
    if (mapRef.current && cameras.length > 0) {
      renderMarkers(mapRef.current, cameras);
    }
  }, [cameras]);

  const panToCamera = (cameraId: string) => {
    const camera = cameras.find((c) => c.cameraId === cameraId);
    if (!camera || !mapRef.current) return;

    setSelectedCamera(cameraId);
    mapRef.current.flyTo([camera.lat, camera.lng], 16, { duration: 0.8 });

    const marker = markersRef.current.get(cameraId);
    if (marker) {
      marker.openPopup();
      handleCameraClick(camera);
    }
  };

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

  const resetView = () => {
    if (mapRef.current) {
      mapRef.current.flyTo(MAP_CENTER, DEFAULT_ZOOM, { duration: 0.5 });
      setSelectedCamera(null);
      clearSchoolMarkers();
      setAlerts([]);
    }
  };

  useEffect(() => {
    renderMap();

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [renderMap]);

  return (
    <div className="map-dashboard">
      <aside className="map-sidebar">
        <div className="sidebar-header">
          <h2>Hotspot Analysis</h2>
          <p className="sidebar-subtitle">GIS Spatial Intelligence</p>
        </div>

        {alerts.length > 0 && (
          <div className="alerts-container">
            {alerts.map((alert, index) => (
              <div key={index} className="alert-item">
                {alert}
              </div>
            ))}
          </div>
        )}

        {loading && (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Analyzing hotspots...</p>
          </div>
        )}

        {error && !loading && (
          <div className="error-container">
            <p className="error-message">{error}</p>
            <button onClick={() => window.location.reload()}>Retry</button>
          </div>
        )}

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
                <span className="stat-label">Safe</span>
              </div>
            </div>

            <div className="algorithm-info">
              <span className="algorithm-badge">{stats.algorithm}</span>
              {suggestions.length > 0 && (
                <span className="suggestion-badge">
                  {suggestions.length} gap{suggestions.length > 1 ? "s" : ""} detected
                </span>
              )}
            </div>

            <div className="sidebar-controls">
              <button
                className={`map-control-btn ${showCoverage ? "active" : ""}`}
                onClick={toggleCoverage}
              >
                {showCoverage ? "Hide Coverage" : "Show Coverage"}
              </button>
              <button
                className={`map-control-btn ${showSchools ? "active" : ""}`}
                onClick={toggleSchools}
                disabled={loadingSchools}
              >
                {loadingSchools ? "Loading..." : showSchools ? "Hide Schools" : "Show Schools"}
              </button>
            </div>

            <div className="sidebar-controls">
              <button
                className={`map-control-btn suggestion ${showSuggestions ? "active" : ""}`}
                onClick={toggleSuggestions}
                disabled={suggestions.length === 0}
              >
                {showSuggestions ? "Hide Suggestions" : "Show Suggestions"}
              </button>
              <button className="map-control-btn" onClick={resetView}>
                Reset View
              </button>
            </div>

            {nearestSchool && selectedCamera && (
              <div className="nearest-school-info">
                <h4>🎓 Nearest School</h4>
                <p className="school-name">{nearestSchool.name}</p>
                <p className="school-distance">
                  {nearestSchool.distance !== undefined
                    ? `${(nearestSchool.distance * 1000).toFixed(0)}m away`
                    : "Distance unknown"}
                </p>
              </div>
            )}

            <div className="map-camera-list">
              <h3>All Cameras</h3>
              {cameras.map((camera) => (
                <div
                  key={camera.cameraId}
                  className={`map-camera-item ${camera.classification} ${selectedCamera === camera.cameraId ? "selected" : ""}`}
                  onClick={() => panToCamera(camera.cameraId)}
                >
                  <div className="map-camera-indicator">
                    <span
                      className={`status-dot ${camera.classification}`}
                      title={getStatusLabel(camera.classification)}
                    />
                  </div>
                  <div className="map-camera-info">
                    <span className="map-camera-name">{camera.cameraNameEn}</span>
                    <span className="map-camera-desc">
                      Score: {(camera.hotspotScore * 100).toFixed(0)}% |
                      Violence: {(camera.violenceRatio * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className={`map-camera-classification ${camera.classification}`}>
                    {getStatusLabel(camera.classification)}
                  </div>
                </div>
              ))}
            </div>

            <div className="map-legend">
              <h3>Legend</h3>
              <div className="legend-items">
                <div className="legend-item">
                  <span className="legend-color hotspot" />
                  <span>Hotspot</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color warning" />
                  <span>Warning</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color safe" />
                  <span>Safe</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color school" />
                  <span>School</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color suggestion" />
                  <span>Suggested</span>
                </div>
              </div>
            </div>
          </>
        )}
      </aside>

      <div className="map-container" ref={mapContainerRef} />
    </div>
  );
};

export default MapDashboard;
