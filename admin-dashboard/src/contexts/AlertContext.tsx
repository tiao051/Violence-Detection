import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Simplified: No severity analysis, just violence detection
export type SeverityLevel = 'DETECTED';  // Only one level now

export interface Alert {
  id: string;              // Firestore event_id (source of truth)
  timestamp: number;       // Detection timestamp from backend (Unix time in seconds)
  camera_id: string;
  violence_score: number;
  raw_violence_score?: number; // Added to store original confidence
  image_base64?: string;
  is_reviewed?: boolean;
  video_url?: string;
  status: 'active' | 'completed';  // Track event status
}

interface AlertContextType {
  alerts: Alert[];
  addOrUpdateEvent: (event: {
    event_id: string;
    camera_id: string;
    timestamp: number;
    confidence: number;
    raw_confidence?: number;
    snapshot?: string;
    video_url?: string;
    status: 'active' | 'completed';
  }) => void;
  clearAlerts: () => void;
  markAsReviewed: (id: string) => void;
  unreadCount: number;
}

const AlertContext = createContext<AlertContextType | undefined>(undefined);

export const AlertProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [alerts, setAlerts] = useState<Alert[]>(() => {
    const saved = localStorage.getItem('violence-alerts');
    return saved ? JSON.parse(saved) : [];
  });

  useEffect(() => {
    localStorage.setItem('violence-alerts', JSON.stringify(alerts));
  }, [alerts]);

  /**
   * Firestore-first: Add new event or update existing event.
   * Events are identified by event_id (Firestore document ID).
   */
  const addOrUpdateEvent = (eventData: {
    event_id: string;
    camera_id: string;
    timestamp: number;
    confidence: number;
    raw_confidence?: number;
    snapshot?: string;
    video_url?: string;
    status: 'active' | 'completed';
  }) => {
    if (!eventData.event_id) {
      console.error('Event must have event_id:', eventData);
      return;
    }

    setAlerts(prev => {
      // Find existing event by Firestore event_id
      const existingIndex = prev.findIndex(a => a.id === eventData.event_id);

      if (existingIndex !== -1) {
        // UPDATE existing event
        const updatedAlerts = [...prev];
        updatedAlerts[existingIndex] = {
          ...updatedAlerts[existingIndex],
          violence_score: eventData.confidence,
          raw_violence_score: eventData.raw_confidence,
          image_base64: eventData.snapshot || updatedAlerts[existingIndex].image_base64,
          video_url: eventData.video_url || updatedAlerts[existingIndex].video_url,
          status: eventData.status,
        };
        return updatedAlerts;
      }

      // CREATE new event (event_started)
      const newAlert: Alert = {
        id: eventData.event_id,
        timestamp: eventData.timestamp,
        camera_id: eventData.camera_id,
        violence_score: eventData.confidence,
        raw_violence_score: eventData.raw_confidence,
        image_base64: eventData.snapshot,
        video_url: eventData.video_url,
        is_reviewed: false,
        status: eventData.status,
      };
      return [newAlert, ...prev].slice(0, 100); // Keep last 100
    });
  };

  const clearAlerts = () => setAlerts([]);

  const markAsReviewed = (id: string) => {
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, is_reviewed: true } : a));
  };

  const unreadCount = alerts.filter(a => !a.is_reviewed).length;

  return (
    <AlertContext.Provider value={{ alerts, addOrUpdateEvent, clearAlerts, markAsReviewed, unreadCount }}>
      {children}
    </AlertContext.Provider>
  );
};

export const useAlerts = () => {
  const context = useContext(AlertContext);
  if (!context) throw new Error('useAlerts must be used within AlertProvider');
  return context;
};
