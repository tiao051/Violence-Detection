import React, { createContext, useContext, useState, useEffect, useRef, ReactNode } from 'react';

const API_BASE = "http://localhost:8000";

// Types
interface Pattern {
  cluster_id: number;
  size: number;
  percentage: number;
  avg_hour: number;
  top_period: string;
  top_camera: string;
  top_day: string;
  weekend_pct: number;
  avg_confidence: number;
  high_severity_pct: number;
  description: string;
}

interface Rule {
  antecedent: string[];
  consequent: string[];
  antecedent_str: string;
  consequent_str: string;
  support: number;
  confidence: number;
  lift: number;
  rule_str: string;
}

interface HighRiskCondition {
  hour: number;
  day: string;
  camera: string;
  high_prob: number;
  risk_level: string;
}

interface Summary {
  total_events: number;
  patterns_discovered: number;
  rules_discovered: number;
  prediction_accuracy: number;
}

interface AnalyticsData {
  summary: Summary | null;
  patterns: Pattern[];
  rules: Rule[];
  highRisk: HighRiskCondition[];
}

interface AnalyticsContextType {
  data: AnalyticsData;
  loading: {
    summary: boolean;
    patterns: boolean;
    rules: boolean;
    highRisk: boolean;
  };
  error: string | null;
  lastFetched: number | null;
  refresh: () => Promise<void>;
}

const AnalyticsContext = createContext<AnalyticsContextType | undefined>(undefined);

export const AnalyticsProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [data, setData] = useState<AnalyticsData>({
    summary: null,
    patterns: [],
    rules: [],
    highRisk: []
  });
  const [loading, setLoading] = useState({
    summary: true,
    patterns: true,
    rules: true,
    highRisk: true
  });
  const [error, setError] = useState<string | null>(null);
  const [lastFetched, setLastFetched] = useState<number | null>(null);

  // Prevent duplicate fetches
  const fetchingRef = useRef(false);

  const fetchData = async () => {
    if (fetchingRef.current) return;

    fetchingRef.current = true;
    setLoading({ summary: true, patterns: true, rules: true, highRisk: true });
    setError(null);

    // Helper for polling endpoints that might return 202
    const fetchWithPoll = async (
      endpoint: string,
      key: keyof AnalyticsData,
      errorMessage: string
    ) => {
      let attempts = 0;
      const maxAttempts = 10; // 20 seconds max

      const poll = async () => {
        try {
          if (attempts >= maxAttempts) {
            setLoading(prev => ({ ...prev, [key]: false }));
            return;
          }

          const res = await fetch(`${API_BASE}${endpoint}`);

          if (res.status === 202) {
            attempts++;
            setTimeout(poll, 1000); // Retry after 1s (was 2s)
            return;
          }

          if (!res.ok) throw new Error(errorMessage);

          const resultData = await res.json();
          setData(prev => ({ ...prev, [key]: resultData }));
        } catch (err) {
          console.error(`${key} fetch error:`, err);
        } finally {
          // Only stop loading if we're not retrying (status != 202)
          if (attempts < maxAttempts || attempts >= maxAttempts) {
            // Logic check: if we are here, we either succeeded, failed, or timed out.
            // If we were retrying (202), we returned early above.
            setLoading(prev => ({ ...prev, [key]: false }));
          }
        }
      };

      poll();
    };

    // Start all fetches in parallel with polling support
    try {
      fetchWithPoll('/api/analytics/summary', 'summary', 'Failed to fetch summary');
      fetchWithPoll('/api/analytics/patterns', 'patterns', 'Failed to fetch patterns');
      fetchWithPoll('/api/analytics/rules?top_n=10', 'rules', 'Failed to fetch rules');
      fetchWithPoll('/api/analytics/predictions?top_n=5', 'highRisk', 'Failed to fetch predictions');

      setLastFetched(Date.now());
    } catch (err) {
      setError("Some analytics data failed to load");
    } finally {
      fetchingRef.current = false;
    }
  };

  // Fetch data immediately on mount (no delay since backend pre-computes)
  useEffect(() => {
    fetchData();
  }, []);

  return (
    <AnalyticsContext.Provider value={{ data, loading, error, lastFetched, refresh: fetchData }}>
      {children}
    </AnalyticsContext.Provider>
  );
};

export const useAnalytics = () => {
  const context = useContext(AnalyticsContext);
  if (!context) throw new Error('useAnalytics must be used within AnalyticsProvider');
  return context;
};
