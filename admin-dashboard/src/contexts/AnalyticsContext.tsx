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

    // Fetch each API independently for progressive loading
    const fetchSummary = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/analytics/summary`);
        if (!res.ok) throw new Error("Failed to fetch summary");
        const summaryData = await res.json();
        setData(prev => ({ ...prev, summary: summaryData }));
      } catch (err) {
        console.error('Summary fetch error:', err);
      } finally {
        setLoading(prev => ({ ...prev, summary: false }));
      }
    };

    const fetchPatterns = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/analytics/patterns`);
        if (!res.ok) throw new Error("Failed to fetch patterns");
        const patternsData = await res.json();
        setData(prev => ({ ...prev, patterns: patternsData }));
      } catch (err) {
        console.error('Patterns fetch error:', err);
      } finally {
        setLoading(prev => ({ ...prev, patterns: false }));
      }
    };

    const fetchRules = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/analytics/rules?top_n=10`);
        if (!res.ok) throw new Error("Failed to fetch rules");
        const rulesData = await res.json();
        setData(prev => ({ ...prev, rules: rulesData }));
      } catch (err) {
        console.error('Rules fetch error:', err);
      } finally {
        setLoading(prev => ({ ...prev, rules: false }));
      }
    };

    const fetchHighRisk = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/analytics/predictions?top_n=5`);
        if (!res.ok) throw new Error("Failed to fetch predictions");
        const predictionsData = await res.json();
        setData(prev => ({ ...prev, highRisk: predictionsData }));
      } catch (err) {
        console.error('Predictions fetch error:', err);
      } finally {
        setLoading(prev => ({ ...prev, highRisk: false }));
      }
    };

    // Start all fetches in parallel but update UI progressively
    try {
      await Promise.all([
        fetchSummary(),
        fetchPatterns(),
        fetchRules(),
        fetchHighRisk()
      ]);
      setLastFetched(Date.now());
    } catch (err) {
      setError("Some analytics data failed to load");
    } finally {
      fetchingRef.current = false;
    }
  };

  // Preload data after a short delay (don't block main stream)
  useEffect(() => {
    // Wait 3 seconds after app start to let main functionality initialize first
    const timer = setTimeout(() => {
      fetchData();
    }, 3000);

    return () => clearTimeout(timer);
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
