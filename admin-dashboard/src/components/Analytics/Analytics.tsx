import React, { useState, useEffect } from "react";
import "./Analytics.css";

// API base URL
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

const Analytics: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [summary, setSummary] = useState<Summary | null>(null);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [rules, setRules] = useState<Rule[]>([]);
  const [highRisk, setHighRisk] = useState<HighRiskCondition[]>([]);

  // Fetch all data on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        const [summaryRes, patternsRes, rulesRes, predictionsRes] =
          await Promise.all([
            fetch(`${API_BASE}/api/analytics/summary`),
            fetch(`${API_BASE}/api/analytics/patterns`),
            fetch(`${API_BASE}/api/analytics/rules?top_n=10`),
            fetch(`${API_BASE}/api/analytics/predictions?top_n=5`),
          ]);

        if (
          !summaryRes.ok ||
          !patternsRes.ok ||
          !rulesRes.ok ||
          !predictionsRes.ok
        ) {
          throw new Error("Failed to fetch analytics data");
        }

        const [summaryData, patternsData, rulesData, predictionsData] =
          await Promise.all([
            summaryRes.json(),
            patternsRes.json(),
            rulesRes.json(),
            predictionsRes.json(),
          ]);

        setSummary(summaryData);
        setPatterns(patternsData);
        setRules(rulesData);
        setHighRisk(predictionsData);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="analytics-container">
        <div className="analytics-loading">
          <div className="loading-spinner"></div>
          <p>Loading analytics data...</p>
          <p className="loading-hint">
            First load may take a minute while training models
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="analytics-container">
        <div className="analytics-error">
          <h2>Error Loading Analytics</h2>
          <p>{error}</p>
          <p className="error-hint">
            Make sure the backend is running on port 8000
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="analytics-container">
      <h1 className="analytics-title">Analytics Overview</h1>

      {/* Summary Cards */}
      <div className="summary-cards">
        <div className="summary-card">
          <div className="card-value">
            {summary?.total_events?.toLocaleString() || 0}
          </div>
          <div className="card-label">Total Events</div>
        </div>
        <div className="summary-card">
          <div className="card-value">{summary?.patterns_discovered || 0}</div>
          <div className="card-label">Patterns Discovered</div>
        </div>
        <div className="summary-card">
          <div className="card-value">{summary?.rules_discovered || 0}</div>
          <div className="card-label">Association Rules</div>
        </div>
        <div className="summary-card">
          <div className="card-value">
            {((summary?.prediction_accuracy || 0) * 100).toFixed(0)}%
          </div>
          <div className="card-label">Prediction Accuracy</div>
        </div>
      </div>

      {/* Patterns Section */}
      <section className="analytics-section">
        <h2>Discovered Patterns (K-means Clustering)</h2>
        <div className="patterns-grid">
          {patterns.map((pattern, idx) => (
            <div key={idx} className="pattern-card">
              <div className="pattern-header">
                <span className="pattern-title">Cluster {idx + 1}</span>
                <span className="pattern-size">
                  {pattern.size} events ({pattern.percentage}%)
                </span>
              </div>
              <p className="pattern-description">{pattern.description}</p>
              <div className="pattern-details">
                <div className="detail-row">
                  <span>Peak Time:</span>
                  <span>
                    {pattern.top_period} (~{Math.round(pattern.avg_hour)}:00)
                  </span>
                </div>
                <div className="detail-row">
                  <span>Top Location:</span>
                  <span>{pattern.top_camera}</span>
                </div>
                <div className="detail-row">
                  <span>Weekend:</span>
                  <span>{pattern.weekend_pct}%</span>
                </div>
                <div className="detail-row">
                  <span>High Severity:</span>
                  <span
                    className={pattern.high_severity_pct > 40 ? "high" : ""}
                  >
                    {pattern.high_severity_pct}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Association Rules Section */}
      <section className="analytics-section">
        <h2>Association Rules (FP-Growth)</h2>
        <div className="rules-table-container">
          <table className="rules-table">
            <thead>
              <tr>
                <th>If (Antecedent)</th>
                <th>Then (Consequent)</th>
                <th>Confidence</th>
                <th>Lift</th>
              </tr>
            </thead>
            <tbody>
              {rules.map((rule, idx) => (
                <tr key={idx}>
                  <td>{rule.antecedent_str}</td>
                  <td>{rule.consequent_str}</td>
                  <td>{(rule.confidence * 100).toFixed(0)}%</td>
                  <td>{rule.lift.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* High Risk Conditions Section */}
      <section className="analytics-section">
        <h2>High Risk Conditions (Random Forest)</h2>
        <div className="risk-list">
          {highRisk.map((cond, idx) => (
            <div key={idx} className="risk-item">
              <div className="risk-badge">
                {(cond.high_prob * 100).toFixed(0)}%
              </div>
              <div className="risk-details">
                <div className="risk-time">
                  {cond.day} {cond.hour.toString().padStart(2, "0")}:00
                </div>
                <div className="risk-location">at {cond.camera}</div>
              </div>
              <div className={`risk-level ${cond.risk_level.toLowerCase()}`}>
                {cond.risk_level}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default Analytics;
