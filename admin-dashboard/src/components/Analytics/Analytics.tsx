import React from "react";
import { useAnalytics } from "../../contexts";
import "./Analytics.css";

// Camera name mapping (for backwards compatibility - now all names are in English)
const CAMERA_NAME_MAP: Record<string, string> = {
  'Le Trong Tan Intersection': 'Le Trong Tan Intersection',
  'Au Co T-junction': 'Au Co T-junction',
  'Tan Ky Tan Quy T-junction': 'Tan Ky Tan Quy T-junction',
  '77 Alley Tan Ky Tan Quy': '77 Alley Tan Ky Tan Quy',
  'Ho Dac Duy Intersection': 'Ho Dac Duy Intersection',
};

const formatCameraName = (name: string): string => {
  return CAMERA_NAME_MAP[name] || name;
};

// Mini loading spinner for sections
const SectionLoader: React.FC<{ text?: string }> = ({ text = "Loading..." }) => (
  <div className="section-loading">
    <div className="mini-spinner"></div>
    <span>{text}</span>
  </div>
);

// Helper to generate actionable insight from pattern
const getPatternInsight = (pattern: any): string => {
  const timeAdvice = pattern.top_period === "Night" 
    ? "Increase night patrol" 
    : pattern.top_period === "Evening"
    ? "Focus on evening hours" 
    : pattern.top_period === "Morning"
    ? "Monitor morning rush hours"
    : "Monitor peak afternoon hours";
  
  const dayAdvice = pattern.weekend_pct > 50 
    ? "especially on weekends" 
    : "mainly on weekdays";
  
  return `${timeAdvice} at ${formatCameraName(pattern.top_camera)}, ${dayAdvice}.`;
};

// Helper to format rule text for display
const formatRuleText = (text: string): string => {
  return text
    .replace(/AND/g, " + ")
    .replace(/severity_High/g, "High Severity")
    .replace(/severity_Medium/g, "Medium Severity")
    .replace(/severity_Low/g, "Low Severity")
    .replace(/confidence_high/g, "High Confidence")
    .replace(/confidence_medium/g, "Medium Confidence")
    .replace(/confidence_low/g, "Low Confidence")
    .replace(/period_Morning/g, "Morning")
    .replace(/period_Afternoon/g, "Afternoon")
    .replace(/period_Evening/g, "Evening")
    .replace(/period_Night/g, "Night")
    .replace(/hour_morning/g, "Morning Hours")
    .replace(/hour_afternoon/g, "Afternoon Hours")
    .replace(/hour_evening/g, "Evening Hours")
    .replace(/hour_night/g, "Night Hours")
    .replace(/is_weekday/g, "Weekday")
    .replace(/is_weekend/g, "Weekend")
    .replace(/_/g, " ");
};

// Helper to translate rule to plain language
const translateRule = (rule: any): { condition: string; result: string; meaning: string } => {
  const antecedent = rule.antecedent_str || "";
  const consequent = rule.consequent_str || "";
  
  const condition = formatRuleText(antecedent);
  const result = formatRuleText(consequent);
  
  // Generate actionable meaning
  const confidence = Math.round(rule.confidence * 100);
  let meaning = "";
  if (confidence >= 90) {
    meaning = `Very strong correlation (${confidence}%) - Priority monitoring needed`;
  } else if (confidence >= 70) {
    meaning = `Strong correlation (${confidence}%) - Attention recommended`;
  } else {
    meaning = `Moderate correlation (${confidence}%)`;
  }
  
  return { condition, result, meaning };
};

const Analytics: React.FC = () => {
  const { data, loading, error, lastFetched, refresh } = useAnalytics();
  const { summary, patterns, rules, highRisk } = data;

  const formatLastFetched = () => {
    if (!lastFetched) return null;
    const seconds = Math.floor((Date.now() - lastFetched) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    return new Date(lastFetched).toLocaleTimeString();
  };

  const isAllLoading = loading.summary && loading.patterns && loading.rules && loading.highRisk;
  const isAnyLoading = loading.summary || loading.patterns || loading.rules || loading.highRisk;

  if (isAllLoading && !summary && !lastFetched) {
    return (
      <div className="analytics-container">
        <div className="analytics-loading">
          <div className="loading-spinner"></div>
          <p>Loading analytics data...</p>
          <p className="loading-hint">First load may take a minute</p>
        </div>
      </div>
    );
  }

  if (error && !summary && !Array.isArray(patterns)) {
    return (
      <div className="analytics-container">
        <div className="analytics-error">
          <h2>Failed to load data</h2>
          <p>{error}</p>
          <p className="error-hint">Check backend connection</p>
          <button className="refresh-btn" onClick={refresh}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="analytics-container">
      <div className="analytics-header">
        <h1 className="analytics-title">Analytics & Predictions</h1>
        <div className="analytics-actions">
          {lastFetched && (
            <span className="last-fetched">Updated: {formatLastFetched()}</span>
          )}
          <button 
            className="refresh-btn" 
            onClick={refresh} 
            disabled={isAnyLoading}
          >
            {isAnyLoading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="summary-cards">
        {loading.summary || !summary ? (
          <div className="summary-card loading-card">
            <SectionLoader text="Loading summary..." />
          </div>
        ) : (
          <>
            <div className="summary-card highlight">
              <div className="card-value">
                {summary.total_events?.toLocaleString() || 0}
              </div>
              <div className="card-label">Total Events</div>
            </div>
            <div className="summary-card">
              <div className="card-value">{summary.patterns_discovered || 0}</div>
              <div className="card-label">Patterns Discovered</div>
            </div>
            <div className="summary-card">
              <div className="card-value">{summary.rules_discovered || 0}</div>
              <div className="card-label">Association Rules</div>
            </div>
            <div className="summary-card success">
              <div className="card-value">
                {((summary?.prediction_accuracy || 0) * 100).toFixed(0)}%
              </div>
              <div className="card-label">Prediction Accuracy</div>
            </div>
          </>
        )}
      </div>

      {/* Patterns Section */}
      <section className="analytics-section">
        <div className="section-header">
          <h2>Behavior Clusters</h2>
          <p className="section-subtitle">
            Events grouped by similar patterns for easier monitoring
          </p>
        </div>
        {loading.patterns ? (
          <SectionLoader text="Analyzing patterns..." />
        ) : (
          <div className="patterns-grid">
            {Array.isArray(patterns) && patterns.map((pattern, idx) => (
              <div key={idx} className="pattern-card">
                <div className="pattern-header">
                  <span className="pattern-title">
                    {idx === 0 ? "Primary Cluster" : idx === 1 ? "Secondary Cluster" : `Cluster ${idx + 1}`}
                  </span>
                  <span className="pattern-size">
                    {pattern.size?.toLocaleString()} events ({pattern.percentage}%)
                  </span>
                </div>
                
                {/* Actionable insight */}
                <div className="pattern-insight">
                  <strong>Recommendation:</strong> {getPatternInsight(pattern)}
                </div>
                
                <div className="pattern-details">
                  <div className="detail-row">
                    <span>Peak Time:</span>
                    <span className="detail-value">
                      {pattern.top_period} (~{Math.round(pattern.avg_hour)}:00)
                    </span>
                  </div>
                  <div className="detail-row">
                    <span>Top Location:</span>
                    <span className="detail-value">{formatCameraName(pattern.top_camera)}</span>
                  </div>
                  <div className="detail-row">
                    <span>Weekend Events:</span>
                    <span className="detail-value">
                      {pattern.weekend_pct > 50 ? `${pattern.weekend_pct}% (mostly weekends)` : 
                       pattern.weekend_pct > 0 ? `${pattern.weekend_pct}%` : "0% (weekdays only)"}
                    </span>
                  </div>
                  <div className="detail-row">
                    <span>High Severity:</span>
                    <span className={`detail-value ${pattern.high_severity_pct > 50 ? "high" : ""}`}>
                      {pattern.high_severity_pct}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Association Rules */}
      <section className="analytics-section">
        <div className="section-header">
          <h2>Discovered Rules</h2>
          <span className="algorithm-badge">FP-Growth Algorithm</span>
        </div>
        <p className="section-subtitle">
          Mines frequent patterns to find IF → THEN relationships. Confidence = probability the rule is correct.
        </p>
        {loading.rules ? (
          <SectionLoader text="Mining rules..." />
        ) : (
          <div className="rules-cards">
            {Array.isArray(rules) && rules.slice(0, 5).map((rule, idx) => {
              const translated = translateRule(rule);
              return (
                <div key={idx} className="rule-card">
                  <div className="rule-number">#{idx + 1}</div>
                  <div className="rule-content">
                    <div className="rule-flow">
                      <span className="rule-condition">{translated.condition}</span>
                      <span className="rule-arrow">→</span>
                      <span className="rule-result">{translated.result}</span>
                    </div>
                    <div className="rule-meaning">{translated.meaning}</div>
                  </div>
                  <div className="rule-strength">
                    <div className="strength-bar">
                      <div 
                        className="strength-fill" 
                        style={{ width: `${rule.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* High Risk Predictions */}
      <section className="analytics-section">
        <div className="section-header">
          <h2>High Risk Predictions</h2>
          <span className="algorithm-badge">Random Forest Classifier</span>
        </div>
        <p className="section-subtitle">
          Predicts severity for each hour/day/location combination. Shows top conditions with highest "High" severity probability.
        </p>
        {loading.highRisk ? (
          <SectionLoader text="Predicting risks..." />
        ) : (
          <div className="risk-list">
            {Array.isArray(highRisk) && highRisk.length > 0 ? (
              highRisk.map((cond, idx) => (
                <div key={idx} className={`risk-item ${cond.risk_level?.toLowerCase()}`}>
                  <div className="risk-probability">
                    {(cond.high_prob * 100).toFixed(0)}%
                  </div>
                  <div className="risk-details">
                    <div className="risk-time">
                      {cond.day} at {cond.hour?.toString().padStart(2, "0")}:00
                    </div>
                    <div className="risk-location">
                      {formatCameraName(cond.camera)}
                    </div>
                  </div>
                  <div className={`risk-level-badge ${cond.risk_level?.toLowerCase()}`}>
                    {cond.risk_level?.toLowerCase() === "high" ? "High Risk" : 
                     cond.risk_level?.toLowerCase() === "medium" ? "Medium Risk" : "Low Risk"}
                  </div>
                </div>
              ))
            ) : (
              <div className="no-risk">
                <p>No high-risk predictions for the upcoming period</p>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Quick Summary */}
      <section className="analytics-section summary-section">
        <h2>Summary & Recommendations</h2>
        <div className="quick-summary">
          {Array.isArray(patterns) && patterns.length > 0 && (
            <div className="summary-item">
              <span>Priority Location: <strong>{formatCameraName(patterns[0]?.top_camera)}</strong></span>
            </div>
          )}
          {Array.isArray(patterns) && patterns.length > 0 && (
            <div className="summary-item">
              <span>Peak Time: <strong>{patterns[0]?.top_period}</strong> (~{Math.round(patterns[0]?.avg_hour || 0)}:00)</span>
            </div>
          )}
          {Array.isArray(highRisk) && highRisk.length > 0 && (
            <div className="summary-item warning">
              <span><strong>{highRisk.filter(r => r.risk_level?.toLowerCase() === "high").length}</strong> high-risk periods require monitoring</span>
            </div>
          )}
        </div>
      </section>
    </div>
  );
};

export default Analytics;
