import React, { useEffect, useState } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    BarChart, Bar, Legend
} from 'recharts';
import './AnalyticsDashboard.css';

interface DashboardStats {
    strategies: Strategy[];
    forecast: {
        forecast: any[];
        history: any[];
        total_predicted_next_week: number;
        trend_direction: string;
    };
    heatmap: HeatmapDay[];
    anomalies: any[];
}

interface Strategy {
    type: string;
    priority: 'HIGH' | 'MEDIUM' | 'LOW' | 'CRITICAL';
    title: string;
    message: string;
    action: string;
}

interface HeatmapDay {
    day: string;
    hours: { hour: number; count: number; level: string }[];
}

const AnalyticsDashboard: React.FC = () => {
    const [stats, setStats] = useState<DashboardStats | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchStats();
    }, []);

    const fetchStats = async () => {
        try {
            const response = await fetch('/api/credibility/dashboard-stats');
            const data = await response.json();
            if (data.success) {
                setStats(data);
            } else {
                // Fallback dummy data if backend not ready (for dev visualization)
                setDummyData();
            }
        } catch (error) {
            console.error("Failed to fetch dashboard stats", error);
            setDummyData();
        } finally {
            setLoading(false);
        }
    };

    const setDummyData = () => {
        setStats({
            strategies: [
                {
                    type: "deployment",
                    priority: "MEDIUM",
                    title: "Predicted Surge: Friday Night",
                    message: "Forecast models predict a +25% violence increase this Friday. Focus patrols on Tan Phu Market.",
                    action: "Schedule Extra Shift"
                },
                {
                    type: "maintenance",
                    priority: "CRITICAL",
                    title: "Anomaly: Cam 05",
                    message: "Camera 05 is showing statistical deviation (Z-Score: 2.8). Likely calibration error.",
                    action: "Inspect Hardware"
                }
            ],
            forecast: {
                forecast: [
                    { day: "Mon", predicted_count: 5 }, { day: "Tue", predicted_count: 4 },
                    { day: "Wed", predicted_count: 6 }, { day: "Thu", predicted_count: 8 },
                    { day: "Fri", predicted_count: 12 }, { day: "Sat", predicted_count: 15 },
                    { day: "Sun", predicted_count: 10 }
                ],
                history: [
                    { day: "Mon", count: 4 }, { day: "Tue", count: 5 }, { day: "Wed", count: 5 },
                    { day: "Thu", count: 7 }, { day: "Fri", count: 10 }, { day: "Sat", count: 12 },
                    { day: "Sun", count: 9 }
                ],
                total_predicted_next_week: 60,
                trend_direction: "increasing"
            },
            heatmap: Array(7).fill(0).map((_, i) => ({
                day: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i],
                hours: Array(24).fill(0).map((_, h) => ({
                    hour: h,
                    count: Math.random() * 10,
                    level: Math.random() > 0.8 ? "high" : Math.random() > 0.5 ? "medium" : "low"
                }))
            })),
            anomalies: []
        });
    };

    // Combine history and forecast for chart
    const getChartData = () => {
        if (!stats) return [];
        // Combine history + forecast into one array with 'type' key
        // Simplified: Just showing forecast for now as per design focus
        return stats.forecast.forecast.map(f => ({
            name: f.day,
            events: f.predicted_count,
            type: 'Forecast'
        }));
    };

    if (loading) return <div className="p-8 text-white">Loading Intelligence...</div>;

    return (
        <div className="analytics-dashboard">
            <div className="analytics-header">
                <h2>📊 Intelligence & Strategy Command</h2>
            </div>

            {/* 1. AI STRATEGY DECK */}
            <section className="command-center">
                {stats?.strategies.map((strat, idx) => (
                    <div key={idx} className={`strategy-card ${strat.priority.toLowerCase()}-priority`}>
                        <div className="card-header">
                            <span className="card-title">
                                {strat.priority === 'CRITICAL' ? '🔥' : '🛡️'} {strat.title}
                            </span>
                            <span className="card-badge">{strat.priority}</span>
                        </div>
                        <p className="card-message">{strat.message}</p>
                        <div className="card-action">
                            ACTION: {strat.action} →
                        </div>
                    </div>
                ))}
            </section>

            {/* 2. CHARTS GRID */}
            <section className="charts-grid">
                {/* LEFT: TREND FORECAST */}
                <div className="chart-panel">
                    <div className="panel-title">
                        <span>Weekly Violence Forecast</span>
                        <span className="highlight-metric">
                            {stats?.forecast.trend_direction === 'increasing' ? '↗ Increasing' : '↘ Decreasing'}
                        </span>
                    </div>
                    <div style={{ width: '100%', height: 300 }}>
                        <ResponsiveContainer>
                            <LineChart data={getChartData()}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis dataKey="name" stroke="#888" />
                                <YAxis stroke="#888" />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1e1e1e', border: '1px solid #333' }}
                                />
                                <Legend />
                                <Line
                                    type="monotone"
                                    dataKey="events"
                                    stroke="#1890ff"
                                    strokeWidth={3}
                                    dot={{ r: 4 }}
                                    activeDot={{ r: 8 }}
                                    name="Predicted Events"
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* RIGHT: PEAK DANGER HEATMAP */}
                <div className="chart-panel">
                    <div className="panel-title">
                        <span>Peak Danger Heatmap (7 Days)</span>
                        <span className="highlight-metric">🔥 Peak: Fri 18:00</span>
                    </div>
                    <div className="heatmap-container">
                        <div className="heatmap-header" style={{ display: 'flex', marginLeft: '40px', marginBottom: '8px' }}>
                            <span style={{ flex: 1, fontSize: '10px', color: '#666' }}>00</span>
                            <span style={{ flex: 1, fontSize: '10px', color: '#666' }}>06</span>
                            <span style={{ flex: 1, fontSize: '10px', color: '#666' }}>12</span>
                            <span style={{ flex: 1, fontSize: '10px', color: '#666' }}>18</span>
                            <span style={{ flex: 1, fontSize: '10px', color: '#666' }}>23</span>
                        </div>
                        {stats?.heatmap.map((day, dIdx) => (
                            <div key={dIdx} className="heatmap-row">
                                <span className="day-label">{day.day}</span>
                                {day.hours.map((h, hIdx) => (
                                    <div
                                        key={hIdx}
                                        className={`hour-cell level-${h.level}`}
                                        title={`${day.day} ${h.hour}:00 - ${h.level.toUpperCase()} Risk`}
                                    />
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* 3. CAMERA HEALTH */}
            <section className="health-panel">
                <div className="health-header">
                    <h3>Camera Health & Anomaly Matrix</h3>
                </div>
                <table className="health-table">
                    <thead>
                        <tr>
                            <th>CAMERA</th>
                            <th>STATUS</th>
                            <th>CREDIBILITY</th>
                            <th>RECOMMENDATION</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Cam 01 - Main Gate</td>
                            <td><span className="status-dot status-good"></span>Online</td>
                            <td>98% (High)</td>
                            <td>Trust High</td>
                        </tr>
                        <tr>
                            <td>Cam 05 - Tan Phu Market</td>
                            <td><span className="status-dot status-warn"></span>Warning</td>
                            <td>45% (Low)</td>
                            <td>Recalibrate Sensitivity</td>
                        </tr>
                        <tr>
                            <td>Cam 03 - Parking B</td>
                            <td><span className="status-dot status-bad"></span>Anomaly</td>
                            <td>12% (Critical)</td>
                            <td>Hardware Inspection</td>
                        </tr>
                    </tbody>
                </table>
            </section>
        </div>
    );
};

export default AnalyticsDashboard;
