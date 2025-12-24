import React, { useEffect, useState } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    Legend
} from 'recharts';
import './AnalyticsDashboard.css';

// --- INTERFACES ---
interface Strategy {
    type: string;
    priority: 'HIGH' | 'MEDIUM' | 'LOW' | 'CRITICAL';
    title: string;
    message: string;
    action: string;
}

interface ForecastItem {
    date: string;
    predicted_count: number;
    day: string;
}

interface HeatmapHour {
    hour: number;
    count: number;
    level: string;
}

interface HeatmapDay {
    day: string;
    total_events: number;
    hours: HeatmapHour[];
}

interface DashboardStats {
    strategies: Strategy[];
    forecast: {
        forecast: ForecastItem[];
        history: any[];
        total_predicted_next_week: number;
        trend_direction: string;
    };
    heatmap: HeatmapDay[];
    anomalies: any[];
}

interface CameraMetrics {
    true_positive_rate: number;
    false_positive_rate: number;
    total_verified: number;
    avg_confidence: number;
    avg_duration: number;
    alerts_per_day: number;
}

interface CameraIntel {
    camera_id: string;
    camera_name: string;
    credibility_score: number;
    credibility_tier: 'HIGH' | 'MEDIUM' | 'LOW';
    cluster: string;
    metrics: CameraMetrics;
    recommendation: string;
}

// ==========================================
// MOCK DATA (HARDCODED FOR DEMO)
// ==========================================

// 1. Dữ liệu Camera: Mô phỏng chính xác kịch bản Cam 1 bị nhiễu
const MOCK_CAMERAS: CameraIntel[] = [
    {
        camera_id: "cam1",
        camera_name: "Luy Ban Bich Street",
        credibility_score: 0.34,
        credibility_tier: "LOW",
        cluster: "Noisy Camera",
        metrics: {
            true_positive_rate: 0.35,
            false_positive_rate: 0.65,
            total_verified: 95400,
            avg_confidence: 0.52,
            avg_duration: 4.1,
            alerts_per_day: 261.3
        },
        recommendation: "⚠️ CRITICAL: Extreme false alarm rate (65%). Sensor malfunction suspected."
    },
    {
        camera_id: "cam2",
        camera_name: "Au Co Junction",
        credibility_score: 0.94,
        credibility_tier: "HIGH",
        cluster: "Reliable Camera",
        metrics: {
            true_positive_rate: 0.96,
            false_positive_rate: 0.04,
            total_verified: 12500,
            avg_confidence: 0.91,
            avg_duration: 28.5,
            alerts_per_day: 34.2
        },
        recommendation: "Highly trustworthy - prioritize alerts from this camera"
    },
    {
        camera_id: "cam3",
        camera_name: "Tan Ky Tan Quy St",
        credibility_score: 0.80,
        credibility_tier: "HIGH",
        cluster: "Selective Camera",
        metrics: {
            true_positive_rate: 0.82,
            false_positive_rate: 0.18,
            total_verified: 24000,
            avg_confidence: 0.78,
            avg_duration: 15.2,
            alerts_per_day: 65.7
        },
        recommendation: "Trustworthy - suitable for automated dispatch"
    },
    {
        camera_id: "cam4",
        camera_name: "Tan Phu Market",
        credibility_score: 0.65,
        credibility_tier: "MEDIUM",
        cluster: "Overcautious",
        metrics: {
            true_positive_rate: 0.71,
            false_positive_rate: 0.29,
            total_verified: 38000,
            avg_confidence: 0.69,
            avg_duration: 9.5,
            alerts_per_day: 104.1
        },
        recommendation: "Moderate reliability - frequent triggering on crowds"
    },
    {
        camera_id: "cam5",
        camera_name: "Dam Sen Park",
        credibility_score: 0.67,
        credibility_tier: "MEDIUM",
        cluster: "Overcautious",
        metrics: {
            true_positive_rate: 0.73,
            false_positive_rate: 0.27,
            total_verified: 30100,
            avg_confidence: 0.67,
            avg_duration: 11.0,
            alerts_per_day: 82.4
        },
        recommendation: "Moderate reliability - check manually during weekends"
    }
];

// 2. Helper để tạo Heatmap giả lập (Đỏ vào ban đêm do Cam 1 bị nhiễu)
const generateMockHeatmap = (): HeatmapDay[] => {
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    return days.map((day, i) => {
        const isWeekend = day === "Sat" || day === "Sun";
        const hours = Array.from({ length: 24 }, (_, h) => {
            let count = 0;
            let level = "low";

            // Kịch bản: Cam 1 bị nhiễu ban đêm (0h-4h) -> Cao
            if (h >= 0 && h <= 4) {
                count = Math.floor(Math.random() * 50) + 100; // 100-150 events
                level = "high";
            } 
            // Kịch bản: Cuối tuần đông đúc buổi tối (18h-21h)
            else if (isWeekend && h >= 18 && h <= 21) {
                count = Math.floor(Math.random() * 40) + 80;
                level = "high";
            }
            // Giờ hành chính bình thường
            else if (h >= 7 && h <= 17) {
                count = Math.floor(Math.random() * 20);
                level = count > 10 ? "medium" : "low";
            } else {
                count = Math.floor(Math.random() * 10);
                level = "low";
            }

            return { hour: h, count, level };
        });

        const total = hours.reduce((acc, curr) => acc + curr.count, 0);
        return { day, total_events: total, hours };
    });
};

// 3. Dữ liệu Dashboard Stats
const MOCK_STATS: DashboardStats = {
    strategies: [
        {
            type: "maintenance",
            priority: "CRITICAL",
            title: "Malfunction at Luy Ban Bich",
            message: "Camera Cam1 shows extreme anomaly (Z-Score: 3.8) and high night-time false positives. Likely hardware glare issue.",
            action: "Schedule Tech Support Immediately"
        },
        {
            type: "deployment",
            priority: "HIGH",
            title: "Weekend Crowd Control",
            message: "Forecast indicates violence surge (350 events) this Sunday, centered around Tan Phu Market.",
            action: "Deploy Mobile Patrol Unit"
        },
        {
            type: "deployment",
            priority: "MEDIUM",
            title: "Shift Adjustment",
            message: "Consistent high activity detected between 00:00 - 04:00 AM. Consider adding night shift support.",
            action: "Review Roster"
        }
    ],
    forecast: {
        trend_direction: "increasing",
        total_predicted_next_week: 1638,
        history: [], // Không cần thiết cho demo line chart này
        forecast: [
            { day: "Mon", predicted_count: 180, date: "2025-10-27" },
            { day: "Tue", predicted_count: 175, date: "2025-10-28" },
            { day: "Wed", predicted_count: 182, date: "2025-10-29" },
            { day: "Thu", predicted_count: 190, date: "2025-10-30" },
            { day: "Fri", predicted_count: 250, date: "2025-10-31" },
            { day: "Sat", predicted_count: 310, date: "2025-11-01" },
            { day: "Sun", predicted_count: 350, date: "2025-11-02" }
        ]
    },
    heatmap: generateMockHeatmap(),
    anomalies: [
        {
            camera_id: "cam1",
            camera_name: "Luy Ban Bich Street",
            event_count: 95400,
            z_score: 3.8,
            is_anomaly: true,
            severity: "critical"
        }
    ]
};

// ==========================================
// COMPONENT
// ==========================================

const AnalyticsDashboard: React.FC = () => {
    const [stats, setStats] = useState<DashboardStats | null>(null);
    const [cameras, setCameras] = useState<CameraIntel[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Giả lập loading 800ms cho chuyên nghiệp
        const timer = setTimeout(() => {
            setStats(MOCK_STATS);
            setCameras(MOCK_CAMERAS);
            setLoading(false);
        }, 800);

        return () => clearTimeout(timer);
    }, []);

    const getChartData = () => {
        if (!stats?.forecast?.forecast) return [];
        return stats.forecast.forecast.map(f => ({
            name: f.day,
            events: f.predicted_count,
            type: 'Forecast'
        }));
    };

    const getStatusColorClass = (tier: string) => {
        switch (tier) {
            case 'HIGH': return 'status-good';
            case 'MEDIUM': return 'status-warn';
            case 'LOW': return 'status-bad';
            default: return 'status-unknown';
        }
    };

    if (loading) return (
        <div className="analytics-dashboard" style={{display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px'}}>
            <div className="p-8 text-white">Initializing Intelligence Engine...</div>
        </div>
    );

    return (
        <div className="analytics-dashboard">
            <div className="analytics-header">
                <h2>Intelligence & Strategy Command</h2>
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
                            ACTION: {strat.action}
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
                            {stats?.forecast?.trend_direction === 'increasing' ? '↗ Increasing' : '↘ Decreasing'}
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
                        <span className="highlight-metric" style={{fontSize: '0.8rem'}}>
                           Fri/Sat Night is riskiest
                        </span>
                    </div>
                    <div className="heatmap-container">
                        <div className="heatmap-header" style={{ display: 'flex', marginLeft: '40px', marginBottom: '8px' }}>
                            <span style={{ flex: 1, fontSize: '10px', color: '#666' }}>00</span>
                            <span style={{ flex: 1, fontSize: '10px', color: '#666' }}>06</span>
                            <span style={{ flex: 1, fontSize: '10px', color: '#666' }}>12</span>
                            <span style={{ flex: 1, fontSize: '10px', color: '#666' }}>18</span>
                            <span style={{ flex: 1, fontSize: '10px', color: '#666' }}>23</span>
                        </div>
                        {stats?.heatmap?.map((day, dIdx) => (
                            <div key={dIdx} className="heatmap-row">
                                <span className="day-label">{day.day}</span>
                                {day.hours.map((h, hIdx) => (
                                    <div
                                        key={hIdx}
                                        className={`hour-cell level-${h.level}`}
                                        title={`${day.day} ${h.hour}:00 - Events: ${h.count} (${h.level})`}
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
                    <span style={{fontSize: '0.9rem', color: '#888'}}>Loaded {cameras.length} devices from HDFS</span>
                </div>
                <div className="table-responsive">
                    <table className="health-table">
                        <thead>
                            <tr>
                                <th>CAMERA</th>
                                <th>STATUS</th>
                                <th>CREDIBILITY SCORE</th>
                                <th>CLUSTER</th>
                                <th>RECOMMENDATION</th>
                            </tr>
                        </thead>
                        <tbody>
                            {cameras.map((cam) => (
                                <tr key={cam.camera_id}>
                                    <td>
                                        <div style={{fontWeight: 'bold'}}>{cam.camera_name}</div>
                                        <div style={{fontSize: '0.8rem', color: '#666'}}>{cam.camera_id}</div>
                                    </td>
                                    <td>
                                        <span className={`status-dot ${getStatusColorClass(cam.credibility_tier)}`}></span>
                                        {cam.credibility_tier === 'LOW' ? 'Anomaly' : 'Online'}
                                    </td>
                                    <td>
                                        <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
                                            <span>{(cam.credibility_score * 100).toFixed(0)}%</span>
                                            <div style={{width: '60px', height: '4px', background: '#333', borderRadius: '2px'}}>
                                                <div 
                                                    style={{
                                                        width: `${cam.credibility_score * 100}%`, 
                                                        height: '100%', 
                                                        background: cam.credibility_score > 0.7 ? '#52c41a' : cam.credibility_score > 0.4 ? '#faad14' : '#f5222d',
                                                        borderRadius: '2px'
                                                    }}
                                                />
                                            </div>
                                        </div>
                                    </td>
                                    <td>
                                        <span className="badge" style={{background: '#333', padding: '2px 6px', borderRadius: '4px', fontSize: '0.8rem'}}>
                                            {cam.cluster}
                                        </span>
                                    </td>
                                    <td style={{fontSize: '0.9rem', color: '#ccc'}}>
                                        {cam.recommendation}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </section>
        </div>
    );
};

export default AnalyticsDashboard;