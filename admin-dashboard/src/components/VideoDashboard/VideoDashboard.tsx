import React from "react";
import CameraVideo from "./CameraVideo";
import "./VideoDashboard.css";

const cameras = ["cam1", "cam2", "cam3", "cam4"];

const VideoDashboard: React.FC = () => {
  return (
    <div className="video-dashboard-grid">
      {cameras.map((cam) => (
        <CameraVideo key={cam} cameraId={cam} />
      ))}
    </div>
  );
};

export default VideoDashboard;
