import React, { useState, useEffect } from 'react';
import './UserManagement.css';

interface User {
  uid: string;
  email: string;
  displayName: string | null;
  authProvider: string | null;
  createdAt: string | null;
  photoUrl: string | null;
  disabled: boolean;
  camerasCount: number;
}

interface Camera {
  id: string;
  name: string;
  location: string;
  owner_uid: string | null;
  owner_email: string | null;
}

const API_BASE = 'http://localhost:8000/api/v1/admin';

const UserManagement: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Modal state
  const [showModal, setShowModal] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [selectedCameras, setSelectedCameras] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);

  // Fetch users and cameras
  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [usersRes, camerasRes] = await Promise.all([
        fetch(`${API_BASE}/users`),
        fetch(`${API_BASE}/cameras`)
      ]);
      
      if (!usersRes.ok || !camerasRes.ok) {
        throw new Error('Failed to fetch data');
      }
      
      const usersData = await usersRes.json();
      const camerasData = await camerasRes.json();
      
      setUsers(usersData);
      setCameras(camerasData);
      setError(null);
    } catch (err) {
      setError('Failed to load data. Is the backend running?');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const openAssignModal = (user: User) => {
    setSelectedUser(user);
    // Get cameras currently assigned to this user
    const userCameras = cameras
      .filter(c => c.owner_uid === user.uid)
      .map(c => c.id);
    setSelectedCameras(userCameras);
    setShowModal(true);
  };

  const toggleCamera = (cameraId: string) => {
    setSelectedCameras(prev => 
      prev.includes(cameraId)
        ? prev.filter(id => id !== cameraId)
        : [...prev, cameraId]
    );
  };

  const saveAssignments = async () => {
    if (!selectedUser) return;
    
    setSaving(true);
    try {
      // Get current state of all cameras
      const currentUserCameras = cameras
        .filter(c => c.owner_uid === selectedUser.uid)
        .map(c => c.id);
      
      // Determine which cameras to assign and unassign
      const toAssign = selectedCameras.filter(id => !currentUserCameras.includes(id));
      const toUnassign = currentUserCameras.filter(id => !selectedCameras.includes(id));
      
      // Optimistic update - update local state immediately
      const updatedCameras = cameras.map(cam => {
        if (toAssign.includes(cam.id)) {
          return { ...cam, owner_uid: selectedUser.uid, owner_email: selectedUser.email };
        }
        if (toUnassign.includes(cam.id)) {
          return { ...cam, owner_uid: null, owner_email: null };
        }
        return cam;
      });
      setCameras(updatedCameras);
      
      // Update users camera count
      const updatedUsers = users.map(u => {
        if (u.uid === selectedUser.uid) {
          return { ...u, camerasCount: selectedCameras.length };
        }
        // Decrease count for users who lost cameras
        const lostCameras = toAssign.filter(camId => {
          const cam = cameras.find(c => c.id === camId);
          return cam?.owner_uid === u.uid;
        });
        if (lostCameras.length > 0) {
          return { ...u, camerasCount: Math.max(0, u.camerasCount - lostCameras.length) };
        }
        return u;
      });
      setUsers(updatedUsers);
      
      // Close modal immediately
      setShowModal(false);
      setSelectedUser(null);
      setSaving(false);
      
      // Process API calls in background
      const promises = [];
      for (const camId of toAssign) {
        promises.push(fetch(`${API_BASE}/cameras/${camId}/assign`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_uid: selectedUser.uid })
        }));
      }
      for (const camId of toUnassign) {
        promises.push(fetch(`${API_BASE}/cameras/${camId}/assign`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_uid: null })
        }));
      }
      
      await Promise.all(promises);
      
    } catch (err) {
      console.error('Error saving assignments:', err);
      // Revert on error
      await fetchData();
    }
  };

  const toggleUserStatus = async (user: User) => {
    // Optimistic update
    const newDisabled = !user.disabled;
    setUsers(prev => prev.map(u => 
      u.uid === user.uid ? { ...u, disabled: newDisabled } : u
    ));
    
    try {
      const res = await fetch(`${API_BASE}/users/${user.uid}/status`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ disabled: newDisabled })
      });
      
      if (!res.ok) {
        // Revert on error
        setUsers(prev => prev.map(u => 
          u.uid === user.uid ? { ...u, disabled: !newDisabled } : u
        ));
        alert('Failed to update user status');
      }
    } catch (err) {
      // Revert on error
      setUsers(prev => prev.map(u => 
        u.uid === user.uid ? { ...u, disabled: !newDisabled } : u
      ));
      console.error('Error updating user status:', err);
      alert('Failed to update user status');
    }
  };

  if (loading) {
    return (
      <div className="user-management-container">
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Loading users...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="user-management-container">
        <div className="error-state">
          <span className="error-icon">!</span>
          <p>{error}</p>
          <button onClick={fetchData} className="retry-btn">Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="user-management-container">
      {/* Header */}
      <div className="user-header">
        <div className="header-title-group">
          <h2>User Management</h2>
          <span className="user-count">{users.length} users</span>
        </div>
        <button onClick={fetchData} className="refresh-btn">
          Refresh
        </button>
      </div>

      {/* Table */}
      <div className="table-container">
        {users.length === 0 ? (
          <div className="empty-state">
            <span className="empty-icon">No Users</span>
            <p>No users found</p>
          </div>
        ) : (
          <table className="user-table">
            <thead>
              <tr>
                <th>Email</th>
                <th>Display Name</th>
                <th>Cameras</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map(user => (
                <tr key={user.uid}>
                  <td className="email-cell">{user.email}</td>
                  <td className="name-cell">{user.displayName || '-'}</td>
                  <td>
                    <span className={`cameras-badge ${user.camerasCount > 0 ? 'has-cameras' : ''}`}>
                      {user.camerasCount} camera{user.camerasCount !== 1 ? 's' : ''}
                    </span>
                  </td>
                  <td>
                    <span className={`status-badge ${user.disabled ? 'disabled' : 'active'}`}>
                      {user.disabled ? 'Disabled' : 'Active'}
                    </span>
                  </td>
                  <td className="actions-cell">
                    <button 
                      className="action-btn assign"
                      onClick={() => openAssignModal(user)}
                    >
                      Assign Cameras
                    </button>
                    <button 
                      className={`action-btn toggle-status ${user.disabled ? 'enable' : 'disable'}`}
                      onClick={() => toggleUserStatus(user)}
                    >
                      {user.disabled ? 'Enable' : 'Disable'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Assign Cameras Modal */}
      {showModal && selectedUser && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Assign Cameras</h3>
              <button className="close-btn" onClick={() => setShowModal(false)}>x</button>
            </div>
            
            <div className="modal-user-info">
              <strong>{selectedUser.email}</strong>
              {selectedUser.displayName && <span> ({selectedUser.displayName})</span>}
            </div>
            
            <div className="camera-list">
              {cameras.map(camera => {
                const isAssigned = selectedCameras.includes(camera.id);
                const isOtherUser = camera.owner_uid && camera.owner_uid !== selectedUser.uid;
                
                return (
                  <label 
                    key={camera.id} 
                    className={`camera-item ${isAssigned ? 'selected' : ''} ${isOtherUser && !isAssigned ? 'assigned-other' : ''}`}
                  >
                    <input
                      type="checkbox"
                      checked={isAssigned}
                      onChange={() => toggleCamera(camera.id)}
                    />
                    <div className="camera-info">
                      <span className="camera-name">{camera.name}</span>
                      <span className="camera-location">{camera.location}</span>
                      {isOtherUser && !isAssigned && (
                        <span className="camera-owner">Currently: {camera.owner_email || 'Unknown user'}</span>
                      )}
                    </div>
                  </label>
                );
              })}
            </div>
            
            <div className="modal-actions">
              <button className="btn-cancel" onClick={() => setShowModal(false)}>
                Cancel
              </button>
              <button 
                className="btn-save" 
                onClick={saveAssignments}
                disabled={saving}
              >
                {saving ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UserManagement;
