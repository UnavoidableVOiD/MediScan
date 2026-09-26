import React, { createContext, useContext, useMemo, useState } from 'react';
export const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('mediScanUser');
    return saved ? JSON.parse(saved) : null;
  });

  const login = (role) => {
    const savedProfile = localStorage.getItem(`mediScanProfile_${role}`);
    let userData;

    if (savedProfile) {
      userData = JSON.parse(savedProfile);
      userData.role = role;
    } else {
      if (role === 'patient') {
        userData = {
          role,
          name: 'John Doe',
          email: 'john@example.com',
          phone: '+977 9800000000',
          age: 45,
          bloodGroup: 'O+',
          address: 'Kathmandu, Nepal',
          plan: 'free'
        };
      } else if (role === 'doctor') {
        userData = {
          role,
          name: 'Dr. Strange',
          email: 'doctor@mediscan.com',
          specialty: 'General Physician',
          license: 'NMC-12345',
          hospital: 'City General Hospital'
        };
      } else {
        userData = { role, name: 'System Admin', email: 'admin@mediscan.com' };
      }
    }

    if (role === 'patient' && !userData.plan) userData.plan = 'free';

    setUser(userData);
    localStorage.setItem('mediScanUser', JSON.stringify(userData));
  };

  const updateUser = (newData) => {
    setUser((prev) => {
      const updated = { ...prev, ...newData };
      localStorage.setItem('mediScanUser', JSON.stringify(updated));
      localStorage.setItem(`mediScanProfile_${updated.role}`, JSON.stringify(updated));
      return updated;
    });
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('mediScanUser');
  };

  const value = useMemo(() => ({ user, login, logout, updateUser }), [user]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}