import React, { createContext, useContext, useState } from 'react';

const AdminContext = createContext();

export function useAdmin() {
    return useContext(AdminContext);
}

export function AdminProvider({ children }) {
    const [admin, setAdmin] = useState(null);
    const [stats, setStats] = useState({
        totalUsers: 0,
        totalDoctors: 0,
        totalPatients: 0,
        verifiedDoctors: 0,
        pendingDoctors: 0,
        totalReports: 0
    });

    const login = (adminData) => {
        setAdmin(adminData);
    };

    const logout = () => {
        setAdmin(null);
    };

    const updateStats = (newStats) => {
        setStats(newStats);
    };

    const value = {
        admin,
        stats,
        login,
        logout,
        updateStats
    };

    return (
        <AdminContext.Provider value={value}>
            {children}
        </AdminContext.Provider>
    );
}
