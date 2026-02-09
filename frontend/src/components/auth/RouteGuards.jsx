import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useSelector } from 'react-redux';

export const ProtectedRoute = ({ children, role }) => {
    const { isAuthenticated, user, initialized } = useSelector(state => state.auth);
    const location = useLocation();

    if (!initialized) return null; // App.jsx already handles global loader, but safe to keep

    if (!isAuthenticated) {
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    if (role && user?.role?.toLowerCase() !== role.toLowerCase()) {
        // If user is doctor and tries to access patient dashboard, redirect to doctor dashboard (placeholder)
        if (user?.role?.toLowerCase() === 'doctor') return <Navigate to="/doctor-dashboard" replace />;
        return <Navigate to="/" replace />;
    }

    return children;
};

export const PublicRoute = ({ children }) => {
    const { isAuthenticated, user, initialized } = useSelector(state => state.auth);

    if (!initialized) return null;

    if (isAuthenticated) {
        if (user?.role?.toLowerCase() === 'patient') return <Navigate to="/dashboard" replace />;
        if (user?.role?.toLowerCase() === 'doctor') return <Navigate to="/doctor-dashboard" replace />;
        return <Navigate to="/" replace />;
    }

    return children;
};
