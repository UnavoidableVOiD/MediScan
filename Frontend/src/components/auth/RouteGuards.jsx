<<<<<<< Updated upstream
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
=======
import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useSelector } from 'react-redux';

export const ProtectedRoute = ({ children, role }) => {
    const { isAuthenticated, user, initialized } = useSelector(state => state.auth);
    const location = useLocation();

    if (!initialized) return null;

    if (!isAuthenticated) {
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    // Role-based access control
    if (role && user?.role?.toLowerCase() !== role.toLowerCase()) {
        const targetPath = user?.role?.toLowerCase() === 'doctor' ? '/doctor-dashboard' : '/dashboard';
        return <Navigate to={targetPath} replace />;
    }

    // Special redirection for shared /profile path
    if (location.pathname === '/profile' && user?.role === 'DOCTOR') {
        return <Navigate to="/doctor-profile" replace />;
    }

    // Handle Clinical Verification for Doctors
    if (user?.role === 'DOCTOR' && user?.doctor_status === 'UNVERIFIED') {
        const allowedPaths = ['/doctor-profile', '/verify-doctor', '/about', '/contact', '/services'];
        if (!allowedPaths.includes(location.pathname)) {
            return <Navigate to="/doctor-profile" replace />;
        }
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

export const AdminRoute = ({ children }) => {
    const { isAuthenticated, user, initialized } = useSelector(state => state.auth);
    const location = useLocation();

    if (!initialized) return null;

    if (!isAuthenticated) {
        return <Navigate to="/admin/login" state={{ from: location }} replace />;
    }

    if (user?.role !== 'ADMIN' && !user?.is_superuser) {
        return <Navigate to="/" replace />;
    }

    return children;
};
>>>>>>> Stashed changes
