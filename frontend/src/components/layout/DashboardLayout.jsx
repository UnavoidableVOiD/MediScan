import React, { useEffect } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import { useAuthStore } from '@/store/authStore';

const DashboardLayout = () => {
    const { user, isAuthenticated } = useAuthStore();
    const navigate = useNavigate();
    const location = useLocation();

    useEffect(() => {
        if (!isAuthenticated) {
            // Redirect to appropriate login based on path if possible, or generic
            if (location.pathname.includes('doctor')) navigate('/doctor/login');
            else if (location.pathname.includes('admin')) navigate('/admin/login');
            else navigate('/patient/login');
        }
    }, [isAuthenticated, navigate, location.pathname]);

    if (!isAuthenticated) return null; // or loading spinner

    return (
        <div className="min-h-screen bg-background flex">
            <Sidebar />
            <main className="flex-1 ml-64 p-8 overflow-y-auto h-screen bg-muted/10">
                <div className="max-w-6xl mx-auto">
                    <Outlet />
                </div>
            </main>
        </div>
    );
};

export default DashboardLayout;
