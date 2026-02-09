import React, { useState } from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import {
    Users,
    UserPlus,
    Stethoscope,
    LogOut,
    LayoutDashboard,
    Menu,
    X,
    ShieldCheck
} from 'lucide-react';
import { logoutUser } from '../store/slices/authSlice';

const AdminLayout = () => {
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const { user } = useSelector((state) => state.auth);
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);

    const handleLogout = () => {
        dispatch(logoutUser());
        navigate('/admin/login');
    };

    const navItems = [
        { path: '/admin/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
        { path: '/admin/doctors', icon: Stethoscope, label: 'Doctors' },
        { path: '/admin/patients', icon: Users, label: 'Patients' },
    ];

    if (user?.is_superuser) {
        navItems.push({ path: '/admin/create-admin', icon: UserPlus, label: 'Create Admin' });
    }

    return (
        <div className="flex h-screen bg-neutral-background overflow-hidden relative">
            {/* Sidebar */}
            <aside
                className={`${isSidebarOpen ? 'w-64 translate-x-0' : 'w-0 -translate-x-full'} 
                bg-medic-dark text-white transition-all duration-300 ease-in-out fixed inset-y-0 left-0 z-50 md:relative md:translate-x-0 flex flex-col`}
            >
                <div className="p-6 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <ShieldCheck className="w-8 h-8 text-medic-accent" />
                        <span className="font-bold text-xl">Admin Portal</span>
                    </div>
                    <button onClick={() => setIsSidebarOpen(false)} className="md:hidden">
                        <X className="w-6 h-6" />
                    </button>
                </div>

                <nav className="flex-1 px-4 py-6 space-y-2">
                    {navItems.map((item) => (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            className={({ isActive }) =>
                                `flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${isActive
                                    ? 'bg-medic-accent text-medic-dark font-bold'
                                    : 'text-gray-300 hover:bg-white/10 hover:text-white'
                                }`
                            }
                        >
                            <item.icon className="w-5 h-5" />
                            <span>{item.label}</span>
                        </NavLink>
                    ))}
                </nav>

                <div className="p-4 border-t border-white/10">
                    <button
                        onClick={handleLogout}
                        className="flex items-center gap-3 w-full px-4 py-3 text-red-400 hover:bg-red-500/10 hover:text-red-300 rounded-xl transition-all"
                    >
                        <LogOut className="w-5 h-5" />
                        <span>Logout</span>
                    </button>
                    <div className="mt-4 px-4 text-xs text-gray-500">
                        Logged in as: <br />
                        <span className="font-bold text-gray-300 truncate block">{user?.email}</span>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
                {/* Mobile Header */}
                <header className={`${isSidebarOpen ? 'md:hidden' : 'flex'} md:hidden bg-white p-4 items-center justify-between shadow-sm z-40`}>
                    <button onClick={() => setIsSidebarOpen(true)} className="p-2 text-medic-dark">
                        <Menu className="w-6 h-6" />
                    </button>
                    <span className="font-bold text-medic-dark">MedisAdmin</span>
                    <div className="w-8" />
                </header>

                <div className="flex-1 overflow-auto p-4 md:p-8 relative">
                    <Outlet />
                </div>
            </main>

            {/* Overlay for mobile */}
            {isSidebarOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-40 md:hidden"
                    onClick={() => setIsSidebarOpen(false)}
                />
            )}
        </div>
    );
};

export default AdminLayout;
