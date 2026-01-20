import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext.jsx';

export default function Navbar() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const getDashboardLink = () => {
        if (!user) return '/';
        if (user.role === 'admin') return '/admin/dashboard';
        if (user.role === 'doctor') return '/doctor/dashboard';
        return '/patient/dashboard';
    };

    return (
        <nav className="bg-gradient-to-r from-green-600 to-green-500 backdrop-blur-md border-b border-green-700 shadow-lg z-50 sticky top-0 w-full">
            <div className="max-w-7xl mx-auto px-0 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-20 md:h-24 px-4">
                    <div className="flex items-center cursor-pointer" onClick={() => navigate('/')}>
                        <div className="shrink-0 flex items-center">
                            <img src="/logo.jpg" alt="MediScan Logo" className="h-16 md:h-20 w-auto mix-blend-multiply" />
                        </div>
                    </div>

                    {/* Desktop Menu */}
                    <div className="hidden md:flex items-center gap-8 pr-6">
                        {!user ? (
                            <>
                                <Link to="/" className="text-sm font-semibold text-white hover:text-green-100 transition duration-200">Home</Link>
                                <Link to="/admin/login" className="text-sm font-semibold text-white hover:text-green-100 transition duration-200">Admin</Link>
                                <Link to="/about" className="text-sm font-semibold text-white hover:text-green-100 transition duration-200">About Us</Link>
                            </>
                        ) : (
                            <div className="flex items-center gap-8">
                                {user.role === 'patient' && user.plan === 'premium' && (
                                    <span className="bg-yellow-300 text-yellow-900 text-[10px] font-black px-3 py-1.5 rounded-full flex items-center gap-1.5 border-2 border-yellow-400 uppercase tracking-tighter">
                                        <i className="fa-solid fa-crown"></i> Pro
                                    </span>
                                )}

                                <Link
                                    to={getDashboardLink()}
                                    className="text-sm font-semibold text-white hover:text-green-100 transition duration-200 flex items-center gap-2"
                                >
                                    Dashboard
                                </Link>

                                <div className="h-6 w-px bg-slate-200"></div>

                                <div className="flex items-center gap-4">
                                    <Link
                                        to="/profile"
                                        className="flex items-center gap-3 group"
                                    >
                                        <div className="relative">
                                            {user.profileImage ? (
                                                <img
                                                    src={user.profileImage}
                                                    alt="Profile"
                                                    className="w-10 h-10 rounded-full object-cover border-2 border-slate-100 group-hover:border-blue-400 transition"
                                                />
                                            ) : (
                                                <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center text-slate-500 group-hover:bg-blue-50 group-hover:text-blue-600 transition">
                                                    <i className={`fa-solid ${user.role === 'admin' ? 'fa-shield-halved' : user.role === 'doctor' ? 'fa-user-doctor' : 'fa-circle-user'} text-lg`}></i>
                                                </div>
                                            )}
                                        </div>
                                    </Link>

                                    <button
                                        onClick={() => {
                                            logout();
                                            navigate('/');
                                        }}
                                        className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition"
                                        title="Logout"
                                    >
                                        <i className="fa-solid fa-right-from-bracket text-lg"></i>
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="flex items-center md:hidden pr-4">
                        <button
                            onClick={() => setIsMenuOpen(!isMenuOpen)}
                            className="inline-flex items-center justify-center p-2 rounded-lg text-white hover:bg-green-700 transition-colors focus:outline-none"
                        >
                            <span className="sr-only">Open main menu</span>
                            <div className="w-6 h-6 flex flex-col justify-center items-center gap-1.5 relative">
                                <span className={`block w-5 h-0.5 bg-current transition-all duration-300 ${isMenuOpen ? 'rotate-45 translate-y-2' : ''}`}></span>
                                <span className={`block w-5 h-0.5 bg-current transition-all duration-300 ${isMenuOpen ? 'opacity-0' : ''}`}></span>
                                <span className={`block w-5 h-0.5 bg-current transition-all duration-300 ${isMenuOpen ? '-rotate-45 -translate-y-2' : ''}`}></span>
                            </div>
                        </button>
                    </div>
                </div>
            </div>


            {/* Mobile Menu */}
            <div className={`md:hidden overflow-hidden transition-all duration-300 ease-in-out border-t border-green-700 bg-green-500 ${isMenuOpen ? 'max-h-96' : 'max-h-0'}`}>
                <div className="px-4 pt-4 pb-6 space-y-3">
                    {!user ? (
                        <div className="space-y-3">
                            <Link
                                to="/"
                                className="block px-4 py-3 rounded-xl text-base font-medium text-white hover:bg-green-400 active:bg-green-700 transition"
                                onClick={() => setIsMenuOpen(false)}
                            >
                                <i className="fa-solid fa-house mr-3 text-green-100"></i> Home
                            </Link>
                            <Link
                                to="/admin/login"
                                className="block px-4 py-3 rounded-xl text-base font-medium text-white hover:bg-green-400 active:bg-green-700 transition"
                                onClick={() => setIsMenuOpen(false)}
                            >
                                <i className="fa-solid fa-shield-halved mr-3 text-green-100"></i> Admin
                            </Link>
                            <Link
                                to="/about"
                                className="block px-4 py-3 rounded-xl text-base font-medium text-white hover:bg-green-400 active:bg-green-700 transition"
                                onClick={() => setIsMenuOpen(false)}
                            >
                                <i className="fa-solid fa-circle-info mr-3 text-slate-400"></i> About Us
                            </Link>
                        </div>
                    ) : (
                        <div className="space-y-2">
                            <div className="flex items-center gap-3 px-4 py-3 mb-2 bg-slate-50 rounded-2xl">
                                {user.profileImage ? (
                                    <img src={user.profileImage} alt="" className="w-10 h-10 rounded-full border border-slate-200" />
                                ) : (
                                    <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center text-slate-400">
                                        <i className="fa-solid fa-user"></i>
                                    </div>
                                )}
                                <div>
                                    <div className="text-sm font-bold text-slate-900">{user.role.charAt(0).toUpperCase() + user.role.slice(1)}</div>
                                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">{user.plan || 'Free'} Plan</div>
                                </div>
                            </div>
                            <Link
                                to={getDashboardLink()}
                                className="block px-4 py-3 rounded-xl text-base font-medium text-slate-700 hover:bg-slate-50 active:bg-slate-100 transition"
                                onClick={() => setIsMenuOpen(false)}
                            >
                                <i className="fa-solid fa-chart-line mr-3 text-slate-400"></i> Dashboard
                            </Link>
                            <Link
                                to="/profile"
                                className="block px-4 py-3 rounded-xl text-base font-medium text-slate-700 hover:bg-slate-50 active:bg-slate-100 transition"
                                onClick={() => setIsMenuOpen(false)}
                            >
                                <i className="fa-solid fa-user-gear mr-3 text-slate-400"></i> Settings
                            </Link>
                            <button
                                onClick={() => {
                                    logout();
                                    navigate('/');
                                    setIsMenuOpen(false);
                                }}
                                className="block w-full text-left px-4 py-3 rounded-xl text-base font-medium text-red-500 hover:bg-red-50 active:bg-red-100 transition mt-4"
                            >
                                <i className="fa-solid fa-right-from-bracket mr-3"></i> Logout
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </nav >
    );
}
