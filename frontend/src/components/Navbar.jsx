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
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-20 md:h-24">
                    {/* Logo Section */}
                    <div className="flex items-center cursor-pointer" onClick={() => navigate('/')}>
                        <div className="shrink-0 flex items-center gap-3">
                            <div className="bg-white p-2 rounded-xl shadow-md">
                                <i className="fa-solid fa-heart-pulse text-2xl text-green-600"></i>
                            </div>
                            <div>
                                <h1 className="text-2xl font-black text-white tracking-tight">MediScan</h1>
                                <p className="text-[10px] text-green-100 font-medium tracking-widest uppercase">AI Diagnostics</p>
                            </div>
                        </div>
                    </div>

                    {/* Desktop Menu */}
                    <div className="hidden md:flex items-center gap-8">
                        {!user ? (
                            <>
                                <Link to="/" className="text-sm font-bold text-white hover:text-green-100 transition duration-200">Home</Link>
                                <Link to="/about" className="text-sm font-bold text-white hover:text-green-100 transition duration-200">About Us</Link>
                            </>
                        ) : (
                            <div className="flex items-center gap-6">
                                {user.role === 'patient' && user.plan === 'premium' && (
                                    <span className="bg-yellow-300 text-yellow-900 text-[10px] font-black px-3 py-1.5 rounded-full flex items-center gap-1.5 border-2 border-yellow-400 uppercase tracking-tighter shadow-sm transform hover:scale-105 transition">
                                        <i className="fa-solid fa-crown"></i> Pro
                                    </span>
                                )}

                                <Link
                                    to={getDashboardLink()}
                                    className="text-sm font-bold text-white hover:text-green-100 transition duration-200 flex items-center gap-2 bg-green-600/30 px-4 py-2 rounded-lg border border-green-400/30 hover:bg-green-600/50"
                                >
                                    <i className="fa-solid fa-chart-line"></i> Dashboard
                                </Link>

                                <div className="h-8 w-px bg-green-400/30"></div>

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
                                                    className="w-10 h-10 rounded-full object-cover border-2 border-white shadow-md group-hover:border-green-200 transition"
                                                />
                                            ) : (
                                                <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center text-green-600 shadow-md group-hover:bg-green-50 transition">
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
                                        className="w-10 h-10 flex items-center justify-center text-white/80 hover:text-white hover:bg-red-500/20 rounded-lg transition"
                                        title="Logout"
                                    >
                                        <i className="fa-solid fa-right-from-bracket text-lg"></i>
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="flex items-center md:hidden">
                        <button
                            onClick={() => setIsMenuOpen(!isMenuOpen)}
                            className="inline-flex items-center justify-center p-2 rounded-xl text-white hover:bg-green-700/50 transition-colors focus:outline-none"
                        >
                            <span className="sr-only">Open main menu</span>
                            <div className="w-6 h-6 flex flex-col justify-center items-center gap-1.5 relative">
                                <span className={`block w-5 h-0.5 bg-current rounded-full transition-all duration-300 ${isMenuOpen ? 'rotate-45 translate-y-2' : ''}`}></span>
                                <span className={`block w-5 h-0.5 bg-current rounded-full transition-all duration-300 ${isMenuOpen ? 'opacity-0' : ''}`}></span>
                                <span className={`block w-5 h-0.5 bg-current rounded-full transition-all duration-300 ${isMenuOpen ? '-rotate-45 -translate-y-2' : ''}`}></span>
                            </div>
                        </button>
                    </div>
                </div>
            </div>


            {/* Mobile Menu */}
            <div className={`md:hidden overflow-hidden transition-all duration-300 ease-in-out border-t border-green-600/30 bg-green-600 ${isMenuOpen ? 'max-h-[32rem] opacity-100' : 'max-h-0 opacity-0'}`}>
                <div className="px-4 pt-4 pb-6 space-y-3">
                    {!user ? (
                        <div className="space-y-3">
                            <Link
                                to="/"
                                className="block px-4 py-3 rounded-xl text-base font-bold text-white hover:bg-green-500 transition border border-transparent hover:border-green-400"
                                onClick={() => setIsMenuOpen(false)}
                            >
                                <i className="fa-solid fa-house mr-3 text-green-200"></i> Home
                            </Link>
                            <Link
                                to="/about"
                                className="block px-4 py-3 rounded-xl text-base font-bold text-white hover:bg-green-500 transition border border-transparent hover:border-green-400"
                                onClick={() => setIsMenuOpen(false)}
                            >
                                <i className="fa-solid fa-circle-info mr-3 text-green-200"></i> About Us
                            </Link>

                            <div className="h-px bg-green-500 my-2"></div>

                            <Link
                                to="/login"
                                className="block px-4 py-3 rounded-xl text-base font-bold text-white hover:bg-green-500 transition text-center"
                                onClick={() => setIsMenuOpen(false)}
                            >
                                Login
                            </Link>
                            <Link
                                to="/signup"
                                className="block px-4 py-3 rounded-xl text-base font-bold text-green-700 bg-white hover:bg-green-50 transition text-center shadow-md"
                                onClick={() => setIsMenuOpen(false)}
                            >
                                Sign Up Now
                            </Link>
                        </div>
                    ) : (
                        <div className="space-y-2">
                            <div className="flex items-center gap-3 px-4 py-4 mb-2 bg-green-700/30 rounded-2xl border border-green-500/30">
                                {user.profileImage ? (
                                    <img src={user.profileImage} alt="" className="w-12 h-12 rounded-full border-2 border-white shadow-sm" />
                                ) : (
                                    <div className="w-12 h-12 rounded-full bg-white flex items-center justify-center text-green-600 shadow-sm">
                                        <i className="fa-solid fa-user text-lg"></i>
                                    </div>
                                )}
                                <div>
                                    <div className="text-base font-black text-white">{user.role.charAt(0).toUpperCase() + user.role.slice(1)}</div>
                                    <div className="text-[10px] text-green-200 uppercase tracking-wider font-bold">{user.plan || 'Free'} Plan</div>
                                </div>
                            </div>

                            <Link
                                to={getDashboardLink()}
                                className="block px-4 py-3 rounded-xl text-base font-bold text-white hover:bg-green-500 transition flex items-center"
                                onClick={() => setIsMenuOpen(false)}
                            >
                                <i className="fa-solid fa-chart-line mr-3 text-green-200"></i> Dashboard
                            </Link>
                            <Link
                                to="/profile"
                                className="block px-4 py-3 rounded-xl text-base font-bold text-white hover:bg-green-500 transition flex items-center"
                                onClick={() => setIsMenuOpen(false)}
                            >
                                <i className="fa-solid fa-user-gear mr-3 text-green-200"></i> Settings
                            </Link>

                            <div className="h-px bg-green-500 my-2"></div>

                            <button
                                onClick={() => {
                                    logout();
                                    navigate('/');
                                    setIsMenuOpen(false);
                                }}
                                className="block w-full text-left px-4 py-3 rounded-xl text-base font-bold text-red-200 hover:text-red-100 hover:bg-red-500/20 transition mt-2"
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
