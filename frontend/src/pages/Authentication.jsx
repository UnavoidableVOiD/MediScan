import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams, useLocation, Link } from 'react-router-dom';
import PatientLogin from './Patient/PatientLogin';
import PatientSignup from './Patient/PatientSignup';
import DoctorLogin from './Doctor/DoctorLogin';
import DoctorSignup from './Doctor/DoctorSignup';
import AdminLogin from './Admin/AdminLogin';

export default function Authentication() {
    const [searchParams] = useSearchParams();
    const location = useLocation();
    const navigate = useNavigate();

    // State for Role ('patient', 'doctor', 'admin') and Mode ('login', 'signup')
    const [role, setRole] = useState('patient');
    const [mode, setMode] = useState('login');

    useEffect(() => {
        // Determine Mode based on path
        if (location.pathname.includes('signup')) {
            setMode('signup');
        } else {
            setMode('login');
        }

        // Determine Role based on query param - Only allow 'patient' or 'admin'
        const roleParam = searchParams.get('role');
        if (roleParam && ['patient', 'admin'].includes(roleParam)) {
            setRole(roleParam);
        } else {
            setRole('patient');
        }
    }, [location.pathname, searchParams]);

    const handleModeChange = (newMode) => {
        setMode(newMode);
        const path = newMode === 'signup' ? '/signup' : '/login';
        navigate(`${path}?role=${role}`);
    };

    return (
        <div className="flex min-h-screen bg-white">
            {/* Left Side - Visual / Branding */}
            <div className="hidden lg:flex lg:w-1/2 relative bg-slate-900 overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-green-600 to-teal-900 opacity-90 z-10"></div>

                {/* Background Image */}
                <img
                    src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80"
                    alt="Medical Background"
                    className="absolute inset-0 w-full h-full object-cover mix-blend-overlay opacity-50"
                />

                <div className="relative z-20 flex flex-col justify-between p-16 w-full text-white">
                    <div>
                        <div className="w-16 h-16 bg-white/10 backdrop-blur-lg rounded-2xl flex items-center justify-center mb-8 border border-white/20">
                            <i className="fa-solid fa-heart-pulse text-3xl text-green-300"></i>
                        </div>
                        <h2 className="text-5xl font-black mb-6 tracking-tight leading-tight">
                            Advanced AI <br />
                            <span className="text-green-300">Diagnostics</span>
                        </h2>
                        <p className="text-xl text-green-100/80 max-w-md leading-relaxed">
                            Upload your medical reports and get instant, easy-to-understand insights powered by state-of-the-art artificial intelligence.
                        </p>
                    </div>

                    <div className="space-y-6">
                        <div className="flex items-center gap-4">
                            <div className="flex -space-x-4">
                                <img className="w-12 h-12 rounded-full border-2 border-slate-900" src="https://i.pravatar.cc/150?u=1" alt="" />
                                <img className="w-12 h-12 rounded-full border-2 border-slate-900" src="https://i.pravatar.cc/150?u=2" alt="" />
                                <img className="w-12 h-12 rounded-full border-2 border-slate-900" src="https://i.pravatar.cc/150?u=3" alt="" />
                                <div className="w-12 h-12 rounded-full border-2 border-slate-900 bg-green-600 flex items-center justify-center text-xs font-bold">10k+</div>
                            </div>
                            <div>
                                <p className="font-bold">Trusted by Users</p>
                                <div className="flex text-yellow-400 text-xs gap-0.5">
                                    <i className="fa-solid fa-star"></i>
                                    <i className="fa-solid fa-star"></i>
                                    <i className="fa-solid fa-star"></i>
                                    <i className="fa-solid fa-star"></i>
                                    <i className="fa-solid fa-star"></i>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Right Side - Form */}
            <div className="w-full lg:w-1/2 flex items-center justify-center p-8 bg-slate-50/50">
                <div className="max-w-md w-full">
                    {/* Simplified Header */}
                    <div className="text-left mb-10">
                        <div className="inline-flex items-center gap-2 mb-2 lg:hidden">
                            <i className="fa-solid fa-heart-pulse text-green-600 text-xl"></i>
                            <span className="font-black text-slate-900 tracking-tight">MediScan</span>
                        </div>
                        <h2 className="text-3xl font-bold text-slate-900">
                            {mode === 'login' ? 'Welcome back' : 'Create an account'}
                        </h2>
                        <p className="text-slate-500 mt-2">
                            {mode === 'login'
                                ? 'Enter your details to access your dashboard'
                                : 'Start your journey to better health understanding'}
                        </p>
                    </div>

                    {/* Role Toggles */}
                    <div className="flex p-1 bg-slate-100 rounded-xl mb-6">
                        <button
                            onClick={() => {
                                setRole('patient');
                                navigate(`${mode === 'signup' ? '/signup' : '/login'}?role=patient`);
                            }}
                            className={`flex-1 py-2 text-sm font-bold rounded-lg transition-all duration-200 ${role === 'patient'
                                ? 'bg-white text-green-600 shadow-sm'
                                : 'text-slate-500 hover:text-slate-700'}`}
                        >
                            Patient
                        </button>
                        <button
                            onClick={() => {
                                setRole('doctor');
                                navigate(`${mode === 'signup' ? '/signup' : '/login'}?role=doctor`);
                            }}
                            className={`flex-1 py-2 text-sm font-bold rounded-lg transition-all duration-200 ${role === 'doctor'
                                ? 'bg-white text-green-600 shadow-sm'
                                : 'text-slate-500 hover:text-slate-700'}`}
                        >
                            Doctor
                        </button>
                    </div>

                    {/* Admin Toggle (Subtle - only show if admin is active or requested) */}
                    {role === 'admin' && (
                        <div className="mb-6 animate-fade-in">
                            <div className="bg-purple-50 text-purple-700 px-4 py-3 rounded-lg border border-purple-200 flex items-center justify-between">
                                <span className="text-sm font-bold flex items-center gap-2">
                                    <i className="fa-solid fa-shield-halved"></i> Admin Portal
                                </span>
                                <button
                                    onClick={() => navigate('/login')} // Switch back to patient
                                    className="text-xs font-semibold hover:underline"
                                >
                                    Switch to Patient
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Auth Content */}
                    <div className="bg-white p-8 rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-100">
                        <div className="animate-fade-in">
                            {role === 'patient' && mode === 'login' && <PatientLogin isEmbedded={true} />}
                            {role === 'patient' && mode === 'signup' && <PatientSignup isEmbedded={true} />}
                            {role === 'doctor' && mode === 'login' && <DoctorLogin isEmbedded={true} />}
                            {role === 'doctor' && mode === 'signup' && <DoctorSignup isEmbedded={true} />}
                            {role === 'admin' && <AdminLogin isEmbedded={true} />}
                        </div>

                        {/* Unified Toggle Footer */}
                        {role !== 'admin' && (
                            <div className="mt-6 pt-6 border-t border-slate-50 text-center">
                                <p className="text-sm text-slate-500">
                                    {mode === 'login' ? "Don't have an account? " : "Already have an account? "}
                                    <button
                                        onClick={() => handleModeChange(mode === 'login' ? 'signup' : 'login')}
                                        className="font-bold text-green-600 hover:text-green-700 transition"
                                    >
                                        {mode === 'login' ? 'Sign up for free' : 'Sign in'}
                                    </button>
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
