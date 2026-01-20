import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams, useLocation } from 'react-router-dom';
import PatientLogin from './Patient/PatientLogin';
import DoctorLogin from './Doctor/DoctorLogin';
import PatientSignup from './Patient/PatientSignup';
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

        // Determine Role based on query param
        const roleParam = searchParams.get('role');
        if (roleParam && ['patient', 'doctor', 'admin'].includes(roleParam)) {
            setRole(roleParam);
        }
    }, [location.pathname, searchParams]);

    const handleRoleChange = (newRole) => {
        setRole(newRole);
        updateUrl(mode, newRole);
    };

    const handleModeChange = (newMode) => {
        setMode(newMode);
        updateUrl(newMode, role);
    };

    const updateUrl = (newMode, newRole) => {
        const path = newMode === 'signup' ? '/signup' : '/login';
        navigate(`${path}?role=${newRole}`);
    };

    return (
        <div className="flex-1 min-h-screen bg-slate-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-md w-full space-y-6 bg-white p-8 rounded-xl shadow-lg border border-slate-100">
                <div className="text-center">
                    <img src="/logo.jpg" alt="MediScan Logo" className="h-16 w-auto mx-auto mb-4 mix-blend-multiply" />
                    <h2 className="mt-2 text-3xl font-extrabold text-slate-900">
                        {mode === 'login' ? 'Welcome Back' : 'Create Account'}
                    </h2>
                    <p className="mt-2 text-sm text-slate-600">
                        {mode === 'login' ? 'Please sign in to your account' : 'Join MediScan today'}
                    </p>
                </div>

                {/* Role Toggles (Patient / Doctor) */}
                <div className="flex rounded-lg bg-slate-100 p-1">
                    <button
                        onClick={() => handleRoleChange('patient')}
                        className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${role === 'patient'
                            ? 'bg-white text-green-600 shadow-sm'
                            : 'text-slate-500 hover:text-slate-700'
                            }`}
                    >
                        Patient
                    </button>
                    <button
                        onClick={() => handleRoleChange('doctor')}
                        className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${role === 'doctor'
                            ? 'bg-white text-green-600 shadow-sm'
                            : 'text-slate-500 hover:text-slate-700'
                            }`}
                    >
                        Doctor
                    </button>
                    {mode === 'login' && (
                        <button
                            onClick={() => handleRoleChange('admin')}
                            className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${role === 'admin'
                                ? 'bg-white text-purple-600 shadow-sm'
                                : 'text-slate-500 hover:text-slate-700'
                                }`}
                        >
                            Admin
                        </button>
                    )}
                </div>

                {/* Mode Toggles (Login / Sign Up) */}
                {role !== 'admin' && (
                    <div className="flex justify-center border-b border-slate-200 mb-6">
                        <button
                            onClick={() => handleModeChange('login')}
                            className={`pb-2 px-6 text-sm font-medium transition-all border-b-2 ${mode === 'login'
                                ? 'border-green-600 text-green-600'
                                : 'border-transparent text-slate-500 hover:text-slate-700'
                                }`}
                        >
                            Login
                        </button>
                        <button
                            onClick={() => handleModeChange('signup')}
                            className={`pb-2 px-6 text-sm font-medium transition-all border-b-2 ${mode === 'signup'
                                ? 'border-green-600 text-green-600'
                                : 'border-transparent text-slate-500 hover:text-slate-700'
                                }`}
                        >
                            Sign Up
                        </button>
                    </div>
                )}


                <div className="mt-6">
                    {/* Header Icon & Title */}
                    <div className="mb-4 text-center">
                        <span className={`inline-flex items-center justify-center w-12 h-12 rounded-full mb-2 ${role === 'admin' ? 'bg-purple-100 text-purple-600' : 'bg-green-100 text-green-600'
                            }`}>
                            <i className={`fa-solid ${role === 'patient' ? 'fa-user-injured' :
                                role === 'doctor' ? 'fa-user-doctor' :
                                    'fa-shield-halved'
                                } text-xl`}></i>
                        </span>
                        <h3 className="text-lg font-medium text-slate-900 capitalize">
                            {role} {mode === 'login' ? 'Login' : 'Registration'}
                        </h3>
                    </div>

                    {/* Content Rendering */}
                    <div className="animate-fade-in">
                        {role === 'patient' && mode === 'login' && <PatientLogin isEmbedded={true} />}
                        {role === 'patient' && mode === 'signup' && <PatientSignup isEmbedded={true} />}
                        {role === 'doctor' && mode === 'login' && <DoctorLogin isEmbedded={true} />}
                        {role === 'doctor' && mode === 'signup' && <DoctorSignup isEmbedded={true} />}
                        {role === 'admin' && <AdminLogin isEmbedded={true} />}
                    </div>
                </div>
            </div>
        </div>
    );
}
