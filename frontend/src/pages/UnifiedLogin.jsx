import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import PatientLogin from './Patient/PatientLogin';
import DoctorLogin from './Doctor/DoctorLogin';
import AdminLogin from './Admin/AdminLogin';

export default function UnifiedLogin() {
    const [searchParams] = useSearchParams();
    const [role, setRole] = useState('patient');
    const navigate = useNavigate();

    useEffect(() => {
        const roleParam = searchParams.get('role');
        if (roleParam && ['patient', 'doctor', 'admin'].includes(roleParam)) {
            setRole(roleParam);
        }
    }, [searchParams]);

    const handleRoleChange = (newRole) => {
        setRole(newRole);
        navigate(`/login?role=${newRole}`);
    };

    return (
        <div className="flex-1 min-h-screen bg-slate-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-xl shadow-lg border border-slate-100">
                <div className="text-center">
                    <img src="/logo.jpg" alt="MediScan Logo" className="h-16 w-auto mx-auto mb-4 mix-blend-multiply" />
                    <h2 className="mt-2 text-3xl font-extrabold text-slate-900">
                        Welcome Back
                    </h2>
                    <p className="mt-2 text-sm text-slate-600">
                        Please sign in to your account
                    </p>
                </div>

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
                            ? 'bg-white text-blue-600 shadow-sm'
                            : 'text-slate-500 hover:text-slate-700'
                            }`}
                    >
                        Doctor
                    </button>
                    <button
                        onClick={() => handleRoleChange('admin')}
                        className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${role === 'admin'
                            ? 'bg-white text-purple-600 shadow-sm'
                            : 'text-slate-500 hover:text-slate-700'
                            }`}
                    >
                        Admin
                    </button>
                </div>

                <div className="mt-6">
                    {role === 'patient' && (
                        <div className="animate-fade-in">
                            <div className="mb-4 text-center">
                                <span className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-green-100 text-green-600 mb-2">
                                    <i className="fa-solid fa-user-injured text-xl"></i>
                                </span>
                                <h3 className="text-lg font-medium text-slate-900">Patient Login</h3>
                            </div>
                            <PatientLogin isEmbedded={true} />
                        </div>
                    )}

                    {role === 'doctor' && (
                        <div className="animate-fade-in">
                            <div className="mb-4 text-center">
                                <span className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-blue-100 text-blue-600 mb-2">
                                    <i className="fa-solid fa-user-doctor text-xl"></i>
                                </span>
                                <h3 className="text-lg font-medium text-slate-900">Doctor Login</h3>
                            </div>
                            <DoctorLogin isEmbedded={true} />
                        </div>
                    )}

                    {role === 'admin' && (
                        <div className="animate-fade-in">
                            <div className="mb-4 text-center">
                                <span className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-purple-100 text-purple-600 mb-2">
                                    <i className="fa-solid fa-shield-halved text-xl"></i>
                                </span>
                                <h3 className="text-lg font-medium text-slate-900">Admin Login</h3>
                            </div>
                            <AdminLogin isEmbedded={true} />
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
