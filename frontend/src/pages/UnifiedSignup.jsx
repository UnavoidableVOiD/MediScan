import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import PatientSignup from './Patient/PatientSignup';
import DoctorSignup from './Doctor/DoctorSignup';

export default function UnifiedSignup() {
    const [searchParams] = useSearchParams();
    const [role, setRole] = useState('patient');
    const navigate = useNavigate();

    useEffect(() => {
        const roleParam = searchParams.get('role');
        if (roleParam && ['patient', 'doctor'].includes(roleParam)) {
            setRole(roleParam);
        }
    }, [searchParams]);

    const handleRoleChange = (newRole) => {
        setRole(newRole);
        navigate(`/signup?role=${newRole}`);
    };

    return (
        <div className="flex-1 min-h-screen bg-slate-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-xl shadow-lg border border-slate-100">
                <div className="text-center">
                    <img src="/logo.jpg" alt="MediScan Logo" className="h-16 w-auto mx-auto mb-4 mix-blend-multiply" />
                    <h2 className="mt-2 text-3xl font-extrabold text-slate-900">
                        Create Account
                    </h2>
                    <p className="mt-2 text-sm text-slate-600">
                        Join MediScan today
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
                </div>

                <div className="mt-6">
                    {role === 'patient' && (
                        <div className="animate-fade-in">
                            <div className="mb-4 text-center">
                                <span className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-green-100 text-green-600 mb-2">
                                    <i className="fa-solid fa-user-injured text-xl"></i>
                                </span>
                                <h3 className="text-lg font-medium text-slate-900">Patient Registration</h3>
                            </div>
                            <PatientSignup isEmbedded={true} />
                        </div>
                    )}

                    {role === 'doctor' && (
                        <div className="animate-fade-in">
                            <div className="mb-4 text-center">
                                <span className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-blue-100 text-blue-600 mb-2">
                                    <i className="fa-solid fa-user-doctor text-xl"></i>
                                </span>
                                <h3 className="text-lg font-medium text-slate-900">Doctor Registration</h3>
                            </div>
                            <DoctorSignup isEmbedded={true} />
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
