import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';

function DoctorDashboard() {
    const location = useLocation();
    const [welcomeMessage, setWelcomeMessage] = useState('');

    useEffect(() => {
        if (location.state?.welcomeMessage) {
            setWelcomeMessage(location.state.welcomeMessage);
            const timer = setTimeout(() => setWelcomeMessage(''), 5000);
            return () => clearTimeout(timer);
        }
    }, [location.state]);

    const stats = [
        { label: 'Today\'s Appointments', value: '8', icon: 'fa-calendar-check', color: 'bg-blue-500' },
        { label: 'Pending Reports', value: '12', icon: 'fa-file-medical', color: 'bg-orange-500' },
        { label: 'Total Patients', value: '154', icon: 'fa-user-group', color: 'bg-green-500' },
        { label: 'Consultations', value: '42', icon: 'fa-comment-medical', color: 'bg-purple-500' },
    ];

    return (
        <div className="flex-1 min-h-screen bg-slate-50 px-4 py-8">
            <div className="max-w-7xl mx-auto">
                {welcomeMessage && (
                    <div className="mb-6 bg-blue-100 border border-blue-200 text-blue-800 px-4 py-3 rounded-xl flex items-center justify-between animate-fade-in shadow-sm">
                        <div className="flex items-center gap-3">
                            <span className="bg-blue-200 text-blue-700 w-8 h-8 rounded-full flex items-center justify-center">
                                <i className={`fa-solid ${welcomeMessage.includes('Back') ? 'fa-user-doctor' : 'fa-hand-holding-medical'}`}></i>
                            </span>
                            <span className="font-semibold">{welcomeMessage}</span>
                        </div>
                        <button onClick={() => setWelcomeMessage('')} className="text-blue-600 hover:text-blue-800">
                            <i className="fa-solid fa-xmark"></i>
                        </button>
                    </div>
                )}

                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-slate-900 mb-2">Doctor Dashboard</h1>
                    <p className="text-slate-600">Welcome back, Dr. Sharma</p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    {stats.map((stat, index) => (
                        <div key={index} className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 flex items-center gap-4">
                            <div className={`${stat.color} w-12 h-12 rounded-lg flex items-center justify-center text-white text-xl`}>
                                <i className={`fa-solid ${stat.icon}`}></i>
                            </div>
                            <div>
                                <p className="text-sm text-slate-500 font-medium">{stat.label}</p>
                                <p className="text-2xl font-bold text-slate-900">{stat.value}</p>
                            </div>
                        </div>
                    ))}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
                        <h2 className="text-xl font-bold text-slate-900 mb-4">Patient Requests</h2>
                        <div className="space-y-4">
                            {[1, 2, 3].map((_, i) => (
                                <div key={i} className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <div className="w-10 h-10 rounded-full bg-slate-200" />
                                        <div>
                                            <p className="font-semibold text-slate-900 text-sm">Patient Name {i + 1}</p>
                                            <p className="text-xs text-slate-500">Blood Report Analysis</p>
                                        </div>
                                    </div>
                                    <button className="text-green-600 font-bold text-sm hover:underline">Review</button>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
                        <h2 className="text-xl font-bold text-slate-900 mb-4">Quick Actions</h2>
                        <div className="grid grid-cols-2 gap-4">
                            <button className="flex flex-col items-center justify-center p-4 rounded-lg border-2 border-dashed border-slate-200 hover:border-green-500 hover:text-green-600 transition group">
                                <i className="fa-solid fa-file-arrow-up text-2xl mb-2 text-slate-400 group-hover:text-green-500"></i>
                                <span className="text-sm font-semibold">Upload Report</span>
                            </button>
                            <button className="flex flex-col items-center justify-center p-4 rounded-lg border-2 border-dashed border-slate-200 hover:border-blue-500 hover:text-blue-600 transition group">
                                <i className="fa-solid fa-message-medical text-2xl mb-2 text-slate-400 group-hover:text-blue-500"></i>
                                <span className="text-sm font-semibold">Start Chat</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default DoctorDashboard;
