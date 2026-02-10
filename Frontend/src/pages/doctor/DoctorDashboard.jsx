import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    User,
    Clock,
    CheckCircle2,
    Sparkles,
    Search,
    Calendar,
    ArrowRight,
    BadgeCheck,
    ArrowUpRight,
    ChevronRight,
    Activity,
    Users,
    Loader2
} from 'lucide-react';
import { useSelector, useDispatch } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { fetchDoctorStats, fetchMyPatients } from '../../store/slices/doctorSlice';
import { fetchAppointments } from '../../store/slices/appointmentSlice';

const StatCard = ({ title, count, icon: Icon, color, trend }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-3xl p-6 shadow-sm border border-gray-100 flex flex-col justify-between gap-4 hover:shadow-md transition-all"
    >
        <div className="flex items-start justify-between">
            <div className={`w-12 h-12 ${color} rounded-2xl flex items-center justify-center text-white shadow-lg shadow-${color}/30`}>
                <Icon className="w-6 h-6" />
            </div>
            <span className="text-xs font-bold text-gray-400 bg-gray-50 px-2.5 py-1 rounded-lg">{trend}</span>
        </div>
        <div>
            <p className="text-3xl font-bold text-gray-900 tracking-tight">{count}</p>
            <p className="text-sm text-gray-500 font-medium mt-1">{title}</p>
        </div>
    </motion.div>
);

const DoctorDashboard = () => {
    const { user } = useSelector(state => state.auth);
    const { stats, patients, statsLoading, loading } = useSelector(state => state.doctor);
    const { appointments, loading: appointmentsLoading } = useSelector(state => state.appointment);
    const dispatch = useDispatch();
    const navigate = useNavigate();

    useEffect(() => {
        dispatch(fetchDoctorStats());
        dispatch(fetchMyPatients());
        dispatch(fetchAppointments());
    }, [dispatch]);

    const statsData = stats || {
        total_patients: 0,
        ongoing_patients: 0,
        completed_patients: 0,
        new_patients_7_days: 0
    };

    const statCards = [
        { title: "Total Patients Treated", count: statsData.total_patients, icon: Users, color: "bg-blue-500", trend: "Total" },
        { title: "Ongoing Patients", count: statsData.ongoing_patients, icon: Clock, color: "bg-orange-500", trend: "Active" },
        { title: "Completed Patients", count: statsData.completed_patients, icon: CheckCircle2, color: "bg-green-500", trend: "Done" },
        { title: "New Patients", count: statsData.new_patients_7_days, icon: Sparkles, color: "bg-purple-500", trend: "7 Days" }
    ];

    if (loading || statsLoading) return (
        <div className="min-h-screen flex items-center justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-medic-dark"></div>
        </div>
    );

    return (
        <div className="min-h-[calc(100vh-80px)] bg-neutral-background py-8 px-6 space-y-8">
            <div className="max-w-7xl mx-auto space-y-8">
                {/* Header Section */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                    <div className="space-y-1">
                        <div className="flex items-center gap-3">
                            <h1 className="text-3xl font-black text-gray-900 tracking-tight">
                                Hello, Dr. {user?.last_name || 'Doe'}
                            </h1>
                            <div className="flex items-center gap-1.5 px-3 py-1 bg-green-50 text-green-600 rounded-full text-xs font-black border border-green-100 shadow-sm">
                                <BadgeCheck size={14} />
                                VERIFIED
                            </div>
                        </div>
                        <p className="text-gray-500 font-medium">Welcome back to your clinical dashboard.</p>
                    </div>

                    <div className="flex items-center gap-3">
                        <div className="relative group hidden sm:block">
                            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 group-focus-within:text-medic-dark transition-colors" />
                            <input
                                type="text"
                                placeholder="Search patients..."
                                className="pl-11 pr-6 py-3 bg-white border border-gray-100 rounded-2xl outline-none focus:border-medic-dark focus:shadow-xl focus:shadow-medic-dark/5 transition-all w-64 text-sm"
                            />
                        </div>
                        <button className="p-3 bg-white border border-gray-100 rounded-2xl hover:bg-neutral-soft transition-colors shadow-sm">
                            <Calendar className="w-5 h-5 text-gray-600" />
                        </button>
                    </div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                    {statCards.map((stat, idx) => (
                        <StatCard key={idx} {...stat} />
                    ))}
                </div>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Recent Patients Table */}
                    <div className="lg:col-span-2 space-y-4">
                        <div className="flex items-center justify-between">
                            <h2 className="text-xl font-black text-gray-900 tracking-tight">Recent Patients</h2>
                            <button
                                onClick={() => navigate('/patients')}
                                className="text-sm font-bold text-medic-dark hover:text-medic-primary flex items-center gap-1 transition-colors"
                            >
                                View All
                                <ChevronRight size={16} />
                            </button>
                        </div>

                        <div className="bg-white rounded-[2rem] shadow-xl shadow-medic-dark/5 border border-medic-light/20 overflow-hidden">
                            <div className="overflow-x-auto">
                                <table className="w-full text-left">
                                    <thead>
                                        <tr className="bg-neutral-soft/50 border-b border-gray-50">
                                            <th className="px-8 py-5 text-gray-500 text-xs font-black uppercase tracking-widest">Patient Name</th>
                                            <th className="px-8 py-5 text-gray-500 text-xs font-black uppercase tracking-widest">Department</th>
                                            <th className="px-8 py-5 text-gray-500 text-xs font-black uppercase tracking-widest">Status</th>
                                            <th className="px-8 py-5 text-gray-500 text-xs font-black uppercase tracking-widest text-right">Last Updated</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-50">
                                        {patients.length > 0 ? patients.map((patient) => (
                                            <tr
                                                key={patient.id}
                                                onClick={() => navigate(`/patient/${patient.id}`)}
                                                className="hover:bg-neutral-soft/20 transition-colors group cursor-pointer"
                                            >
                                                <td className="px-8 py-5">
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-9 h-9 rounded-xl bg-medic-light/30 flex items-center justify-center font-black text-medic-dark uppercase">
                                                            {patient.first_name[0]}
                                                        </div>
                                                        <span className="font-bold text-gray-900">{patient.first_name} {patient.last_name}</span>
                                                    </div>
                                                </td>
                                                <td className="px-8 py-5">
                                                    <span className="text-sm text-gray-500 font-medium">{patient.condition}</span>
                                                </td>
                                                <td className="px-8 py-5">
                                                    <span className={`px-4 py-1.5 rounded-full text-xs font-black border uppercase ${patient.status === 'ONGOING'
                                                        ? 'bg-blue-50 text-blue-600 border-blue-100'
                                                        : 'bg-green-50 text-green-600 border-green-100'
                                                        }`}>
                                                        {patient.status}
                                                    </span>
                                                </td>
                                                <td className="px-8 py-5 text-right">
                                                    <span className="text-sm text-gray-400 font-semibold">
                                                        {patient.last_visit ? new Date(patient.last_visit).toLocaleDateString() : 'Never'}
                                                    </span>
                                                </td>
                                            </tr>
                                        )) : (
                                            <tr>
                                                <td colSpan="4" className="px-8 py-10 text-center text-gray-400 font-bold">
                                                    No recent patients found.
                                                </td>
                                            </tr>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    {/* Quick Access / Actions */}
                    <div className="space-y-6">
                        <div className="space-y-4">
                            <h2 className="text-xl font-black text-gray-900 tracking-tight">Today's Schedule</h2>
                            <div className="space-y-3">
                                {appointmentsLoading ? (
                                    <div className="flex justify-center p-6"><Loader2 className="animate-spin text-medic-dark" /></div>
                                ) : appointments.filter(a => a.status === 'PAID').length > 0 ? (
                                    appointments.filter(a => a.status === 'PAID').map((appt) => (
                                        <div key={appt.id} className="p-4 bg-white border border-gray-100 rounded-2xl shadow-sm flex items-center gap-4">
                                            <div className="w-10 h-10 bg-medic-light/30 rounded-xl flex items-center justify-center text-medic-dark font-black text-sm">
                                                {appt.start_time.slice(0, 5)}
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <h4 className="font-bold text-gray-900 truncate">{appt.patient_email.split('@')[0]}</h4>
                                                <p className="text-[10px] text-gray-400 font-bold uppercase truncate">{appt.notes || 'No notes'}</p>
                                            </div>
                                            <div className="w-2 h-2 rounded-full bg-green-500 shadow-sm shadow-green-200"></div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="p-10 bg-neutral-soft/30 rounded-3xl border border-dashed border-gray-200 text-center">
                                        <p className="text-gray-400 font-medium italic text-xs">No pending appointments.</p>
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="space-y-4">
                            <h2 className="text-xl font-black text-gray-900 tracking-tight">Quick Actions</h2>
                            <div className="grid grid-cols-1 gap-4">
                                {[
                                    { title: 'My Schedule', subtitle: 'View all appointments', color: 'bg-medic-dark', icon: Calendar, onClick: () => navigate('/appointments') },
                                    { title: 'Manage Availability', subtitle: 'Set your hours', color: 'bg-medic-primary', icon: Activity, onClick: () => navigate('/doctor-profile') },
                                    { title: 'Patient Records', subtitle: 'Access history', color: 'bg-neutral-dark', icon: Users, onClick: () => navigate('/patients') },
                                ].map((action, idx) => (
                                    <button
                                        key={idx}
                                        onClick={action.onClick}
                                        className="p-6 rounded-3xl bg-white border border-medic-light/20 shadow-lg shadow-medic-dark/5 hover:border-medic-dark transition-all text-left flex items-start gap-4 group w-full"
                                    >
                                        <div className={`p-3 rounded-2xl ${action.color} text-white group-hover:scale-110 transition-transform`}>
                                            <action.icon size={20} />
                                        </div>
                                        <div>
                                            <h4 className="font-black text-gray-900 capitalize">{action.title}</h4>
                                            <p className="text-xs text-gray-500 mt-1">{action.subtitle}</p>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DoctorDashboard;
