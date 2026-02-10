import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Calendar,
    Clock,
    User,
    Filter,
    Search,
    ChevronRight,
    MoreVertical,
    CheckCircle2,
    XCircle,
    Loader2,
    Mail,
    Phone,
    History
} from 'lucide-react';
import { useDispatch, useSelector } from 'react-redux';
import { fetchAppointments } from '../../store/slices/appointmentSlice';

const StatusBadge = ({ status }) => {
    const styles = {
        PAID: 'bg-green-50 text-green-600 border-green-100',
        COMPLETED: 'bg-blue-50 text-blue-600 border-blue-100',
        CANCELLED: 'bg-red-50 text-red-600 border-red-100',
        PENDING: 'bg-orange-50 text-orange-600 border-orange-100'
    };
    return (
        <span className={`px-4 py-1.5 rounded-full text-[10px] font-black tracking-widest uppercase border ${styles[status] || styles.PENDING}`}>
            {status}
        </span>
    );
};

const DoctorAppointments = () => {
    const dispatch = useDispatch();
    const { appointments, loading } = useSelector(state => state.appointment);
    const [searchTerm, setSearchTerm] = useState('');
    const [activeTab, setActiveTab] = useState('UPCOMING'); // UPCOMING or HISTORY

    useEffect(() => {
        dispatch(fetchAppointments());
    }, [dispatch]);

    const isUpcoming = (apptDate) => {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const date = new Date(apptDate);
        return date >= today;
    };

    const filteredAppointments = appointments.filter(appt => {
        const matchesSearch = appt.patient_email.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesTab = activeTab === 'UPCOMING'
            ? isUpcoming(appt.appointment_date) && appt.status !== 'CANCELLED' && appt.status !== 'COMPLETED'
            : !isUpcoming(appt.appointment_date) || appt.status === 'CANCELLED' || appt.status === 'COMPLETED';
        return matchesSearch && matchesTab;
    }).sort((a, b) => {
        if (activeTab === 'UPCOMING') {
            return new Date(a.appointment_date) - new Date(b.appointment_date);
        }
        return new Date(b.appointment_date) - new Date(a.appointment_date);
    });

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <Loader2 className="w-12 h-12 text-medic-dark animate-spin" />
            </div>
        );
    }

    return (
        <div className="min-h-[calc(100vh-80px)] bg-neutral-background py-8 px-6 space-y-8">
            <div className="max-w-7xl mx-auto space-y-8">
                {/* Header */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                    <div className="space-y-1">
                        <h1 className="text-3xl font-black text-gray-900 tracking-tight">Clinical Schedule</h1>
                        <p className="text-gray-500 font-medium">Manage and monitor all your patient appointments.</p>
                    </div>

                    <div className="flex items-center gap-4 flex-wrap">
                        <div className="relative group">
                            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 group-focus-within:text-medic-dark" />
                            <input
                                type="text"
                                placeholder="Search by email..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="pl-11 pr-6 py-3 bg-white border border-gray-100 rounded-2xl outline-none focus:border-medic-dark w-64 text-sm font-medium transition-all"
                            />
                        </div>
                    </div>
                </div>

                {/* Tabs */}
                <div className="flex p-1.5 bg-white rounded-3xl border border-gray-100 shadow-sm w-fit">
                    {[
                        { id: 'UPCOMING', label: 'Upcoming Sessions', icon: Clock },
                        { id: 'HISTORY', label: 'Past History', icon: History }
                    ].map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`flex items-center gap-2 px-6 py-3 rounded-2xl text-sm font-black transition-all ${activeTab === tab.id
                                    ? 'bg-medic-dark text-white shadow-lg shadow-medic-dark/20'
                                    : 'text-gray-400 hover:text-gray-600'
                                }`}
                        >
                            <tab.icon size={16} />
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Main Table Card */}
                <div className="bg-white rounded-[2.5rem] shadow-xl shadow-medic-dark/5 border border-medic-light/20 overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full text-left">
                            <thead>
                                <tr className="bg-neutral-soft/50 border-b border-gray-50">
                                    <th className="px-8 py-6 text-gray-500 text-[10px] font-black uppercase tracking-widest">Patient</th>
                                    <th className="px-8 py-6 text-gray-500 text-[10px] font-black uppercase tracking-widest">Date & Time</th>
                                    <th className="px-8 py-6 text-gray-500 text-[10px] font-black uppercase tracking-widest">Status</th>
                                    <th className="px-8 py-6 text-gray-500 text-[10px] font-black uppercase tracking-widest">Consultation Fee</th>
                                    <th className="px-8 py-6 text-gray-500 text-[10px] font-black uppercase tracking-widest text-right">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-50">
                                <AnimatePresence mode='popLayout'>
                                    {filteredAppointments.length > 0 ? filteredAppointments.map((appt, idx) => (
                                        <motion.tr
                                            layout
                                            initial={{ opacity: 0, x: activeTab === 'UPCOMING' ? -20 : 20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            exit={{ opacity: 0, scale: 0.95 }}
                                            transition={{ duration: 0.2 }}
                                            key={appt.id}
                                            className="hover:bg-neutral-soft/20 transition-colors group"
                                        >
                                            <td className="px-8 py-6">
                                                <div className="flex items-center gap-4">
                                                    <div className="w-11 h-11 rounded-2xl bg-medic-light/30 flex items-center justify-center font-black text-medic-dark uppercase shadow-inner">
                                                        {appt.patient_email[0]}
                                                    </div>
                                                    <div className="flex flex-col">
                                                        <span className="font-bold text-gray-900">{appt.patient_email.split('@')[0]}</span>
                                                        <span className="text-xs text-gray-400 font-medium">{appt.patient_email}</span>
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="px-8 py-6">
                                                <div className="flex flex-col gap-1">
                                                    <div className="flex items-center gap-2 text-gray-900 font-bold text-sm">
                                                        <Calendar size={14} className="text-medic-dark" />
                                                        {new Date(appt.appointment_date).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })}
                                                    </div>
                                                    <div className="flex items-center gap-2 text-gray-400 font-bold text-[10px] tracking-wider uppercase">
                                                        <Clock size={12} />
                                                        {appt.start_time.slice(0, 5)} - {appt.end_time.slice(0, 5)}
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="px-8 py-6">
                                                <StatusBadge status={appt.status} />
                                            </td>
                                            <td className="px-8 py-6">
                                                <span className="font-bold text-gray-900">Rs. {appt.fee}</span>
                                            </td>
                                            <td className="px-8 py-6 text-right">
                                                <div className="flex items-center justify-end gap-2">
                                                    <button className="p-2.5 bg-neutral-soft rounded-xl text-gray-400 hover:bg-medic-dark hover:text-white transition-all shadow-sm">
                                                        <Mail size={16} />
                                                    </button>
                                                    <button className="p-2.5 bg-neutral-soft rounded-xl text-gray-400 hover:bg-medic-dark hover:text-white transition-all shadow-sm">
                                                        <MoreVertical size={16} />
                                                    </button>
                                                </div>
                                            </td>
                                        </motion.tr>
                                    )) : (
                                        <motion.tr
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            key="empty"
                                        >
                                            <td colSpan="5" className="px-8 py-20 text-center">
                                                <div className="flex flex-col items-center gap-4 opacity-30">
                                                    {activeTab === 'UPCOMING' ? <Calendar size={48} /> : <History size={48} />}
                                                    <p className="font-black uppercase tracking-widest text-xs">
                                                        {activeTab === 'UPCOMING' ? 'No upcoming sessions scheduled' : 'No past appointment history'}
                                                    </p>
                                                </div>
                                            </td>
                                        </motion.tr>
                                    )}
                                </AnimatePresence>
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Footer Stats */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                    {[
                        { label: 'Total Revenue', value: `Rs. ${appointments.filter(a => a.status === 'PAID' || a.status === 'COMPLETED').reduce((acc, curr) => acc + (curr.fee || 0), 0)}`, color: 'text-green-600' },
                        { label: 'Completed Visits', value: appointments.filter(a => a.status === 'COMPLETED').length, color: 'text-blue-600' },
                        { label: 'Confirmed (Upcoming)', value: appointments.filter(a => a.status === 'PAID' && isUpcoming(a.appointment_date)).length, color: 'text-orange-600' }
                    ].map((stat, i) => (
                        <div key={i} className="bg-white p-6 rounded-[2rem] border border-gray-100 shadow-sm flex flex-col items-center text-center gap-2">
                            <span className="text-[10px] font-black text-gray-400 uppercase tracking-[0.2em]">{stat.label}</span>
                            <span className={`text-2xl font-black ${stat.color}`}>{stat.value}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default DoctorAppointments;
