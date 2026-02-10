import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Users,
    Search,
    ChevronRight,
    User,
    Clock,
    Activity,
    MessageSquare,
    Loader2,
    Filter,
    ArrowUpRight,
    MapPin,
    Calendar,
    BadgeCheck
} from 'lucide-react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { fetchMyPatients } from '../../store/slices/doctorSlice';

const PatientCard = ({ patient, idx, navigate }) => (
    <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: idx * 0.05 }}
        onClick={() => navigate(`/patient/${patient.id}`)}
        className="bg-white rounded-[2.5rem] p-8 border border-medic-light/20 shadow-xl shadow-medic-dark/5 hover:border-medic-dark transition-all cursor-pointer group relative overflow-hidden"
    >
        <div className="absolute top-0 right-0 p-6 opacity-0 group-hover:opacity-100 transition-all translate-x-4 group-hover:translate-x-0">
            <ArrowUpRight className="text-medic-dark" size={24} />
        </div>

        <div className="space-y-6">
            <div className="flex items-center gap-4">
                <div className="w-16 h-16 rounded-3xl bg-medic-light/30 flex items-center justify-center font-black text-medic-dark text-xl uppercase shadow-inner relative">
                    {patient.first_name[0]}
                    {patient.status === 'ONGOING' && (
                        <div className="absolute -top-1 -right-1 w-4 h-4 bg-blue-500 rounded-full border-2 border-white animate-pulse shadow-sm shadow-blue-200" />
                    )}
                </div>
                <div className="flex flex-col">
                    <h3 className="font-black text-gray-900 text-lg leading-tight">{patient.first_name} {patient.last_name}</h3>
                    <span className="text-sm text-gray-400 font-bold">{patient.email}</span>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-neutral-soft/50 rounded-2xl border border-gray-50 flex flex-col gap-1">
                    <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Medical Goal</span>
                    <span className="text-xs font-bold text-gray-900 truncate">{patient.condition || 'General Checkup'}</span>
                </div>
                <div className="p-4 bg-neutral-soft/50 rounded-2xl border border-gray-50 flex flex-col gap-1">
                    <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Status</span>
                    <span className={`text-xs font-black uppercase ${patient.status === 'ONGOING' ? 'text-blue-600' : 'text-green-600'
                        }`}>
                        {patient.status}
                    </span>
                </div>
            </div>

            <div className="pt-4 border-t border-gray-50 flex items-center justify-between text-gray-400">
                <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest">
                    <Calendar size={14} className="text-medic-primary" />
                    Since {new Date(patient.last_visit).toLocaleDateString(undefined, { month: 'short', year: 'numeric' })}
                </div>
                <div className="flex items-center gap-1 text-[10px] font-black uppercase tracking-widest text-medic-dark group-hover:translate-x-1 transition-transform">
                    View Records
                    <ChevronRight size={12} />
                </div>
            </div>
        </div>
    </motion.div>
);

const DoctorPatients = () => {
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const { patients, loading } = useSelector(state => state.doctor);
    const [searchTerm, setSearchTerm] = useState('');
    const [viewMode, setViewMode] = useState('ALL'); // ALL or ONGOING

    useEffect(() => {
        dispatch(fetchMyPatients());
    }, [dispatch]);

    const filteredPatients = patients.filter(patient => {
        const name = `${patient.first_name} ${patient.last_name}`.toLowerCase();
        const matchesSearch = name.includes(searchTerm.toLowerCase()) ||
            patient.email.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesView = viewMode === 'ALL' || patient.status === 'ONGOING';
        return matchesSearch && matchesView;
    });

    const ongoingPatients = filteredPatients.filter(p => p.status === 'ONGOING');
    const completedPatients = filteredPatients.filter(p => p.status === 'COMPLETED');

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <Loader2 className="w-12 h-12 text-medic-dark animate-spin" />
            </div>
        );
    }

    return (
        <div className="min-h-[calc(100vh-80px)] bg-neutral-background py-8 px-6 space-y-8 transition-all">
            <div className="max-w-7xl mx-auto space-y-8">
                {/* Header */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                    <div className="space-y-1">
                        <h1 className="text-3xl font-black text-gray-900 tracking-tight">Patient Directory</h1>
                        <p className="text-gray-500 font-medium">Manage your clinical relationships and patient records.</p>
                    </div>

                    <div className="flex items-center gap-4">
                        <div className="relative group">
                            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 group-focus-within:text-medic-dark transition-colors" />
                            <input
                                type="text"
                                placeholder="Search patients..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="pl-11 pr-6 py-3 bg-white border border-gray-100 rounded-2xl outline-none focus:border-medic-dark focus:shadow-xl focus:shadow-medic-dark/5 transition-all w-64 text-sm font-medium"
                            />
                        </div>
                        <div className="flex bg-white p-1 rounded-2xl border border-gray-100 shadow-sm">
                            {[
                                { id: 'ALL', label: 'All Patients', count: patients.length },
                                { id: 'ONGOING', label: 'Ongoing Cases', count: patients.filter(p => p.status === 'ONGOING').length }
                            ].map((mode) => (
                                <button
                                    key={mode.id}
                                    onClick={() => setViewMode(mode.id)}
                                    className={`px-6 py-2 rounded-xl text-[10px] font-black tracking-widest uppercase transition-all flex items-center gap-2 ${viewMode === mode.id
                                            ? 'bg-medic-dark text-white shadow-lg shadow-medic-dark/20'
                                            : 'text-gray-400 hover:text-gray-600'
                                        }`}
                                >
                                    {mode.label}
                                    <span className={`px-1.5 py-0.5 rounded-md text-[8px] ${viewMode === mode.id ? 'bg-white/20 text-white' : 'bg-gray-100 text-gray-500'}`}>
                                        {mode.count}
                                    </span>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Content Sections */}
                <div className="space-y-12">
                    {/* Ongoing Section */}
                    {ongoingPatients.length > 0 && (
                        <div className="space-y-6">
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 rounded-xl bg-blue-50 flex items-center justify-center text-blue-600">
                                    <Activity size={18} />
                                </div>
                                <h2 className="text-xl font-black text-gray-900 tracking-tight uppercase tracking-widest text-sm">Active Clinical Cases</h2>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                <AnimatePresence mode='popLayout'>
                                    {ongoingPatients.map((patient, idx) => (
                                        <PatientCard key={patient.id} patient={patient} idx={idx} navigate={navigate} />
                                    ))}
                                </AnimatePresence>
                            </div>
                        </div>
                    )}

                    {/* Completed Section (Only in ALL mode) */}
                    {viewMode === 'ALL' && completedPatients.length > 0 && (
                        <div className="space-y-6">
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 rounded-xl bg-green-50 flex items-center justify-center text-green-600">
                                    <BadgeCheck size={18} />
                                </div>
                                <h2 className="text-xl font-black text-gray-900 tracking-tight uppercase tracking-widest text-sm">Completed Records</h2>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                <AnimatePresence mode='popLayout'>
                                    {completedPatients.map((patient, idx) => (
                                        <PatientCard key={patient.id} patient={patient} idx={idx} navigate={navigate} />
                                    ))}
                                </AnimatePresence>
                            </div>
                        </div>
                    )}

                    {filteredPatients.length === 0 && (
                        <div className="py-32 text-center">
                            <div className="flex flex-col items-center gap-4 opacity-20">
                                <Users size={64} />
                                <p className="font-black uppercase tracking-[0.3em] text-sm">No clinical records matched</p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Summary Banner */}
                <div className="bg-medic-dark rounded-[3rem] p-10 text-white flex flex-col md:flex-row items-center justify-between gap-8 shadow-2xl shadow-medic-dark/30 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
                    <div className="space-y-4 text-center md:text-left relative z-10">
                        <h2 className="text-3xl font-black tracking-tight">Expand Your Clinical Practice</h2>
                        <p className="text-white/60 font-medium max-w-md">Connect with more patients and manage your records with our HIPAA-compliant platform.</p>
                    </div>
                    <div className="flex items-center gap-8 px-10 py-6 bg-white/5 rounded-3xl backdrop-blur-md border border-white/10 relative z-10">
                        <div className="text-center">
                            <div className="text-2xl font-black">{patients.length}</div>
                            <div className="text-[10px] font-black text-white/40 uppercase tracking-widest">Active Patients</div>
                        </div>
                        <div className="w-px h-8 bg-white/10" />
                        <div className="text-center">
                            <div className="text-2xl font-black text-blue-400">{patients.filter(p => p.status === 'ONGOING').length}</div>
                            <div className="text-[10px] font-black text-white/40 uppercase tracking-widest">Ongoing Cases</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DoctorPatients;
