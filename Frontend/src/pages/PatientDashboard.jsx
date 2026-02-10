<<<<<<< Updated upstream
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    FileText,
    Clock,
    CheckCircle2,
    AlertCircle,
    ChevronRight,
    TrendingUp,
    Plus,
    Search,
    MoreVertical,
    Loader2,
    Trash2,
    Activity
} from 'lucide-react';


import { useSelector, useDispatch } from 'react-redux';
import { fetchReports, deleteReport, setCurrentReport } from '../store/slices/reportsSlice';
import { Link, useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';


const PatientDashboard = () => {
    const dispatch = useDispatch();
    const navigate = useNavigate();

    const { user } = useSelector(state => state.auth);
    const { reports, loading } = useSelector(state => state.reports);

    useEffect(() => {
        dispatch(fetchReports());
    }, [dispatch]);

    const stats = [
        { label: 'Total Reports', value: reports.length, icon: FileText, color: 'text-blue-500', bg: 'bg-blue-50' },
        { label: 'In Queue', value: reports.filter(r => r.status === 'PENDING').length, icon: Clock, color: 'text-amber-500', bg: 'bg-amber-50' },
        { label: 'Completed', value: reports.filter(r => r.status === 'PROCESSED').length, icon: CheckCircle2, color: 'text-medic-accent', bg: 'bg-medic-light/50' },
    ];

    const getStatusStyle = (status) => {
        switch (status) {
            case 'PROCESSED': return 'bg-medic-light text-medic-dark border-medic-dark/10';
            case 'PENDING': return 'bg-blue-50 text-blue-600 border-blue-100 animate-pulse';
            case 'FAILED': return 'bg-red-50 text-red-600 border-red-100';
            default: return 'bg-gray-50 text-gray-500';
        }
    };

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleDateString();
    };

    const handleDelete = async (id) => {
        if (window.confirm("Are you sure you want to delete this report? This action cannot be undone.")) {
            try {
                await dispatch(deleteReport(id)).unwrap();
                toast.success("Report deleted successfully");
            } catch (err) {
                toast.error(err?.message || "Failed to delete report");
            }
        }
    };



    const handleExtract = (report) => {
        dispatch(setCurrentReport(report));
        navigate('/check-reports');
    };

    const handleViewResult = (report) => {
        navigate(`/reports/${report.id}/result/`);
    };


    return (


        <div className="max-w-7xl mx-auto px-6 py-10 space-y-10">
            {/* Welcome Section */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-medic-dark rounded-[2rem] p-6 sm:p-8 md:p-12 text-white relative overflow-hidden shadow-xl shadow-medic-dark/20"
            >
                <div className="relative z-10 max-w-2xl">
                    <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-3 sm:mb-4">Welcome back, {user?.first_name || 'Patient'}</h1>
                    <p className="text-medic-light/70 text-base sm:text-lg leading-relaxed">
                        Track and review your medical reports below. Our AI is ready to explain your latest results.
                    </p>
                </div>
                <div className="absolute right-0 bottom-0 top-0 w-1/3 opacity-10 pointer-events-none hidden lg:block">
                    <HeartPulseIcon className="w-full h-full p-12" />
                </div>
            </motion.div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {stats.map((stat, i) => (
                    <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="bg-white p-6 rounded-3xl shadow-sm border border-gray-100 flex items-center gap-6"
                    >
                        <div className={`w-14 h-14 ${stat.bg} ${stat.color} rounded-2xl flex items-center justify-center flex-shrink-0`}>
                            <stat.icon className="w-7 h-7" />
                        </div>
                        <div>
                            <p className="text-sm font-bold text-gray-400 uppercase tracking-wider">{stat.label}</p>
                            <p className="text-3xl font-bold text-gray-900">{stat.value}</p>
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* Reports List */}
            <section className="bg-white rounded-[2rem] shadow-sm border border-gray-100 overflow-hidden">
                <div className="p-8 border-b border-gray-50 flex flex-col sm:flex-row justify-between items-center gap-4">
                    <h2 className="text-2xl font-bold text-gray-900">Recent Reports</h2>
                    <div className="flex items-center gap-4 w-full sm:w-auto">
                        <div className="relative flex-grow sm:flex-grow-0">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                            <input
                                type="text"
                                placeholder="Search reports..."
                                className="pl-10 pr-4 py-2 bg-neutral-soft border-transparent focus:bg-white focus:border-medic-dark/20 rounded-xl text-sm outline-none transition-all w-full"
                            />
                        </div>
                        <Link to="/check-reports" className="bg-medic-dark text-white p-2.5 rounded-xl hover:bg-medic-primary transition-all">
                            <Plus className="w-5 h-5" />
                        </Link>
                    </div>
                </div>

                <div className="overflow-x-auto">
                    {loading ? (
                        <div className="flex flex-col items-center justify-center py-20 gap-4">
                            <Loader2 className="w-10 h-10 text-medic-dark animate-spin" />
                            <p className="text-gray-500 font-medium">Crunching your health data...</p>
                        </div>
                    ) : reports.length > 0 ? (
                        <table className="w-full text-left">
                            <thead>
                                <tr className="bg-neutral-soft/30 text-xs font-bold text-gray-400 uppercase tracking-widest">
                                    <th className="px-8 py-4">Report Name</th>
                                    <th className="px-8 py-4">Upload Date</th>
                                    <th className="px-8 py-4">Status</th>
                                    <th className="px-8 py-4 text-right">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-50">
                                {reports.map((report) => (
                                    <tr key={report.id} className="hover:bg-neutral-soft/20 transition-colors group">
                                        <td className="px-8 py-5">
                                            <div className="flex items-center gap-3">
                                                <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center text-gray-400 group-hover:bg-medic-light transition-colors group-hover:text-medic-dark">
                                                    <FileText className="w-5 h-5" />
                                                </div>
                                                <span className="font-bold text-gray-900">{report.file.split('/').pop()}</span>
                                            </div>
                                        </td>
                                        <td className="px-8 py-5 text-sm text-gray-500">{formatDate(report.uploaded_at)}</td>
                                        <td className="px-8 py-5">
                                            <span className={`px-3 py-1 rounded-full text-xs font-bold border ${getStatusStyle(report.status)}`}>
                                                {report.status}
                                            </span>
                                        </td>
                                        <td className="px-8 py-5 text-right">
                                            <div className="flex items-center justify-end gap-2">
                                                <button
                                                    onClick={() => handleDelete(report.id)}
                                                    className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
                                                    title="Delete Report"
                                                >
                                                    <Trash2 className="w-5 h-5" />
                                                </button>
                                                {report.status === 'PENDING' ? (
                                                    <button
                                                        onClick={() => handleExtract(report)}
                                                        className="px-4 py-2 rounded-lg text-sm font-bold transition-all bg-medic-accent text-medic-dark hover:bg-medic-accent/80 flex items-center gap-2"
                                                    >
                                                        Extract Data <Activity className="w-4 h-4" />
                                                    </button>
                                                ) : (
                                                    <button
                                                        onClick={() => handleViewResult(report)}
                                                        className="px-4 py-2 rounded-lg text-sm font-bold transition-all bg-medic-dark text-white hover:bg-medic-primary disabled:opacity-30 disabled:pointer-events-none flex items-center gap-2"
                                                    >
                                                        View Result <ChevronRight className="w-4 h-4" />
                                                    </button>

                                                )}

                                            </div>
                                        </td>

                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <div className="text-center py-20 px-8">
                            <div className="w-20 h-20 bg-neutral-soft rounded-full flex items-center justify-center mx-auto mb-6">
                                <FileText className="w-10 h-10 text-gray-300" />
                            </div>
                            <h3 className="text-xl font-bold text-gray-900 mb-2">No reports found</h3>
                            <p className="text-gray-500 max-w-xs mx-auto mb-8">
                                You haven't uploaded any medical reports yet. Start by adding your first report.
                            </p>
                            <Link to="/check-reports" className="inline-flex items-center gap-2 px-8 py-3 bg-medic-dark text-white rounded-xl font-bold hover:bg-medic-primary transition-all">
                                <Plus className="w-5 h-5" /> Upload Report
                            </Link>
                        </div>
                    )}
                </div>

            </section>
        </div>
    );
};

const HeartPulseIcon = (props) => (
    <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
        <path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27" />
    </svg>
);

export default PatientDashboard;
=======
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    FileText,
    Clock,
    CheckCircle2,
    AlertCircle,
    ChevronRight,
    TrendingUp,
    Plus,
    Search,
    MoreVertical,
    Loader2,
    Trash2,
    Activity
} from 'lucide-react';


import { useSelector, useDispatch } from 'react-redux';
import { fetchReports, deleteReport, setCurrentReport } from '../store/slices/reportsSlice';
import { fetchAppointments } from '../store/slices/appointmentSlice';
import { Link, useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';


const PatientDashboard = () => {
    const dispatch = useDispatch();
    const navigate = useNavigate();

    const { user } = useSelector(state => state.auth);
    const { reports, loading } = useSelector(state => state.reports);
    const { appointments, loading: appointmentsLoading } = useSelector(state => state.appointment);

    useEffect(() => {
        dispatch(fetchReports());
        dispatch(fetchAppointments());
    }, [dispatch]);

    const stats = [
        { label: 'Total Reports', value: reports.length, icon: FileText, color: 'text-blue-500', bg: 'bg-blue-50' },
        { label: 'In Queue', value: reports.filter(r => r.status === 'PENDING').length, icon: Clock, color: 'text-amber-500', bg: 'bg-amber-50' },
        { label: 'Completed', value: reports.filter(r => r.status === 'PROCESSED').length, icon: CheckCircle2, color: 'text-medic-accent', bg: 'bg-medic-light/50' },
    ];

    const getStatusStyle = (status) => {
        switch (status) {
            case 'PROCESSED': return 'bg-medic-light text-medic-dark border-medic-dark/10';
            case 'PENDING': return 'bg-blue-50 text-blue-600 border-blue-100 animate-pulse';
            case 'FAILED': return 'bg-red-50 text-red-600 border-red-100';
            default: return 'bg-gray-50 text-gray-500';
        }
    };

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleDateString();
    };

    const handleDelete = async (id) => {
        if (window.confirm("Are you sure you want to delete this report? This action cannot be undone.")) {
            try {
                await dispatch(deleteReport(id)).unwrap();
                toast.success("Report deleted successfully");
            } catch (err) {
                toast.error(err?.message || "Failed to delete report");
            }
        }
    };



    const handleExtract = (report) => {
        dispatch(setCurrentReport(report));
        navigate('/check-reports');
    };

    const handleViewResult = (report) => {
        navigate(`/reports/${report.id}/result/`);
    };


    return (


        <div className="max-w-7xl mx-auto px-6 py-10 space-y-10">
            {/* Welcome Section */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-medic-dark rounded-[2rem] p-6 sm:p-8 md:p-12 text-white relative overflow-hidden shadow-xl shadow-medic-dark/20"
            >
                <div className="relative z-10 max-w-2xl">
                    <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-3 sm:mb-4">Welcome back, {user?.first_name || 'Patient'}</h1>
                    <p className="text-medic-light/70 text-base sm:text-lg leading-relaxed">
                        Track and review your medical reports below. Our AI is ready to explain your latest results.
                    </p>
                </div>
                <div className="absolute right-0 bottom-0 top-0 w-1/3 opacity-10 pointer-events-none hidden lg:block">
                    <HeartPulseIcon className="w-full h-full p-12" />
                </div>
            </motion.div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {stats.map((stat, i) => (
                    <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="bg-white p-6 rounded-3xl shadow-sm border border-gray-100 flex items-center gap-6"
                    >
                        <div className={`w-14 h-14 ${stat.bg} ${stat.color} rounded-2xl flex items-center justify-center flex-shrink-0`}>
                            <stat.icon className="w-7 h-7" />
                        </div>
                        <div>
                            <p className="text-sm font-bold text-gray-400 uppercase tracking-wider">{stat.label}</p>
                            <p className="text-3xl font-bold text-gray-900">{stat.value}</p>
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* Appointments Section */}
            <section className="grid grid-cols-1 lg:grid-cols-3 gap-10">
                <div className="lg:col-span-2 space-y-6">
                    <div className="flex items-center justify-between">
                        <h2 className="text-2xl font-bold text-gray-900">Upcoming Appointments</h2>
                        <Link to="/doctors" className="text-medic-dark font-bold text-sm hover:underline">Find a Doctor</Link>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {appointmentsLoading ? (
                            <div className="col-span-2 py-10 flex justify-center"><Loader2 className="animate-spin text-medic-dark" /></div>
                        ) : appointments.filter(a => a.status === 'PAID').length > 0 ? (
                            appointments.filter(a => a.status === 'PAID').slice(0, 2).map((appt) => (
                                <div key={appt.id} className="bg-white p-6 rounded-3xl border border-gray-100 shadow-sm flex items-start gap-4">
                                    <div className="w-12 h-12 bg-medic-light/30 rounded-2xl flex items-center justify-center text-medic-dark">
                                        <Calendar className="w-6 h-6" />
                                    </div>
                                    <div className="space-y-1">
                                        <h4 className="font-bold text-gray-900">Dr. {appt.doctor_full_name}</h4>
                                        <div className="flex items-center gap-2 text-xs text-gray-400 font-bold">
                                            <Clock className="w-3.5 h-3.5" /> {appt.appointment_date} @ {appt.start_time.slice(0, 5)}
                                        </div>
                                        <span className="inline-block px-2 py-0.5 bg-green-50 text-green-600 text-[10px] font-bold rounded-full">Confirmed</span>
                                    </div>
                                </div>
                            ))
                        ) : (
                            <div className="col-span-2 p-10 bg-neutral-soft/30 rounded-3xl border border-dashed border-gray-200 text-center">
                                <p className="text-gray-400 font-medium italic">No upcoming appointments.</p>
                            </div>
                        )}
                    </div>
                </div>

            </section>

            {/* Reports List */}
            <section className="bg-white rounded-[2rem] shadow-sm border border-gray-100 overflow-hidden">
                <div className="p-8 border-b border-gray-50 flex flex-col sm:flex-row justify-between items-center gap-4">
                    <h2 className="text-2xl font-bold text-gray-900">Recent Reports</h2>
                    <div className="flex items-center gap-4 w-full sm:w-auto">
                        <div className="relative flex-grow sm:flex-grow-0">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                            <input
                                type="text"
                                placeholder="Search reports..."
                                className="pl-10 pr-4 py-2 bg-neutral-soft border-transparent focus:bg-white focus:border-medic-dark/20 rounded-xl text-sm outline-none transition-all w-full"
                            />
                        </div>
                        <Link to="/check-reports" className="bg-medic-dark text-white p-2.5 rounded-xl hover:bg-medic-primary transition-all">
                            <Plus className="w-5 h-5" />
                        </Link>
                    </div>
                </div>

                <div className="overflow-x-auto">
                    {loading ? (
                        <div className="flex flex-col items-center justify-center py-20 gap-4">
                            <Loader2 className="w-10 h-10 text-medic-dark animate-spin" />
                            <p className="text-gray-500 font-medium">Crunching your health data...</p>
                        </div>
                    ) : reports.length > 0 ? (
                        <table className="w-full text-left">
                            <thead>
                                <tr className="bg-neutral-soft/30 text-xs font-bold text-gray-400 uppercase tracking-widest">
                                    <th className="px-8 py-4">Report Name</th>
                                    <th className="px-8 py-4">Upload Date</th>
                                    <th className="px-8 py-4">Status</th>
                                    <th className="px-8 py-4 text-right">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-50">
                                {reports.map((report) => (
                                    <tr key={report.id} className="hover:bg-neutral-soft/20 transition-colors group">
                                        <td className="px-8 py-5">
                                            <div className="flex items-center gap-3">
                                                <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center text-gray-400 group-hover:bg-medic-light transition-colors group-hover:text-medic-dark">
                                                    <FileText className="w-5 h-5" />
                                                </div>
                                                <span className="font-bold text-gray-900">{report.file?.split('/').pop() || 'Medical Report'}</span>
                                            </div>
                                        </td>
                                        <td className="px-8 py-5 text-sm text-gray-500">{formatDate(report.uploaded_at)}</td>
                                        <td className="px-8 py-5">
                                            <span className={`px-3 py-1 rounded-full text-xs font-bold border ${getStatusStyle(report.status)}`}>
                                                {report.status}
                                            </span>
                                        </td>
                                        <td className="px-8 py-5 text-right">
                                            <div className="flex items-center justify-end gap-2">
                                                <button
                                                    onClick={() => handleDelete(report.id)}
                                                    className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
                                                    title="Delete Report"
                                                >
                                                    <Trash2 className="w-5 h-5" />
                                                </button>
                                                {report.status === 'PENDING' ? (
                                                    <button
                                                        onClick={() => handleExtract(report)}
                                                        className="px-4 py-2 rounded-lg text-sm font-bold transition-all bg-medic-accent text-medic-dark hover:bg-medic-accent/80 flex items-center gap-2"
                                                    >
                                                        Extract Data <Activity className="w-4 h-4" />
                                                    </button>
                                                ) : (
                                                    <button
                                                        onClick={() => handleViewResult(report)}
                                                        className="px-4 py-2 rounded-lg text-sm font-bold transition-all bg-medic-dark text-white hover:bg-medic-primary disabled:opacity-30 disabled:pointer-events-none flex items-center gap-2"
                                                    >
                                                        View Result <ChevronRight className="w-4 h-4" />
                                                    </button>

                                                )}

                                            </div>
                                        </td>

                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <div className="text-center py-20 px-8">
                            <div className="w-20 h-20 bg-neutral-soft rounded-full flex items-center justify-center mx-auto mb-6">
                                <FileText className="w-10 h-10 text-gray-300" />
                            </div>
                            <h3 className="text-xl font-bold text-gray-900 mb-2">No reports found</h3>
                            <p className="text-gray-500 max-w-xs mx-auto mb-8">
                                You haven't uploaded any medical reports yet. Start by adding your first report.
                            </p>
                            <Link to="/check-reports" className="inline-flex items-center gap-2 px-8 py-3 bg-medic-dark text-white rounded-xl font-bold hover:bg-medic-primary transition-all">
                                <Plus className="w-5 h-5" /> Upload Report
                            </Link>
                        </div>
                    )}
                </div>

            </section>
        </div>
    );
};

const HeartPulseIcon = (props) => (
    <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
        <path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27" />
    </svg>
);

export default PatientDashboard;
>>>>>>> Stashed changes
