import React, { useEffect, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { fetchReports, uploadReport } from '../store/slices/reportSlice';
import {
    FileText,
    Clock,
    CheckCircle2,
    AlertCircle,
    ChevronRight,
    Plus,
    Search,
    Filter,
    ArrowUpRight,
    Loader2
} from 'lucide-react';
import { toast } from 'react-toastify';
import { parseError } from '../utils/errorParser';

const Dashboard = () => {
    const dispatch = useDispatch();
    const fileInputRef = useRef(null);
    const [searchTerm, setSearchTerm] = React.useState('');
    const { user } = useSelector((state) => state.auth);
    const { reports, loading, stats, uploading } = useSelector((state) => state.reports);

    // Filter reports based on search term
    const filteredReports = reports.filter(report => {
        const fileName = report.file.split('/').pop().toLowerCase();
        return fileName.includes(searchTerm.toLowerCase());
    });

    // Initial fetch on mount
    useEffect(() => {
        dispatch(fetchReports());
    }, [dispatch]);

    // Polling logic
    useEffect(() => {
        const hasPending = reports.some(r => r.status === 'PENDING');
        if (!hasPending) return;

        const interval = setInterval(() => {
            dispatch(fetchReports());
        }, 15000); // Polling every 15 seconds for pending reports

        return () => clearInterval(interval);
    }, [dispatch, reports]);

    const handleUploadClick = () => {
        fileInputRef.current.click();
    };

    const handleFileChange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Basic validation
        if (file.type !== 'application/pdf') {
            toast.error("Please upload a PDF file.");
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        const resultAction = await dispatch(uploadReport(formData));
        if (uploadReport.fulfilled.match(resultAction)) {
            toast.success("Report uploaded successfully! Processing started.");
        } else {
            const errorMsg = parseError(resultAction.payload);
            toast.error(errorMsg);
        }

        // Clear input
        e.target.value = '';
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'PROCESSED': return 'bg-emerald-100 text-emerald-700 border-emerald-200';
            case 'PENDING': return 'bg-amber-100 text-amber-700 border-amber-200';
            case 'FAILED': return 'bg-red-100 text-red-700 border-red-200';
            default: return 'bg-gray-100 text-gray-700 border-gray-200';
        }
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case 'PROCESSED': return <CheckCircle2 className="h-4 w-4" />;
            case 'PENDING': return <Clock className="h-4 w-4" />;
            case 'FAILED': return <AlertCircle className="h-4 w-4" />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 pt-24 pb-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-7xl mx-auto">
                {/* Header section */}
                <div className="flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4">
                    <div>
                        <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                            Welcome back, <span className="text-blue-600 capitalize">{user?.first_name}</span>! 👋
                        </h1>
                        <p className="text-gray-500 mt-1 font-medium">Here's what's happening with your medical reports today.</p>
                    </div>

                    <div className="flex items-center gap-3">
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            className="hidden"
                            accept=".pdf"
                        />
                        <button
                            onClick={handleUploadClick}
                            disabled={uploading}
                            className="flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 to-emerald-500 text-white px-6 py-3 rounded-2xl font-bold shadow-lg shadow-blue-200 hover:shadow-xl hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:translate-y-0"
                        >
                            {uploading ? (
                                <Loader2 className="h-5 w-5 animate-spin" />
                            ) : (
                                <Plus className="h-5 w-5" />
                            )}
                            {uploading ? "Uploading..." : "Upload New Report"}
                        </button>
                    </div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
                    <StatCard
                        title="Total Reports"
                        value={stats.total}
                        icon={<FileText className="h-6 w-6 text-blue-600" />}
                        bgColor="bg-blue-50"
                        borderColor="border-blue-100"
                    />
                    <StatCard
                        title="In Queue"
                        value={stats.pending}
                        icon={<Clock className="h-6 w-6 text-amber-500" />}
                        bgColor="bg-amber-50"
                        borderColor="border-amber-100"
                    />
                    <StatCard
                        title="Reports Completed"
                        value={stats.completed}
                        icon={<CheckCircle2 className="h-6 w-6 text-emerald-500" />}
                        bgColor="bg-emerald-50"
                        borderColor="border-emerald-100"
                    />
                    <StatCard
                        title="Processing Failed"
                        value={stats.failed}
                        icon={<AlertCircle className="h-6 w-6 text-red-500" />}
                        bgColor="bg-red-50"
                        borderColor="border-red-100"
                    />
                </div>

                {/* Reports List Section */}
                <div className="bg-white rounded-3xl shadow-xl shadow-gray-200/50 border border-gray-100 overflow-hidden">
                    <div className="p-6 border-b border-gray-100 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                        <h2 className="text-xl font-bold text-gray-900">Recent Reports</h2>
                        <div className="flex items-center gap-3">
                            <div className="relative">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                                <input
                                    type="text"
                                    placeholder="Search reports..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    className="pl-10 pr-4 py-2 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all w-full sm:w-64"
                                />
                            </div>
                            <button className="p-2 border border-gray-200 rounded-xl hover:bg-gray-50 transition-colors">
                                <Filter className="h-5 w-5 text-gray-500" />
                            </button>
                        </div>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full text-left">
                            <thead className="bg-gray-50 text-gray-500 text-xs uppercase tracking-wider font-bold">
                                <tr>
                                    <th className="px-6 py-4">Report Name</th>
                                    <th className="px-6 py-4">Upload Date</th>
                                    <th className="px-6 py-4">Status</th>
                                    <th className="px-6 py-4 text-right">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-100">
                                {loading && reports.length === 0 ? (
                                    <tr>
                                        <td colSpan="4" className="px-6 py-12 text-center">
                                            <div className="flex flex-col items-center gap-3">
                                                <Loader2 className="h-8 w-8 text-blue-600 animate-spin" />
                                                <p className="text-gray-500 font-medium">Loading your reports...</p>
                                            </div>
                                        </td>
                                    </tr>
                                ) : filteredReports.length === 0 ? (
                                    <tr>
                                        <td colSpan="4" className="px-6 py-12 text-center">
                                            <div className="flex flex-col items-center gap-3 text-gray-400">
                                                <FileText className="h-12 w-12 opacity-20" />
                                                <p className="text-gray-500 font-medium">{searchTerm ? "No reports match your search." : "No reports found. Start by uploading one!"}</p>
                                            </div>
                                        </td>
                                    </tr>
                                ) : (
                                    filteredReports.map((report) => (
                                        <tr key={report.id} className="hover:bg-gray-50 transition-colors group">
                                            <td className="px-6 py-4">
                                                <div className="flex items-center gap-3">
                                                    <div className={`p-2 rounded-lg ${getStatusColor(report.status)} opacity-80 group-hover:opacity-100 transition-opacity`}>
                                                        <FileText className="h-5 w-5" />
                                                    </div>
                                                    <span className="font-bold text-gray-900">{report.file.split('/').pop() || 'Medical Report'}</span>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className="text-gray-600 text-sm font-medium">
                                                    {new Date(report.uploaded_at).toLocaleDateString(undefined, {
                                                        year: 'numeric',
                                                        month: 'short',
                                                        day: 'numeric'
                                                    })}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border ${getStatusColor(report.status)}`}>
                                                    {getStatusIcon(report.status)}
                                                    {report.status === 'PROCESSED' ? 'Completed' : report.status.charAt(0) + report.status.slice(1).toLowerCase()}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 text-right">
                                                {report.status === 'PROCESSED' ? (
                                                    <button className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-700 font-bold text-sm transition-colors">
                                                        View Result
                                                        <ChevronRight className="h-4 w-4" />
                                                    </button>
                                                ) : (
                                                    <span className="text-gray-400 text-sm font-medium italic">Processing...</span>
                                                )}
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
};

const StatCard = ({ title, value, icon, bgColor, borderColor }) => (
    <div className={`p-6 bg-white rounded-3xl border ${borderColor} shadow-lg shadow-gray-100/50 hover:shadow-xl hover:-translate-y-1 transition-all group`}>
        <div className="flex items-center justify-between mb-4">
            <div className={`p-3 ${bgColor} rounded-2xl group-hover:scale-110 transition-transform`}>
                {icon}
            </div>
            <ArrowUpRight className="h-5 w-5 text-gray-300 group-hover:text-gray-500 transition-colors" />
        </div>
        <p className="text-sm font-bold text-gray-500 mb-1">{title}</p>
        <p className="text-3xl font-black text-gray-900 tracking-tight">{value}</p>
    </div>
);

export default Dashboard;
