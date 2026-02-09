import React, { useState, useEffect } from 'react';
import { adminApi } from '../../services/api';
import { toast } from 'react-toastify';
import { CheckCircle, XCircle, Clock, FileText, ChevronRight, X, User } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const AdminDoctors = () => {
    const [activeTab, setActiveTab] = useState('pending'); // pending, verified, rejected
    const [doctors, setDoctors] = useState([]);
    const [loading, setLoading] = useState(false);
    const [selectedDoctor, setSelectedDoctor] = useState(null);

    useEffect(() => {
        fetchDoctors();
    }, [activeTab]);

    const fetchDoctors = async () => {
        setLoading(true);
        try {
            const response = await adminApi.getDoctors(activeTab);
            setDoctors(response.data);
        } catch (error) {
            toast.error("Failed to fetch doctors");
        } finally {
            setLoading(false);
        }
    };

    const handleVerify = async (status, rejectionReason = '') => {
        if (!selectedDoctor?.license_info?.id) return;

        try {
            await adminApi.verifyDoctor(selectedDoctor.license_info.id, {
                status,
                rejection_reason: rejectionReason
            });
            toast.success(`Doctor ${status === 'APPROVED' ? 'verified' : 'rejected'} successfully`);
            setSelectedDoctor(null);
            fetchDoctors();
        } catch (error) {
            toast.error("Failed to update status");
        }
    };

    return (
        <div className="space-y-6">
            <h1 className="text-2xl font-bold text-gray-900">Manage Doctors</h1>

            {/* Tabs */}
            <div className="flex gap-2 border-b border-gray-200">
                {['pending', 'verified', 'rejected'].map((tab) => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`px-6 py-3 font-medium text-sm transition-all relative ${activeTab === tab
                                ? 'text-medic-dark'
                                : 'text-gray-500 hover:text-gray-700'
                            }`}
                    >
                        {tab.charAt(0).toUpperCase() + tab.slice(1)}
                        {activeTab === tab && (
                            <motion.div
                                layoutId="activeTab"
                                className="absolute bottom-0 left-0 right-0 h-0.5 bg-medic-dark"
                            />
                        )}
                    </button>
                ))}
            </div>

            {/* List */}
            {loading ? (
                <div className="flex justify-center py-12">
                    <div className="w-8 h-8 border-2 border-medic-dark/30 border-t-medic-dark rounded-full animate-spin" />
                </div>
            ) : doctors.length === 0 ? (
                <div className="text-center py-12 text-gray-500 bg-white rounded-2xl border border-dashed border-gray-200">
                    No doctors found in this category
                </div>
            ) : (
                <div className="grid gap-4">
                    {doctors.map((doctor) => (
                        <div
                            key={doctor.id}
                            onClick={() => setSelectedDoctor(doctor)}
                            className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-all cursor-pointer flex items-center justify-between group"
                        >
                            <div className="flex items-center gap-4">
                                <div className="w-10 h-10 bg-neutral-soft rounded-full flex items-center justify-center text-gray-500 font-bold">
                                    {doctor.first_name[0]}{doctor.last_name[0]}
                                </div>
                                <div>
                                    <h3 className="font-bold text-gray-900">{doctor.first_name} {doctor.last_name}</h3>
                                    <p className="text-xs text-gray-500">{doctor.email} • {doctor.specialization || 'No Specialization'}</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-3">
                                {activeTab === 'pending' && <Clock className="w-5 h-5 text-orange-400" />}
                                {activeTab === 'verified' && <CheckCircle className="w-5 h-5 text-green-500" />}
                                {activeTab === 'rejected' && <XCircle className="w-5 h-5 text-red-500" />}
                                <ChevronRight className="w-5 h-5 text-gray-300 group-hover:text-gray-600 transition-colors" />
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Detail Modal */}
            <AnimatePresence>
                {selectedDoctor && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setSelectedDoctor(null)}
                            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
                        />
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            className="bg-white w-full max-w-2xl rounded-3xl shadow-2xl relative z-10 max-h-[90vh] overflow-auto"
                        >
                            <div className="sticky top-0 bg-white p-6 border-b flex items-center justify-between z-10">
                                <h2 className="text-xl font-bold">Doctor Application</h2>
                                <button onClick={() => setSelectedDoctor(null)} className="p-2 hover:bg-gray-100 rounded-full">
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            <div className="p-6 space-y-8">
                                <div className="flex items-center gap-6">
                                    <div className="w-20 h-20 bg-medic-dark text-white rounded-2xl flex items-center justify-center text-2xl font-bold">
                                        {selectedDoctor.first_name[0]}{selectedDoctor.last_name[0]}
                                    </div>
                                    <div>
                                        <h3 className="text-2xl font-bold">{selectedDoctor.first_name} {selectedDoctor.last_name}</h3>
                                        <p className="text-gray-500">{selectedDoctor.email}</p>
                                        <div className="flex gap-2 mt-2">
                                            <span className="px-3 py-1 bg-neutral-soft rounded-full text-xs font-bold text-gray-600">
                                                {selectedDoctor.specialization || 'N/A'}
                                            </span>
                                            <span className="px-3 py-1 bg-neutral-soft rounded-full text-xs font-bold text-gray-600">
                                                {selectedDoctor.phone_number}
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                <div className="space-y-4">
                                    <h4 className="font-bold text-gray-900 border-b pb-2">Documents</h4>

                                    {selectedDoctor.license_info ? (
                                        <div className="grid sm:grid-cols-2 gap-4">
                                            <div className="p-4 border rounded-xl hover:border-medic-dark transition-colors group">
                                                <div className="flex items-start justify-between mb-2">
                                                    <FileText className="w-6 h-6 text-medic-dark" />
                                                    <span className="text-xs font-bold px-2 py-1 bg-gray-100 rounded">License</span>
                                                </div>
                                                <p className="text-sm font-medium truncate mb-2">{selectedDoctor.license_info.license_number}</p>
                                                {selectedDoctor.license_info.license_file && (
                                                    <a
                                                        href={selectedDoctor.license_info.license_file}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="text-xs font-bold text-medic-accent hover:underline block"
                                                    >
                                                        View Document
                                                    </a>
                                                )}
                                            </div>

                                            {selectedDoctor.license_info.other_certificates && (
                                                <div className="p-4 border rounded-xl hover:border-medic-dark transition-colors group">
                                                    <div className="flex items-start justify-between mb-2">
                                                        <FileText className="w-6 h-6 text-medic-dark" />
                                                        <span className="text-xs font-bold px-2 py-1 bg-gray-100 rounded">Certificate</span>
                                                    </div>
                                                    <p className="text-sm font-medium text-gray-500 mb-2">Additional Docs</p>
                                                    <a
                                                        href={selectedDoctor.license_info.other_certificates}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="text-xs font-bold text-medic-accent hover:underline block"
                                                    >
                                                        View Document
                                                    </a>
                                                </div>
                                            )}
                                        </div>
                                    ) : (
                                        <div className="p-8 text-center bg-neutral-soft rounded-xl text-gray-500">
                                            No license information submitted yet.
                                        </div>
                                    )}
                                </div>

                                {selectedDoctor.license_info?.rejection_reason && (
                                    <div className="p-4 bg-red-50 border border-red-100 rounded-xl text-red-600 text-sm">
                                        <strong>Rejection Reason:</strong> {selectedDoctor.license_info.rejection_reason}
                                    </div>
                                )}
                            </div>

                            {activeTab === 'pending' && selectedDoctor.license_info && (
                                <div className="p-6 border-t bg-gray-50 flex gap-4 sticky bottom-0">
                                    <button
                                        onClick={() => {
                                            const reason = prompt("Enter rejection reason:");
                                            if (reason) handleVerify('REJECTED', reason);
                                        }}
                                        className="flex-1 py-3 px-6 rounded-xl font-bold bg-white border border-red-200 text-red-500 hover:bg-red-50 hover:border-red-300 transition-all"
                                    >
                                        Reject
                                    </button>
                                    <button
                                        onClick={() => handleVerify('APPROVED')}
                                        className="flex-1 py-3 px-6 rounded-xl font-bold bg-medic-dark text-white hover:bg-medic-primary transition-all shadow-lg shadow-medic-dark/20"
                                    >
                                        Verify & Approve
                                    </button>
                                </div>
                            )}
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default AdminDoctors;
