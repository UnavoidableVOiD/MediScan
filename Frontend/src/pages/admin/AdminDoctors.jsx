import React, { useState, useEffect } from 'react';
import { CheckCircle, XCircle, Clock, FileText, ChevronRight, X, User, ShieldAlert, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSelector, useDispatch } from 'react-redux';
import { fetchAdminDoctors, verifyDoctor, updateDoctor, deleteDoctor, unverifyDoctor } from '../../store/slices/adminSlice';

const AdminDoctors = () => {
    const [activeTab, setActiveTab] = useState('pending'); // pending, verified, rejected
    const [selectedDoctor, setSelectedDoctor] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const dispatch = useDispatch();
    const { user } = useSelector(state => state.auth);
    const { doctors, loading } = useSelector(state => state.admin);

    useEffect(() => {
        dispatch(fetchAdminDoctors(activeTab));
    }, [activeTab, dispatch]);

    const handleCloseModal = () => {
        setSelectedDoctor(null);
        setPreviewUrl(null);
    };

    const renderPreview = (url) => {
        if (!url) return null;

        const isImage = /\.(jpg|jpeg|png|webp|gif)$/i.test(url);
        const isPdf = /\.pdf$/i.test(url);

        return (
            <div className="mt-4 p-4 bg-gray-100 rounded-2xl border-2 border-dashed border-gray-300 relative">
                <button
                    onClick={() => setPreviewUrl(null)}
                    className="absolute top-2 right-2 p-1 bg-white rounded-full shadow-md hover:bg-gray-50 z-10"
                >
                    <X className="w-4 h-4" />
                </button>
                <div className="flex justify-center items-center min-h-[300px]">
                    {isImage ? (
                        <img src={url} alt="Document Preview" className="max-w-full max-h-[500px] object-contain rounded-lg shadow-lg" />
                    ) : isPdf ? (
                        <iframe src={url} className="w-full h-[500px] rounded-lg border shadow-lg" title="PDF Preview" />
                    ) : (
                        <div className="text-center p-8">
                            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                            <p className="text-gray-600 mb-4">Preview not available for this file type.</p>
                            <a
                                href={url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-2 px-4 py-2 bg-medic-dark text-white rounded-lg hover:bg-medic-primary transition-colors"
                            >
                                Open in New Tab
                            </a>
                        </div>
                    )}
                </div>
            </div>
        );
    };

    const handleVerify = async (status, rejectionReason = '') => {
        if (!selectedDoctor?.license_info?.id) return;

        try {
            await dispatch(verifyDoctor({
                id: selectedDoctor.license_info.id,
                data: { status, rejection_reason: rejectionReason }
            })).unwrap();
            handleCloseModal();
            dispatch(fetchAdminDoctors(activeTab));
        } catch (error) {
            // toast handled by slice
        }
    };

    const handleUpdateDoctor = async (id, data) => {
        try {
            await dispatch(updateDoctor({ id, data })).unwrap();
            dispatch(fetchAdminDoctors(activeTab));
            if (selectedDoctor && selectedDoctor.id === id) {
                setSelectedDoctor({ ...selectedDoctor, ...data });
            }
        } catch (error) {
            // toast handled by slice
        }
    };

    const handleDeleteDoctor = async (id) => {
        if (!window.confirm("Are you sure you want to delete this doctor? This action cannot be undone.")) return;
        try {
            await dispatch(deleteDoctor(id)).unwrap();
            handleCloseModal();
        } catch (error) {
            // toast handled by slice
        }
    };

    const handleUnverify = async (id) => {
        if (!window.confirm("Are you sure you want to unverify this doctor? This will remove their current license data.")) return;
        try {
            await dispatch(unverifyDoctor(id)).unwrap();
            handleCloseModal();
            dispatch(fetchAdminDoctors(activeTab));
        } catch (error) {
            // toast handled by slice
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
                                    {(doctor.first_name?.[0] || '') + (doctor.last_name?.[0] || '')}
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
                            onClick={handleCloseModal}
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
                                <button onClick={handleCloseModal} className="p-2 hover:bg-gray-100 rounded-full">
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            <div className="p-6 space-y-8">
                                <div className="flex items-center gap-6">
                                    <div className="w-20 h-20 bg-medic-dark text-white rounded-2xl flex items-center justify-center text-2xl font-bold">
                                        {(selectedDoctor.first_name?.[0] || '') + (selectedDoctor.last_name?.[0] || '')}
                                    </div>
                                    <div>
                                        <h3 className="text-2xl font-bold">{selectedDoctor.first_name} {selectedDoctor.last_name}</h3>
                                        <p className="text-gray-500">{selectedDoctor.email}</p>
                                        <div className="flex flex-wrap gap-2 mt-2">
                                            <span className="px-3 py-1 bg-neutral-soft rounded-full text-xs font-bold text-gray-600">
                                                {selectedDoctor.specialization?.replace(/_/g, ' ') || 'N/A'}
                                            </span>
                                            <span className="px-3 py-1 bg-neutral-soft rounded-full text-xs font-bold text-gray-600">
                                                {selectedDoctor.phone_number}
                                            </span>
                                            {!selectedDoctor.is_active && (
                                                <span className="px-3 py-1 bg-red-100 rounded-full text-xs font-bold text-red-600 flex items-center gap-1">
                                                    <ShieldAlert className="w-3 h-3" />
                                                    Blocked
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                <div className="space-y-4">
                                    <h4 className="font-bold text-gray-900 border-b pb-2">Documents</h4>

                                    {selectedDoctor.license_info ? (
                                        <>
                                            <div className="grid sm:grid-cols-2 gap-4">
                                                {/* License File */}
                                                <div
                                                    onClick={() => setPreviewUrl(selectedDoctor.license_info.license_file)}
                                                    className={`p-4 border rounded-xl transition-all group cursor-pointer ${previewUrl === selectedDoctor.license_info.license_file ? 'border-medic-dark bg-medic-dark/5' : 'hover:border-medic-dark'}`}
                                                >
                                                    <div className="flex items-start justify-between mb-2">
                                                        <FileText className="w-6 h-6 text-medic-dark" />
                                                        <span className="text-xs font-bold px-2 py-1 bg-gray-100 rounded">License</span>
                                                    </div>
                                                    <p className="text-sm font-medium truncate mb-2">{selectedDoctor.license_info.license_number}</p>
                                                    <div className="flex justify-between items-center">
                                                        <span className="text-xs font-bold text-medic-accent hover:underline">Preview</span>
                                                        <a
                                                            href={selectedDoctor.license_info.license_file}
                                                            target="_blank"
                                                            rel="noopener noreferrer"
                                                            onClick={(e) => e.stopPropagation()}
                                                            className="text-[10px] text-gray-400 hover:text-gray-600"
                                                        >
                                                            Open Link
                                                        </a>
                                                    </div>
                                                </div>

                                                {/* Supporting Documents */}
                                                {selectedDoctor.license_info.supporting_documents?.map((doc, index) => (
                                                    <div
                                                        key={index}
                                                        onClick={() => setPreviewUrl(doc)}
                                                        className={`p-4 border rounded-xl transition-all group cursor-pointer ${previewUrl === doc ? 'border-medic-dark bg-medic-dark/5' : 'hover:border-medic-dark'}`}
                                                    >
                                                        <div className="flex items-start justify-between mb-2">
                                                            <FileText className="w-6 h-6 text-indigo-500" />
                                                            <span className="text-xs font-bold px-2 py-1 bg-gray-100 rounded">Certificate</span>
                                                        </div>
                                                        <p className="text-sm font-medium text-gray-500 mb-2">Supporting Doc {index + 1}</p>
                                                        <div className="flex justify-between items-center">
                                                            <span className="text-xs font-bold text-medic-accent hover:underline">Preview</span>
                                                            <a
                                                                href={doc}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                onClick={(e) => e.stopPropagation()}
                                                                className="text-[10px] text-gray-400 hover:text-gray-600"
                                                            >
                                                                Open Link
                                                            </a>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>

                                            {/* Preview Pane */}
                                            {renderPreview(previewUrl)}
                                        </>
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

                            <div className="p-6 border-t bg-gray-50 flex flex-wrap gap-4 sticky bottom-0">
                                {/* Verification Buttons - Only for Pending */}
                                {activeTab === 'pending' && selectedDoctor.license_info && (
                                    <>
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
                                        <div className="w-full h-px bg-gray-200 my-2" />
                                    </>
                                )}

                                {activeTab === 'verified' && (
                                    <button
                                        onClick={() => handleUnverify(selectedDoctor.id)}
                                        className="w-full py-3 px-6 rounded-xl font-bold bg-amber-50 border border-amber-200 text-amber-600 hover:bg-amber-100 transition-all mb-4"
                                    >
                                        Unverify Profile
                                    </button>
                                )}

                                {/* Account Management Buttons - Admins only */}
                                {(user?.is_staff || user?.is_superuser) && (
                                    <div className="flex flex-wrap gap-4 w-full">
                                        <button
                                            onClick={() => handleUpdateDoctor(selectedDoctor.id, { is_active: !selectedDoctor.is_active })}
                                            className={`flex-1 min-w-[200px] py-3 px-6 rounded-xl font-bold flex items-center justify-center gap-2 border transition-all ${selectedDoctor.is_active
                                                ? 'bg-white border-orange-200 text-orange-500 hover:bg-orange-50'
                                                : 'bg-orange-500 border-orange-500 text-white hover:bg-orange-600'
                                                }`}
                                        >
                                            <ShieldAlert className="w-4 h-4" />
                                            {selectedDoctor.is_active ? 'Block Account' : 'Unblock Account'}
                                        </button>
                                        <button
                                            onClick={() => handleDeleteDoctor(selectedDoctor.id)}
                                            className="flex-1 min-w-[200px] py-3 px-6 rounded-xl font-bold bg-white border border-red-200 text-red-500 hover:bg-red-50 hover:border-red-300 transition-all flex items-center justify-center gap-2"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                            Delete Account
                                        </button>
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default AdminDoctors;
