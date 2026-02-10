import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    ShieldCheck,
    FileText,
    Upload,
    X,
    Plus,
    AlertCircle,
    File,
    Image as ImageIcon,
    Loader2,
    ArrowLeft,
    CheckCircle2
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { submitVerification } from '../../store/slices/authSlice';
import { toast } from 'react-toastify';

const VerifyDoctor = () => {
    const navigate = useNavigate();
    const dispatch = useDispatch();
    const { loading } = useSelector(state => state.auth);

    const [nmcNumber, setNmcNumber] = useState('');
    const [licenseImage, setLicenseImage] = useState(null);
    const [supportingDocs, setSupportingDocs] = useState([]);
    const [errors, setErrors] = useState({});

    const licenseInputRef = useRef(null);
    const supportInputRef = useRef(null);

    const handleNmcChange = (e) => {
        setNmcNumber(e.target.value);
        if (errors.nmcNumber) {
            setErrors(prev => ({ ...prev, nmcNumber: null }));
        }
    };

    const handleLicenseUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            if (file.size > 5 * 1024 * 1024) {
                toast.error("File size should be less than 5MB");
                return;
            }
            setLicenseImage(file);
            if (errors.licenseImage) {
                setErrors(prev => ({ ...prev, licenseImage: null }));
            }
        }
    };

    const handleSupportUpload = (e) => {
        const files = Array.from(e.target.files);
        const validFiles = files.filter(file => {
            if (file.size > 5 * 1024 * 1024) {
                toast.error(`${file.name} is too large (>5MB)`);
                return false;
            }
            return true;
        });
        setSupportingDocs(prev => [...prev, ...validFiles]);
    };

    const removeSupportDoc = (index) => {
        setSupportingDocs(prev => prev.filter((_, i) => i !== index));
    };

    const validateForm = () => {
        const newErrors = {};
        if (!nmcNumber.trim()) newErrors.nmcNumber = "NMC License Number is required";
        if (!licenseImage) newErrors.licenseImage = "NMC License Image is required";

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!validateForm()) return;

        const formData = new FormData();
        formData.append('license_number', nmcNumber);
        formData.append('license_file', licenseImage);

        supportingDocs.forEach((doc) => {
            formData.append('supporting_documents_upload', doc);
        });

        const result = await dispatch(submitVerification(formData));
        if (submitVerification.fulfilled.match(result)) {
            navigate('/doctor-unverified');
        }
    };

    const isSubmitDisabled = !nmcNumber.trim() || !licenseImage || loading;

    const FilePreview = ({ file, onRemove }) => {
        const isImage = file.type.startsWith('image/');
        return (
            <div className="relative group bg-neutral-soft rounded-2xl p-3 flex items-center gap-3 border border-medic-light/20">
                <div className="w-10 h-10 bg-white rounded-xl flex items-center justify-center text-medic-dark shadow-sm">
                    {isImage ? <ImageIcon size={20} /> : <File size={20} />}
                </div>
                <div className="flex-1 min-w-0">
                    <p className="text-sm font-bold text-gray-900 truncate">{file.name}</p>
                    <p className="text-[10px] text-gray-500 uppercase tracking-tight">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
                <button
                    type="button"
                    onClick={onRemove}
                    className="p-1.5 hover:bg-red-50 text-gray-400 hover:text-red-500 rounded-lg transition-colors"
                >
                    <X size={16} />
                </button>
            </div>
        );
    };

    return (
        <div className="min-h-[calc(100vh-80px)] bg-neutral-background py-12 px-6">
            <div className="max-w-3xl mx-auto">
                <button
                    onClick={() => navigate('/doctor-unverified')}
                    className="flex items-center gap-2 text-gray-500 hover:text-medic-dark font-bold text-sm mb-8 transition-colors group"
                >
                    <ArrowLeft size={18} className="group-hover:-translate-x-1 transition-transform" />
                    Back to Profile
                </button>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-white rounded-[2rem] shadow-2xl shadow-medic-dark/5 border border-medic-light/20 overflow-hidden"
                >
                    <div className="bg-medic-dark p-8 md:p-10 text-white flex items-center justify-between">
                        <div className="space-y-1">
                            <h1 className="text-2xl md:text-3xl font-bold">Verification Center</h1>
                            <p className="text-medic-light/70 text-sm md:text-base">Provide your credentials to unlock all platform features.</p>
                        </div>
                        <div className="hidden sm:flex w-16 h-16 bg-white/10 rounded-2xl items-center justify-center backdrop-blur-md">
                            <ShieldCheck size={32} />
                        </div>
                    </div>

                    <form onSubmit={handleSubmit} className="p-8 md:p-10 space-y-10">
                        {/* NMC Number */}
                        <div className="space-y-4">
                            <label className="flex items-center gap-2 text-sm font-extrabold text-gray-900 uppercase tracking-wider">
                                <FileText size={16} className="text-medic-dark" />
                                NMC License Number
                            </label>
                            <div className="relative">
                                <input
                                    type="text"
                                    value={nmcNumber}
                                    onChange={handleNmcChange}
                                    placeholder="Enter your NMC license number (e.g. 123456)"
                                    className={`w-full px-6 py-4 bg-neutral-soft border-2 rounded-2xl outline-none transition-all font-bold text-gray-900 placeholder:text-gray-400 ${errors.nmcNumber ? 'border-red-500' : 'border-transparent focus:border-medic-dark focus:bg-white focus:shadow-xl focus:shadow-medic-dark/5'
                                        }`}
                                />
                                {errors.nmcNumber && (
                                    <p className="text-red-500 text-xs font-bold mt-2 flex items-center gap-1">
                                        <AlertCircle size={12} />
                                        {errors.nmcNumber}
                                    </p>
                                )}
                            </div>
                        </div>

                        {/* License Image Upload */}
                        <div className="space-y-4">
                            <label className="flex items-center gap-2 text-sm font-extrabold text-gray-900 uppercase tracking-wider">
                                <Upload size={16} className="text-medic-dark" />
                                NMC License Document (Image/PDF)
                            </label>

                            {!licenseImage ? (
                                <div
                                    onClick={() => licenseInputRef.current?.click()}
                                    className={`group cursor-pointer border-2 border-dashed rounded-3xl p-10 flex flex-col items-center justify-center gap-4 transition-all hover:bg-medic-light/10 ${errors.licenseImage ? 'border-red-500' : 'border-gray-200 hover:border-medic-dark'
                                        }`}
                                >
                                    <div className="w-16 h-16 bg-medic-light/30 rounded-2xl flex items-center justify-center text-medic-dark group-hover:scale-110 transition-transform">
                                        <Upload size={28} />
                                    </div>
                                    <div className="text-center">
                                        <p className="font-bold text-gray-900">Click to upload license</p>
                                        <p className="text-xs text-gray-500 mt-1">Accepts JPG, PNG, PDF (Max 5MB)</p>
                                    </div>
                                    <input
                                        type="file"
                                        ref={licenseInputRef}
                                        onChange={handleLicenseUpload}
                                        accept=".jpg,.jpeg,.png,.pdf"
                                        className="hidden"
                                    />
                                </div>
                            ) : (
                                <FilePreview file={licenseImage} onRemove={() => setLicenseImage(null)} />
                            )}
                            {errors.licenseImage && (
                                <p className="text-red-500 text-xs font-bold mt-2 flex items-center gap-1">
                                    <AlertCircle size={12} />
                                    {errors.licenseImage}
                                </p>
                            )}
                        </div>

                        {/* Optional Supporting Docs */}
                        <div className="space-y-4 pt-4 border-t border-gray-100">
                            <div className="flex items-center justify-between">
                                <label className="flex items-center gap-2 text-sm font-extrabold text-gray-400 uppercase tracking-wider">
                                    Supporting Documents (Optional)
                                </label>
                                <button
                                    type="button"
                                    onClick={() => supportInputRef.current?.click()}
                                    className="text-xs font-bold text-medic-dark hover:text-medic-primary flex items-center gap-1"
                                >
                                    <Plus size={14} />
                                    Add More
                                </button>
                            </div>

                            <input
                                type="file"
                                ref={supportInputRef}
                                onChange={handleSupportUpload}
                                multiple
                                accept=".jpg,.jpeg,.png,.pdf"
                                className="hidden"
                            />

                            {supportingDocs.length > 0 ? (
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                    {supportingDocs.map((doc, idx) => (
                                        <FilePreview
                                            key={idx}
                                            file={doc}
                                            onRemove={() => removeSupportDoc(idx)}
                                        />
                                    ))}
                                </div>
                            ) : (
                                <p className="text-xs text-gray-400 italic">Degree certificates, ID, or hospital affiliation letters help speed up the process.</p>
                            )}
                        </div>

                        {/* Submit Button */}
                        <div className="pt-6">
                            <button
                                type="submit"
                                disabled={isSubmitDisabled}
                                className="w-full bg-medic-dark text-white py-5 rounded-2xl font-bold text-lg shadow-xl shadow-medic-dark/20 hover:bg-medic-primary transition-all active:scale-[0.98] disabled:opacity-50 disabled:grayscale flex items-center justify-center gap-3"
                            >
                                {loading ? (
                                    <>
                                        <Loader2 className="w-6 h-6 animate-spin" />
                                        <span>Submitting for Review...</span>
                                    </>
                                ) : (
                                    <>
                                        <span>Submit for Verification</span>
                                        <CheckCircle2 size={22} />
                                    </>
                                )}
                            </button>
                            <p className="text-[10px] text-gray-400 text-center mt-4">
                                By submitting, you agree to our terms regarding professional verification and data processing.
                            </p>
                        </div>
                    </form>
                </motion.div>
            </div>
        </div>
    );
};

export default VerifyDoctor;
