import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User, Mail, Phone, Shield, Edit3, Save, X, Camera, Lock } from 'lucide-react';
import { useSelector, useDispatch } from 'react-redux';
import { toast } from 'react-toastify';
import { updateProfile } from '../store/slices/authSlice';

const Field = ({ label, name, value, icon: Icon, disabled, isEditing, type = "text", onChange }) => (
    <div className="space-y-1.5 px-4 py-3 rounded-2xl transition-all duration-300">
        <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest block ml-1">{label}</label>
        <div className="flex items-center gap-4 group">
            <div className={`p-2 rounded-xl transition-all duration-300 ${isEditing && !disabled ? 'bg-medic-dark/10 text-medic-dark' : 'bg-gray-50 text-gray-400 group-hover:text-gray-600'}`}>
                <Icon className="w-5 h-5" />
            </div>
            {isEditing && !disabled ? (
                <input
                    type={type}
                    name={name}
                    value={value}
                    onChange={onChange}
                    className="flex-grow bg-white border-b-2 border-medic-light focus:border-medic-dark py-1 outline-none font-medium text-gray-900 transition-all placeholder:text-gray-300"
                    placeholder={`Enter ${label.toLowerCase()}`}
                />
            ) : (
                <span className={`flex-grow font-medium py-1 ${disabled ? 'text-gray-400' : 'text-gray-900'}`}>
                    {value || 'Not provided'}
                </span>
            )}
            {disabled && isEditing && (
                <Lock className="w-4 h-4 text-gray-300" title="This field cannot be edited" />
            )}
        </div>
    </div>
);

const PatientProfile = () => {
    const { user, loading } = useSelector(state => state.auth);
    const dispatch = useDispatch();
    const [isEditing, setIsEditing] = useState(false);

    const [formData, setFormData] = useState({
        firstName: user?.first_name || '',
        lastName: user?.last_name || '',
        phone: user?.phone_number || '',
        email: user?.email || '',
        role: user?.role || 'patient'
    });

    const [backupData, setBackupData] = useState(null);

    React.useEffect(() => {
        if (user && !isEditing) {
            setFormData({
                firstName: user.first_name || '',
                lastName: user.last_name || '',
                phone: user.phone_number || '',
                email: user.email || '',
                role: user.role || 'patient'
            });
        }
    }, [user, isEditing]);

    const handleEditToggle = () => {
        if (!isEditing) {
            setBackupData({ ...formData });
        }
        setIsEditing(!isEditing);
    };

    const handleCancel = () => {
        setFormData(backupData);
        setIsEditing(false);
    };

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSave = (e) => {
        e.preventDefault();

        const profileData = {
            first_name: formData.firstName,
            last_name: formData.lastName,
            phone_number: formData.phone
        };

        dispatch(updateProfile(profileData))
            .unwrap()
            .then(() => setIsEditing(false));
    };

    return (
        <div className="max-w-4xl mx-auto px-6 py-12">
            <header className="mb-10 flex flex-col sm:flex-row sm:items-end justify-between gap-6">
                <div>
                    <h1 className="text-4xl font-bold text-medic-dark mb-2 tracking-tight">Your Profile</h1>
                    <p className="text-gray-500 font-medium">Manage your personal information and account settings.</p>
                </div>

                <div className="flex items-center gap-3">
                    <AnimatePresence mode="wait">
                        {!isEditing ? (
                            <motion.button
                                key="edit-btn"
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.9 }}
                                onClick={handleEditToggle}
                                className="flex items-center gap-2 px-6 py-3 border-2 border-medic-dark text-medic-dark rounded-2xl font-bold hover:bg-medic-light/10 transition-all active:scale-95"
                            >
                                <Edit3 className="w-5 h-5" />
                                Edit Profile
                            </motion.button>
                        ) : (
                            <motion.div
                                key="save-btns"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 20 }}
                                className="flex items-center gap-3"
                            >
                                <button
                                    onClick={handleCancel}
                                    className="px-6 py-3 text-gray-500 font-bold hover:bg-gray-100 rounded-2xl transition-all"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handleSave}
                                    disabled={loading}
                                    className="flex items-center gap-2 px-8 py-3 bg-medic-dark text-white rounded-2xl font-bold shadow-lg shadow-medic-dark/20 hover:bg-medic-primary transition-all active:scale-95 disabled:opacity-50"
                                >
                                    {loading ? (
                                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    ) : (
                                        <>
                                            <Save className="w-5 h-5" />
                                            Save Changes
                                        </>
                                    )}
                                </button>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </header>

            <motion.div
                layout
                className="bg-white rounded-3xl shadow-xl shadow-medic-dark/5 overflow-hidden border border-medic-light/20"
            >
                {/* Profile Header Background */}
                <div className="h-32 bg-medic-dark/5 border-b border-medic-light/10 relative">
                    <div className="absolute -bottom-12 left-10">
                        <div className="relative group">
                            <div className="w-24 h-24 rounded-3xl bg-white p-1.5 shadow-lg">
                                <div className="w-full h-full rounded-2xl bg-medic-dark flex items-center justify-center text-white text-3xl font-bold uppercase ring-4 ring-white">
                                    {user?.first_name?.charAt(0) || user?.email?.charAt(0) || 'U'}
                                </div>
                            </div>
                            <button className="absolute -bottom-1 -right-1 w-8 h-8 bg-medic-accent text-medic-dark rounded-xl flex items-center justify-center shadow-lg border-2 border-white hover:scale-110 transition-transform">
                                <Camera className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </div>

                <div className="pt-20 pb-10 px-10">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="space-y-6">
                            <h3 className="text-xl font-bold text-gray-900 border-l-4 border-medic-dark pl-4 mb-6">Personal Details</h3>
                            <div className="grid grid-cols-1 gap-4">
                                <Field
                                    label="First Name"
                                    name="firstName"
                                    value={formData.firstName}
                                    icon={User}
                                    isEditing={isEditing}
                                    onChange={handleChange}
                                />
                                <Field
                                    label="Last Name"
                                    name="lastName"
                                    value={formData.lastName}
                                    icon={User}
                                    isEditing={isEditing}
                                    onChange={handleChange}
                                />
                                <Field
                                    label="Phone Number"
                                    name="phone"
                                    value={formData.phone}
                                    icon={Phone}
                                    type="tel"
                                    isEditing={isEditing}
                                    onChange={handleChange}
                                />
                            </div>
                        </div>

                        <div className="space-y-6 md:border-l md:border-gray-100 md:pl-10">
                            <h3 className="text-xl font-bold text-gray-900 border-l-4 border-medic-light pl-4 mb-6">Account Verification</h3>
                            <div className="grid grid-cols-1 gap-4">
                                <Field
                                    label="Email Address"
                                    name="email"
                                    value={formData.email}
                                    icon={Mail}
                                    disabled
                                    isEditing={isEditing}
                                    onChange={handleChange}
                                />
                                <Field
                                    label="User Role"
                                    name="role"
                                    value={formData.role.charAt(0).toUpperCase() + formData.role.slice(1)}
                                    icon={Shield}
                                    disabled
                                    isEditing={isEditing}
                                    onChange={handleChange}
                                />
                            </div>

                            <div className="mt-10 p-5 bg-medic-light/10 rounded-2xl border border-medic-light/20 flex items-start gap-4">
                                <Shield className="w-6 h-6 text-medic-dark flex-shrink-0 mt-0.5" />
                                <div className="text-xs text-medic-dark/70 leading-relaxed">
                                    <p className="font-bold mb-1 uppercase tracking-wider text-[10px]">Security Note</p>
                                    Your email and account role are verified identifiers and cannot be changed. For account recovery support, please contact help@mediscan.com.
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};

export default PatientProfile;
