import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { updateProfile } from '../store/slices/authSlice';
import {
    User,
    Mail,
    Phone,
    Shield,
    Save,
    Loader2,
    CheckCircle2,
    Edit3,
    X
} from 'lucide-react';
import { toast } from 'react-toastify';
import { parseError } from '../utils/errorParser';

const Profile = () => {
    const dispatch = useDispatch();
    const { user, loading } = useSelector((state) => state.auth);

    // Form data state
    const [formData, setFormData] = useState({
        first_name: '',
        last_name: '',
        phone_number: '',
    });

    // Tracking which fields are in edit mode
    const [editingFields, setEditingFields] = useState({
        first_name: false,
        last_name: false,
        phone_number: false
    });

    useEffect(() => {
        if (user) {
            setFormData({
                first_name: user.first_name || '',
                last_name: user.last_name || '',
                phone_number: user.phone_number || '',
            });
        }
    }, [user]);

    const toggleEdit = (field) => {
        setEditingFields(prev => ({ ...prev, [field]: !prev[field] }));
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleCancel = (field) => {
        // Reset field to original value
        if (user) {
            setFormData(prev => ({ ...prev, [field]: user[field] || '' }));
        }
        toggleEdit(field);
    };

    const handleSave = async (field) => {
        const value = formData[field];
        if (!value && (field === 'first_name' || field === 'last_name')) {
            toast.error("This field cannot be empty.");
            return;
        }

        const resultAction = await dispatch(updateProfile({ [field]: value }));
        if (updateProfile.fulfilled.match(resultAction)) {
            toast.success(`${field.replace('_', ' ')} updated!`);
            toggleEdit(field);
        } else {
            const errorMsg = parseError(resultAction.payload);
            toast.error(errorMsg);
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 pt-24 pb-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-4xl mx-auto">
                <div className="bg-white rounded-3xl shadow-xl shadow-gray-200/50 border border-gray-100 overflow-hidden">
                    {/* Hero Cover */}
                    <div className="h-32 bg-gradient-to-r from-blue-600 to-indigo-600"></div>

                    {/* Profile Info Header */}
                    <div className="px-8 pb-8 relative">
                        <div className="flex flex-col sm:flex-row sm:items-end gap-6 -mt-12 sm:-mt-16">
                            <div className="h-24 w-24 sm:h-32 sm:w-32 bg-white rounded-3xl p-1 shadow-xl shadow-gray-200">
                                <div className="h-full w-full bg-blue-100 rounded-2xl flex items-center justify-center">
                                    <User size={64} className="text-blue-600" />
                                </div>
                            </div>
                            <div className="flex-1 pb-2">
                                <h1 className="text-3xl font-bold text-gray-900 capitalize">{user?.first_name} {user?.last_name}</h1>
                                <div className="flex items-center gap-2 mt-1">
                                    <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-xs font-bold border border-blue-100 uppercase tracking-wide">
                                        <CheckCircle2 size={12} />
                                        {user?.role || 'Patient'}
                                    </span>
                                    <span className="text-gray-500 text-sm font-medium flex items-center gap-1">
                                        <Mail size={14} />
                                        {user?.email}
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Profile Sections */}
                        <div className="mt-12 space-y-8">
                            <div>
                                <h3 className="text-lg font-bold text-gray-900 border-b border-gray-100 pb-3 mb-6">Personal Information</h3>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-8">
                                    {/* First Name Field */}
                                    <ProfileField
                                        label="First Name"
                                        name="first_name"
                                        value={formData.first_name}
                                        isEditing={editingFields.first_name}
                                        onEdit={() => toggleEdit('first_name')}
                                        onChange={handleChange}
                                        onSave={() => handleSave('first_name')}
                                        onCancel={() => handleCancel('first_name')}
                                        loading={loading}
                                    />

                                    {/* Last Name Field */}
                                    <ProfileField
                                        label="Last Name"
                                        name="last_name"
                                        value={formData.last_name}
                                        isEditing={editingFields.last_name}
                                        onEdit={() => toggleEdit('last_name')}
                                        onChange={handleChange}
                                        onSave={() => handleSave('last_name')}
                                        onCancel={() => handleCancel('last_name')}
                                        loading={loading}
                                    />

                                    {/* Email Field (Static) */}
                                    <div className="space-y-1">
                                        <label className="text-xs font-bold text-gray-500 uppercase tracking-wider">Email Address</label>
                                        <div className="flex items-center justify-between py-2 border-b border-gray-100">
                                            <span className="text-gray-600 font-medium">{user?.email}</span>
                                            <Shield size={16} className="text-gray-300" />
                                        </div>
                                        <p className="text-[10px] text-gray-400">Fixed account identifier</p>
                                    </div>

                                    {/* Phone Number Field */}
                                    <ProfileField
                                        label="Phone Number"
                                        name="phone_number"
                                        value={formData.phone_number}
                                        isEditing={editingFields.phone_number}
                                        onEdit={() => toggleEdit('phone_number')}
                                        onChange={handleChange}
                                        onSave={() => handleSave('phone_number')}
                                        onCancel={() => handleCancel('phone_number')}
                                        loading={loading}
                                        icon={<Phone size={14} />}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const ProfileField = ({ label, name, value, isEditing, onEdit, onChange, onSave, onCancel, loading, icon }) => (
    <div className="space-y-1 text-left group">
        <label className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
            {icon}
            {label}
        </label>
        <div className="flex items-center justify-between gap-4 py-2 border-b border-gray-100 min-h-[48px]">
            {isEditing ? (
                <div className="flex-1 flex items-center gap-2">
                    <input
                        type="text"
                        name={name}
                        value={value}
                        onChange={onChange}
                        autoFocus
                        className="flex-1 bg-blue-50 border-none rounded px-2 py-1 text-gray-900 font-medium focus:ring-0"
                    />
                    <div className="flex items-center gap-1">
                        <button
                            onClick={onSave}
                            disabled={loading}
                            className="p-1.5 text-emerald-600 hover:bg-emerald-50 rounded transition-colors disabled:opacity-50"
                        >
                            {loading ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
                        </button>
                        <button
                            onClick={onCancel}
                            className="p-1.5 text-gray-400 hover:bg-gray-100 rounded transition-colors"
                        >
                            <X size={16} />
                        </button>
                    </div>
                </div>
            ) : (
                <>
                    <span className="text-gray-900 font-bold">{value || <span className="text-gray-300 font-medium">Not set</span>}</span>
                    <button
                        onClick={onEdit}
                        className="p-1.5 text-blue-600 hover:bg-blue-50 rounded transition-all opacity-0 group-hover:opacity-100 md:opacity-100"
                    >
                        <Edit3 size={16} />
                    </button>
                </>
            )}
        </div>
    </div>
);

export default Profile;
