import React, { useState } from 'react';
import { adminApi } from '../../services/api';
import { toast } from 'react-toastify';
import { UserPlus, Mail, Lock, Phone, User, CheckCircle } from 'lucide-react';

const CreateAdmin = () => {
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({
        first_name: '',
        last_name: '',
        email: '',
        phone_number: '',
        password: ''
    });

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            await adminApi.createAdmin(formData);
            toast.success("New Admin created successfully!");
            setFormData({
                first_name: '',
                last_name: '',
                email: '',
                phone_number: '',
                password: ''
            });
        } catch (error) {
            toast.error(error.response?.data?.error || "Failed to create admin");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-2xl mx-auto">
            <h1 className="text-2xl font-bold text-gray-900 mb-6">Create New Admin</h1>

            <div className="bg-white rounded-3xl p-8 shadow-sm border border-gray-100">
                <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-1.5">
                            <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">First Name</label>
                            <div className="relative">
                                <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                                <input
                                    type="text"
                                    name="first_name"
                                    required
                                    value={formData.first_name}
                                    onChange={handleChange}
                                    className="w-full pl-11 pr-4 py-3 bg-neutral-soft border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-sm transition-all focus:ring-4 focus:ring-medic-dark/5 outline-none"
                                />
                            </div>
                        </div>
                        <div className="space-y-1.5">
                            <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Last Name</label>
                            <div className="relative">
                                <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                                <input
                                    type="text"
                                    name="last_name"
                                    required
                                    value={formData.last_name}
                                    onChange={handleChange}
                                    className="w-full pl-11 pr-4 py-3 bg-neutral-soft border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-sm transition-all focus:ring-4 focus:ring-medic-dark/5 outline-none"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="space-y-1.5">
                        <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Email Address</label>
                        <div className="relative">
                            <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                            <input
                                type="email"
                                name="email"
                                required
                                value={formData.email}
                                onChange={handleChange}
                                className="w-full pl-11 pr-4 py-3 bg-neutral-soft border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-sm transition-all focus:ring-4 focus:ring-medic-dark/5 outline-none"
                            />
                        </div>
                    </div>

                    <div className="space-y-1.5">
                        <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Phone Number</label>
                        <div className="relative">
                            <Phone className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                            <input
                                type="tel"
                                name="phone_number"
                                required
                                value={formData.phone_number}
                                onChange={handleChange}
                                placeholder="+977"
                                className="w-full pl-11 pr-4 py-3 bg-neutral-soft border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-sm transition-all focus:ring-4 focus:ring-medic-dark/5 outline-none"
                            />
                        </div>
                    </div>

                    <div className="space-y-1.5">
                        <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Password</label>
                        <div className="relative">
                            <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                            <input
                                type="password"
                                name="password"
                                required
                                value={formData.password}
                                onChange={handleChange}
                                className="w-full pl-11 pr-4 py-3 bg-neutral-soft border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-sm transition-all focus:ring-4 focus:ring-medic-dark/5 outline-none"
                            />
                        </div>
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-medic-dark text-white py-4 rounded-xl font-bold text-sm shadow-lg shadow-medic-dark/10 hover:bg-medic-primary transition-all active:scale-[0.98] disabled:opacity-70 disabled:pointer-events-none flex items-center justify-center gap-2"
                    >
                        {loading ? (
                            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        ) : (
                            <>
                                <UserPlus className="w-5 h-5" />
                                Create Admin Account
                            </>
                        )}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default CreateAdmin;
