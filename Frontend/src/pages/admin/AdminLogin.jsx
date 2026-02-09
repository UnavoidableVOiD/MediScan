import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDispatch } from 'react-redux';
import { ShieldCheck, Lock, Mail, ArrowRight } from 'lucide-react';
import { toast } from 'react-toastify';
import { adminApi } from '../../services/api';
import { setCredentials } from '../../store/slices/authSlice';

const AdminLogin = () => {
    const navigate = useNavigate();
    const dispatch = useDispatch();
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({
        email: '',
        password: ''
    });

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const response = await adminApi.login(formData);
            dispatch(setCredentials({
                user: response.data.user,
                token: null // Handled by cookie
            }));

            if (response.data.user.role === 'ADMIN' || response.data.user.is_superuser) {
                toast.success("Welcome back, Admin!");
                navigate('/admin/dashboard');
            } else {
                toast.error("Unauthorized access.");
            }
        } catch (error) {
            toast.error(error.response?.data?.error || "Login failed");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-neutral-background p-6">
            <div className="max-w-md w-full bg-white rounded-3xl shadow-xl overflow-hidden p-8 md:p-12">
                <div className="text-center mb-8">
                    <div className="w-16 h-16 bg-medic-dark/5 rounded-full flex items-center justify-center mx-auto mb-4">
                        <ShieldCheck className="w-8 h-8 text-medic-dark" />
                    </div>
                    <h1 className="text-2xl font-bold text-gray-900">Admin Portal</h1>
                    <p className="text-gray-500 mt-2">Secure access for administrators</p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
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
                                placeholder="admin@mediscan.com"
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
                                placeholder="••••••••"
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
                                Access Portal
                                <ArrowRight className="w-4 h-4" />
                            </>
                        )}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default AdminLogin;
