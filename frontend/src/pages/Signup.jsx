import React, { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Link, useNavigate } from 'react-router-dom';
import { signup } from '../store/slices/authSlice';
import { toast } from 'react-toastify';
import { Shield, User, Mail, Phone, Lock, Eye, EyeOff, Loader2, Activity } from 'lucide-react';
import { parseError } from '../utils/errorParser';

const Signup = () => {
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const { loading } = useSelector((state) => state.auth);

    const [formData, setFormData] = useState({
        first_name: '',
        last_name: '',
        email: '',
        phone_number: '',
        password: '',
        confirm_password: '',
        role: 'PATIENT',
    });

    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const validateForm = () => {
        if (formData.password !== formData.confirm_password) {
            toast.error("Passwords do not match");
            return false;
        }
        if (formData.password.length < 8) {
            toast.error("Password must be at least 8 characters long");
            return false;
        }
        if (!formData.phone_number) {
            toast.error("Phone number is required");
            return false;
        }
        return true;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!validateForm()) return;

        const resultAction = await dispatch(signup(formData));
        if (signup.fulfilled.match(resultAction)) {
            toast.success("OTP sent to your email!");
            navigate('/otp-verify');
        } else {
            const errorMessage = parseError(resultAction.payload);
            toast.error(errorMessage);
        }
    };

    return (
        <div className="min-h-screen pt-24 pb-12 flex flex-col justify-center bg-gray-50 px-4 sm:px-6 lg:px-8">
            <div className="sm:mx-auto sm:w-full sm:max-w-md text-center">
                <div className="flex justify-center mb-6">
                    <div className="bg-blue-600 p-3 rounded-2xl shadow-lg shadow-blue-200">
                        <Shield className="h-10 w-10 text-white" />
                    </div>
                </div>
                <h2 className="text-3xl font-extrabold text-gray-900">Create your account</h2>
                <p className="mt-2 text-sm text-gray-600 font-medium">Start your health journey with MediScan</p>
            </div>

            <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-xl">
                <div className="bg-white py-8 px-6 shadow-xl shadow-gray-200/50 rounded-3xl border border-gray-100 sm:px-10">
                    <form className="space-y-6" onSubmit={handleSubmit}>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                            <div>
                                <label className="block text-sm font-bold text-gray-700 mb-2">First Name</label>
                                <div className="relative">
                                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                        <User className="h-5 w-5 text-gray-400" />
                                    </div>
                                    <input
                                        name="first_name"
                                        type="text"
                                        required
                                        className="appearance-none block w-full pl-10 pr-3 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all font-medium"
                                        placeholder="John"
                                        value={formData.first_name}
                                        onChange={handleChange}
                                    />
                                </div>
                            </div>
                            <div>
                                <label className="block text-sm font-bold text-gray-700 mb-2">Last Name</label>
                                <div className="relative">
                                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                        <User className="h-5 w-5 text-gray-400" />
                                    </div>
                                    <input
                                        name="last_name"
                                        type="text"
                                        required
                                        className="appearance-none block w-full pl-10 pr-3 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all font-medium"
                                        placeholder="Doe"
                                        value={formData.last_name}
                                        onChange={handleChange}
                                    />
                                </div>
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-bold text-gray-700 mb-2">Email address</label>
                            <div className="relative">
                                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                    <Mail className="h-5 w-5 text-gray-400" />
                                </div>
                                <input
                                    name="email"
                                    type="email"
                                    required
                                    className="appearance-none block w-full pl-10 pr-3 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all font-medium"
                                    placeholder="john@example.com"
                                    value={formData.email}
                                    onChange={handleChange}
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-bold text-gray-700 mb-2">Phone Number</label>
                            <div className="relative">
                                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                    <Phone className="h-5 w-5 text-gray-400" />
                                </div>
                                <input
                                    name="phone_number"
                                    type="tel"
                                    required
                                    className="appearance-none block w-full pl-10 pr-3 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all font-medium"
                                    placeholder="+1234567890"
                                    value={formData.phone_number}
                                    onChange={handleChange}
                                />
                            </div>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                            <div>
                                <label className="block text-sm font-bold text-gray-700 mb-2">Password</label>
                                <div className="relative">
                                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                        <Lock className="h-5 w-5 text-gray-400" />
                                    </div>
                                    <input
                                        name="password"
                                        type={showPassword ? "text" : "password"}
                                        required
                                        className="appearance-none block w-full pl-10 pr-10 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all font-medium"
                                        placeholder="••••••••"
                                        value={formData.password}
                                        onChange={handleChange}
                                    />
                                    <button
                                        type="button"
                                        className="absolute inset-y-0 right-0 pr-3 flex items-center"
                                        onClick={() => setShowPassword(!showPassword)}
                                    >
                                        {showPassword ? <EyeOff className="h-5 w-5 text-gray-400" /> : <Eye className="h-5 w-5 text-gray-400" />}
                                    </button>
                                </div>
                            </div>
                            <div>
                                <label className="block text-sm font-bold text-gray-700 mb-2">Confirm Password</label>
                                <div className="relative">
                                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                        <Lock className="h-5 w-5 text-gray-400" />
                                    </div>
                                    <input
                                        name="confirm_password"
                                        type={showConfirmPassword ? "text" : "password"}
                                        required
                                        className="appearance-none block w-full pl-10 pr-10 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all font-medium"
                                        placeholder="••••••••"
                                        value={formData.confirm_password}
                                        onChange={handleChange}
                                    />
                                    <button
                                        type="button"
                                        className="absolute inset-y-0 right-0 pr-3 flex items-center"
                                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                    >
                                        {showConfirmPassword ? <EyeOff className="h-5 w-5 text-gray-400" /> : <Eye className="h-5 w-5 text-gray-400" />}
                                    </button>
                                </div>
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-bold text-gray-700 mb-4 text-center">Select Role</label>
                            <div className="grid grid-cols-2 gap-4">
                                <button
                                    type="button"
                                    onClick={() => setFormData({ ...formData, role: 'PATIENT' })}
                                    className={`py-4 px-6 rounded-2xl border-2 font-bold transition-all flex flex-col items-center gap-2 ${formData.role === 'PATIENT'
                                            ? 'border-blue-600 bg-blue-50 text-blue-700 shadow-sm shadow-blue-100'
                                            : 'border-gray-100 bg-white text-gray-500 hover:border-gray-200'
                                        }`}
                                >
                                    <User className="h-6 w-6" />
                                    Patient
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setFormData({ ...formData, role: 'DOCTOR' })}
                                    className={`py-4 px-6 rounded-2xl border-2 font-bold transition-all flex flex-col items-center gap-2 ${formData.role === 'DOCTOR'
                                            ? 'border-emerald-500 bg-emerald-50 text-emerald-700 shadow-sm shadow-emerald-100'
                                            : 'border-gray-100 bg-white text-gray-500 hover:border-gray-200'
                                        }`}
                                >
                                    <Activity className="h-6 w-6" />
                                    Doctor
                                </button>
                            </div>
                        </div>

                        <div>
                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full flex justify-center py-4 px-4 border border-transparent rounded-2xl shadow-lg text-lg font-bold text-white bg-gradient-to-r from-blue-600 to-emerald-500 hover:from-blue-700 hover:to-emerald-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all disabled:opacity-50"
                            >
                                {loading ? <Loader2 className="animate-spin h-6 w-6" /> : "Sign Up"}
                            </button>
                        </div>
                    </form>

                    <div className="mt-8">
                        <div className="relative">
                            <div className="absolute inset-0 flex items-center">
                                <div className="w-full border-t border-gray-200"></div>
                            </div>
                            <div className="relative flex justify-center text-sm font-bold">
                                <span className="px-4 bg-white text-gray-500 uppercase tracking-widest text-xs">Already have an account?</span>
                            </div>
                        </div>

                        <div className="mt-6 text-center">
                            <Link to="/login" className="text-blue-600 hover:text-blue-700 font-bold transition-colors">
                                Sign in here
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Signup;
