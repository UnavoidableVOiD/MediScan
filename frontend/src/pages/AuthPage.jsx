import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HeartPulse, Eye, EyeOff, Mail, Lock, User, Phone, ArrowRight, ShieldCheck } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { toast } from 'react-toastify';
import { useGoogleLogin } from '@react-oauth/google';
import { registerUser, loginUser, googleLogin } from '../store/slices/authSlice';

const AuthPage = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const dispatch = useDispatch();
    const { loading, isVerifying, isAuthenticated, user } = useSelector(state => state.auth);

    // Determine if we start on signup or login from URL
    const isInitialSignup = location.pathname.includes('signup');
    const [isLogin, setIsLogin] = useState(!isInitialSignup);
    const [showPassword, setShowPassword] = useState(false);
    const [role, setRole] = useState('patient');

    // Effect to handle redirection
    React.useEffect(() => {
        if (isVerifying) {
            navigate('/verify-otp');
            return;
        }

        if (isAuthenticated && user) {
            if (user.role === 'ADMIN' || user.is_superuser) {
                navigate('/admin/dashboard');
            } else if (user.role === 'DOCTOR') {
                navigate('/doctor-dashboard');
            } else {
                navigate('/dashboard');
            }
        }
    }, [isVerifying, isAuthenticated, user, navigate]);

    const [formData, setFormData] = useState({
        firstName: '',
        lastName: '',
        email: '',
        phone: '',
        password: '',
        confirmPassword: '',
        specialization: ''
    });

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (isLogin) {
            dispatch(loginUser({
                email: formData.email,
                password: formData.password,
                role: role.toUpperCase() // Backend expects uppercase
            }));
        } else {
            if (formData.password !== formData.confirmPassword) {
                toast.error("Passwords do not match!");
                return;
            }

            const signupData = {
                first_name: formData.firstName,
                last_name: formData.lastName,
                email: formData.email,
                phone_number: formData.phone,
                password: formData.password,
                confirm_password: formData.confirmPassword,
                role: role.toUpperCase(),
                specialization: role === 'doctor' ? formData.specialization : null
            };
            dispatch(registerUser(signupData));
        }
    };

    const handleGoogleLoginSuccess = (tokenResponse) => {
        dispatch(googleLogin(tokenResponse.access_token))
            .unwrap()
            .then((response) => {
                const userRole = response.user.role;
                if (userRole === 'ADMIN') {
                    navigate('/admin/dashboard');
                } else if (userRole === 'DOCTOR') {
                    navigate('/doctor-dashboard');
                } else {
                    navigate('/dashboard');
                }
            });
    };

    const loginWithGoogle = useGoogleLogin({
        onSuccess: handleGoogleLoginSuccess,
        onError: () => toast.error("Google Login Failed")
    });

    return (
        <div className="min-h-[calc(100vh-80px)] flex items-center justify-center p-6 bg-neutral-background">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="max-w-5xl w-full bg-white rounded-3xl shadow-2xl shadow-medic-dark/10 overflow-hidden flex flex-col md:flex-row min-h-[600px]"
            >
                {/* Left Side: Brand Panel */}
                <div className="md:w-5/12 bg-medic-dark p-8 md:p-12 text-white flex flex-col justify-between relative overflow-hidden">
                    <div className="relative z-10">
                        <div className="flex items-center gap-2 mb-6 sm:mb-8">
                            <HeartPulse className="w-8 h-8 sm:w-10 sm:h-10 text-medic-accent" />
                            <span className="text-2xl sm:text-3xl font-bold tracking-tight">Mediscan</span>
                        </div>
                        <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-4 sm:mb-6 leading-tight">
                            Secure access to your medical insights.
                        </h1>
                        <p className="text-medic-light/70 text-base sm:text-lg leading-relaxed">
                            Your health data is encrypted and handled with the highest standards of medical privacy.
                        </p>
                    </div>

                    {/* <div className="relative z-10 pt-8 sm:pt-12 hidden sm:block">
                        <div className="flex items-center gap-4 mb-4">
                            <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center">
                                <ShieldCheck className="w-6 h-6 text-medic-accent" />
                            </div>
                            <span className="text-sm font-medium opacity-80">HIPAA Compliant Encryption</span>
                        </div>
                    </div> */}

                    {/* Abstract Wave Graphic */}
                    <div className="absolute bottom-0 left-0 w-full h-1/2 opacity-10 pointer-events-none hidden md:block">
                        <svg viewBox="0 0 500 200" className="w-full h-full">
                            <path d="M0,100 C150,200 350,0 500,100 L500,200 L0,200 Z" fill="white" />
                        </svg>
                    </div>
                </div>

                {/* Right Side: Auth Card */}
                <div className="md:w-7/12 p-6 sm:p-8 md:p-14 bg-white">
                    <div className="mb-8 sm:mb-10 flex flex-col sm:flex-row gap-6 items-center justify-between">
                        <div className="text-center sm:text-left">
                            <h2 className="text-2xl font-bold text-gray-900 mb-1">
                                {isLogin ? 'Welcome back' : 'Create an account'}
                            </h2>
                            <p className="text-gray-500 text-sm">Please enter your details below.</p>
                        </div>

                        <div className="bg-neutral-soft p-1 rounded-full flex w-full sm:w-auto">
                            <button
                                onClick={() => setIsLogin(true)}
                                className={`flex-1 sm:flex-none px-6 py-2 rounded-full text-sm font-bold transition-all ${isLogin ? 'bg-white text-medic-dark shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                            >
                                Login
                            </button>
                            <button
                                onClick={() => setIsLogin(false)}
                                className={`flex-1 sm:flex-none px-6 py-2 rounded-full text-sm font-bold transition-all ${!isLogin ? 'bg-white text-medic-dark shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                            >
                                Sign Up
                            </button>
                        </div>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-5">
                        <AnimatePresence mode="wait">
                            {!isLogin && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="grid grid-cols-1 sm:grid-cols-2 gap-4 overflow-hidden"
                                >
                                    <div className="space-y-1.5">
                                        <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">First Name</label>
                                        <div className="relative">
                                            <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                                            <input
                                                type="text"
                                                name="firstName"
                                                required={!isLogin}
                                                placeholder="first name"
                                                className="w-full pl-11 pr-4 py-3 bg-neutral-soft border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-sm transition-all focus:ring-4 focus:ring-medic-dark/5 outline-none"
                                                onChange={handleChange}
                                            />
                                        </div>
                                    </div>
                                    <div className="space-y-1.5">
                                        <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Last Name</label>
                                        <div className="relative">
                                            <input
                                                type="text"
                                                name="lastName"
                                                required={!isLogin}
                                                placeholder="Last Name"
                                                className="w-full px-4 py-3 bg-neutral-soft border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-sm transition-all focus:ring-4 focus:ring-medic-dark/5 outline-none"
                                                onChange={handleChange}
                                            />
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        <div className="space-y-1.5">
                            <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Email Address</label>
                            <div className="relative">
                                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                                <input
                                    type="email"
                                    name="email"
                                    required
                                    placeholder="example@example.com"
                                    className="w-full pl-11 pr-4 py-3 bg-neutral-soft border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-sm transition-all focus:ring-4 focus:ring-medic-dark/5 outline-none"
                                    onChange={handleChange}
                                />
                            </div>
                        </div>

                        <AnimatePresence mode="wait">
                            {!isLogin && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="space-y-1.5 overflow-hidden"
                                >
                                    <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Phone Number</label>
                                    <div className="relative">
                                        <Phone className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                                        <input
                                            type="tel"
                                            name="phone"
                                            required={!isLogin}
                                            placeholder="+9779800000000"
                                            className="w-full pl-11 pr-4 py-3 bg-neutral-soft border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-sm transition-all focus:ring-4 focus:ring-medic-dark/5 outline-none"
                                            onChange={handleChange}
                                        />
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        <div className="space-y-1.5">
                            <div className="flex justify-between items-center">
                                <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Password</label>
                                {isLogin && <button type="button" className="text-[10px] font-bold text-medic-dark hover:underline uppercase tracking-wider underline-offset-4">Forgot Password?</button>}
                            </div>
                            <div className="relative">
                                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                                <input
                                    type={showPassword ? 'text' : 'password'}
                                    name="password"
                                    required
                                    placeholder="••••••••"
                                    className="w-full pl-11 pr-12 py-3 bg-neutral-soft border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-sm transition-all focus:ring-4 focus:ring-medic-dark/5 outline-none"
                                    onChange={handleChange}
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                                >
                                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                </button>
                            </div>
                        </div>

                        {!isLogin && (
                            <div className="space-y-1.5 overflow-hidden">
                                <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Confirm Password</label>
                                <div className="relative">
                                    <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                                    <input
                                        type={showPassword ? 'text' : 'password'}
                                        name="confirmPassword"
                                        required={!isLogin}
                                        placeholder="••••••••"
                                        className="w-full pl-11 pr-4 py-3 bg-neutral-soft border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-sm transition-all focus:ring-4 focus:ring-medic-dark/5 outline-none"
                                        onChange={handleChange}
                                    />
                                </div>
                            </div>
                        )}

                        <AnimatePresence mode="wait">
                            {!isLogin && role === 'doctor' && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="space-y-1.5 overflow-hidden"
                                >
                                    <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Specialization</label>
                                    <div className="relative">
                                        <select
                                            name="specialization"
                                            required={!isLogin && role === 'doctor'}
                                            value={formData.specialization}
                                            onChange={handleChange}
                                            className="w-full px-4 py-3 bg-neutral-soft border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-sm transition-all focus:ring-4 focus:ring-medic-dark/5 outline-none appearance-none cursor-pointer"
                                        >
                                            <option value="">Select Specialization</option>
                                            <option value="GENERAL_PHYSICIAN">General Physician</option>
                                            <option value="CARDIOLOGIST">Cardiologist</option>
                                            <option value="DERMATOLOGIST">Dermatologist</option>
                                            <option value="NEUROLOGIST">Neurologist</option>
                                            <option value="ORTHOPEDIC">Orthopedic</option>
                                            <option value="PEDIATRICIAN">Pediatrician</option>
                                            <option value="OTHER">Other</option>
                                        </select>
                                        <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-gray-400">
                                            <ArrowRight className="w-4 h-4 rotate-90" />
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        <div className="space-y-3">
                            <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Role Selection</label>
                            <div className="grid grid-cols-2 gap-3">
                                <button
                                    type="button"
                                    onClick={() => setRole('patient')}
                                    className={`py-3 rounded-xl text-sm font-bold border-2 transition-all ${role === 'patient' ? 'border-medic-dark bg-medic-light/20 text-medic-dark' : 'border-neutral-soft bg-neutral-soft text-gray-500'}`}
                                >
                                    Patient
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setRole('doctor')}
                                    className={`py-3 rounded-xl text-sm font-bold border-2 transition-all ${role === 'doctor' ? 'border-medic-dark bg-medic-light/20 text-medic-dark' : 'border-neutral-soft bg-neutral-soft text-gray-500'}`}
                                >
                                    Doctor
                                </button>
                            </div>
                            {!isLogin && <p className="text-[10px] text-gray-400 italic">Note: Role cannot be changed once the account is created.</p>}
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full bg-medic-dark text-white py-4 rounded-xl font-bold text-lg shadow-lg shadow-medic-dark/10 hover:bg-medic-primary transition-all active:scale-[0.98] disabled:opacity-70 disabled:pointer-events-none flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            ) : (
                                <>
                                    {isLogin ? 'Login' : 'Create Account'}
                                    <ArrowRight className="w-5 h-5" />
                                </>
                            )}
                        </button>

                        <div className="relative py-4">
                            <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-gray-100"></div></div>
                            <div className="relative flex justify-center text-xs uppercase"><span className="bg-white px-2 text-gray-400 font-bold tracking-widest">Or continue with</span></div>
                        </div>

                        <button
                            type="button"
                            onClick={() => loginWithGoogle()}
                            className="w-full bg-white border-2 border-neutral-soft py-3.5 rounded-xl font-bold text-sm text-gray-700 hover:bg-neutral-soft transition-all flex items-center justify-center gap-3"
                        >
                            <svg className="w-5 h-5" viewBox="0 0 24 24">
                                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" /><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" /><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05" /><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                            </svg>
                            Google
                        </button>
                    </form>
                </div>
            </motion.div>
        </div>
    );
};

export default AuthPage;
