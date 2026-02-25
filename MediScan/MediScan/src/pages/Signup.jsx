import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import emailjs from '@emailjs/browser';
import { useAuth } from '../contexts/AuthContext.jsx';
import { emailConfig } from '../config/emailConfig.js';

export default function Signup() {
    const navigate = useNavigate();
    const { login } = useAuth();
    const [step, setStep] = useState(1); // 1: Form, 2: OTP
    const [loading, setLoading] = useState(false);
    const [role, setRole] = useState('patient');

    const [formData, setFormData] = useState({
        firstName: '',
        lastName: '',
        email: '',
        phone: '',
        password: '',
        confirmPassword: ''
    });

    const [otp, setOtp] = useState(['', '', '', '', '', '']);
    const [error, setError] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
        setError('');
    };

    const validateForm = () => {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;

        if (!formData.firstName || !formData.lastName) return 'Name fields are required.';
        if (!emailRegex.test(formData.email)) return 'Please enter a valid email address.';
        if (formData.phone.length < 10) return 'Please enter a valid phone number.';
        if (!passwordRegex.test(formData.password)) {
            return 'Password must be at least 8 chars, include uppercase, lowercase, number & symbol.';
        }
        if (formData.password !== formData.confirmPassword) return 'Passwords do not match.';
        return null;
    };

    const handleSignup = (e) => {
        e.preventDefault();
        const validationError = validateForm();
        if (validationError) {
            setError(validationError);
            return;
        }

        setLoading(true);

        // Generate random 6-digit OTP
        const generatedOtp = Math.floor(100000 + Math.random() * 900000).toString();
        console.log("Generated OTP:", generatedOtp); // Fallback for testing if email fails

        // EmailJS Configuration from src/config/emailConfig.js
        const SERVICE_ID = emailConfig.serviceId;
        const TEMPLATE_ID = emailConfig.templateId;
        const PUBLIC_KEY = emailConfig.publicKey;



        const templateParams = {
            to_name: formData.firstName,
            to_email: formData.email,
            otp_code: generatedOtp,
        };

        // Attempt to send email
        emailjs.send(SERVICE_ID, TEMPLATE_ID, templateParams, PUBLIC_KEY)
            .then((response) => {
                console.log('SUCCESS!', response.status, response.text);
                alert(`OTP sent to ${formData.email}`);
                // Store generated OTP in state or localStorage to verify later
                // For security in a real app, verification should happen on backend. 
                // Here we store it in a temporary state variable for the frontend check.
                // We will append it to the otp state for verification logic.
                window.confirmationOtp = generatedOtp;
                setLoading(false);
                setStep(2);
            }, (err) => {
                console.log('FAILED...', err);
                alert(`Failed to send email. Check console for error. Falling back to Demo OTP: ${generatedOtp}`);
                window.confirmationOtp = generatedOtp;
                setLoading(false);
                setStep(2);
            });
    };

    const handleOtpChange = (element, index) => {
        if (isNaN(element.value)) return;
        const newOtp = [...otp];
        newOtp[index] = element.value;
        setOtp(newOtp);

        // Focus next input
        if (element.nextSibling && element.value) {
            element.nextSibling.focus();
        }
    };

    const verifyOtp = () => {
        const enteredOtp = otp.join('');
        // Compare against the stored OTP (window.confirmationOtp)
        if (enteredOtp !== window.confirmationOtp) {
            setError('Invalid OTP. Please try again.');
            return;
        }

        setLoading(true);
        // Simulate OTP verification
        setTimeout(() => {
            login(role); // Auto-login after signup
            setLoading(false);
            navigate(role === 'doctor' ? '/doctor-dashboard' : '/dashboard');
        }, 1500);
    };

    return (
        <div className="flex-1 flex items-center justify-center p-4 min-h-[calc(100vh-64px)]">
            <div className="max-w-md w-full bg-white p-8 rounded-2xl shadow-xl border border-slate-100 fade-in">
                <div className="text-center mb-8">
                    <div className="mx-auto flex justify-center mb-4">
                        <img src="/logo.jpg" alt="MediScan Logo" className="h-24 w-auto" />
                    </div>
                    <h2 className="text-3xl font-extrabold text-slate-900">
                        {step === 1 ? 'Create Account' : 'Verify Email'}
                    </h2>
                    <p className="mt-2 text-sm text-slate-600">
                        {step === 1
                            ? 'Join MediScan for AI-powered health insights'
                            : `Enter the code sent to ${formData.email}`}
                    </p>
                </div>

                {error && (
                    <div className="bg-red-50 text-red-600 p-3 rounded-lg text-sm mb-6 flex items-center gap-2">
                        <i className="fa-solid fa-circle-exclamation"></i>
                        {error}
                    </div>
                )}

                {step === 1 ? (
                    <form onSubmit={handleSignup} className="space-y-4">
                        <div className="flex p-1 bg-slate-100 rounded-xl mb-6">
                            <button
                                type="button"
                                onClick={() => setRole('patient')}
                                className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all ${role === 'patient' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-900'
                                    }`}
                            >
                                Patient
                            </button>
                            <button
                                type="button"
                                onClick={() => setRole('doctor')}
                                className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all ${role === 'doctor' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-900'
                                    }`}
                            >
                                Doctor
                            </button>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <input
                                name="firstName"
                                placeholder="First Name"
                                onChange={handleChange}
                                className="w-full px-4 py-3 rounded-lg border border-slate-300 focus:ring-2 focus:ring-green-500 focus:outline-none"
                            />
                            <input
                                name="lastName"
                                placeholder="Last Name"
                                onChange={handleChange}
                                className="w-full px-4 py-3 rounded-lg border border-slate-300 focus:ring-2 focus:ring-green-500 focus:outline-none"
                            />
                        </div>

                        <input
                            name="email"
                            type="email"
                            placeholder="Email Address"
                            onChange={handleChange}
                            className="w-full px-4 py-3 rounded-lg border border-slate-300 focus:ring-2 focus:ring-green-500 focus:outline-none"
                        />

                        <input
                            name="phone"
                            type="tel"
                            placeholder="Phone Number"
                            onChange={handleChange}
                            className="w-full px-4 py-3 rounded-lg border border-slate-300 focus:ring-2 focus:ring-green-500 focus:outline-none"
                        />

                        <div className="relative">
                            <input
                                name="password"
                                type={showPassword ? "text" : "password"}
                                placeholder="Password (Min 8 chars)"
                                onChange={handleChange}
                                className="w-full px-4 py-3 rounded-lg border border-slate-300 focus:ring-2 focus:ring-green-500 focus:outline-none pr-10"
                            />
                            <button
                                type="button"
                                onClick={() => setShowPassword(!showPassword)}
                                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                            >
                                <i className={`fa-solid ${showPassword ? 'fa-eye-slash' : 'fa-eye'}`}></i>
                            </button>
                        </div>

                        <div className="relative">
                            <input
                                name="confirmPassword"
                                type={showConfirmPassword ? "text" : "password"}
                                placeholder="Confirm Password"
                                onChange={handleChange}
                                className="w-full px-4 py-3 rounded-lg border border-slate-300 focus:ring-2 focus:ring-green-500 focus:outline-none pr-10"
                            />
                            <button
                                type="button"
                                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                            >
                                <i className={`fa-solid ${showConfirmPassword ? 'fa-eye-slash' : 'fa-eye'}`}></i>
                            </button>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full bg-green-600 text-white font-bold py-3 rounded-xl hover:bg-green-700 transition shadow-lg mt-4"
                        >
                            {loading ? <span className="loader w-5 h-5 border-2 border-white/30 border-t-white inline-block"></span> : 'Sign Up'}
                        </button>

                        <p className="text-center text-sm text-slate-500 mt-4">
                            Already have an account?{' '}
                            <Link to="/login" className="text-green-600 font-bold hover:underline">
                                Login
                            </Link>
                        </p>
                    </form>
                ) : (
                    <div className="space-y-6">
                        <div className="flex justify-center gap-2">
                            {otp.map((digit, index) => (
                                <input
                                    key={index}
                                    type="text"
                                    maxLength="1"
                                    value={digit}
                                    onChange={(e) => handleOtpChange(e.target, index)}
                                    onFocus={(e) => e.target.select()}
                                    className="w-12 h-12 text-center text-2xl font-bold border border-slate-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:outline-none"
                                />
                            ))}
                        </div>

                        <button
                            onClick={verifyOtp}
                            disabled={loading}
                            className="w-full bg-green-600 text-white font-bold py-3 rounded-xl hover:bg-green-700 transition shadow-lg"
                        >
                            {loading ? <span className="loader w-5 h-5 border-2 border-white/30 border-t-white inline-block"></span> : 'Verify & Login'}
                        </button>

                        <button
                            onClick={() => setStep(1)}
                            className="w-full text-slate-500 text-sm hover:text-slate-800"
                        >
                            Back to Signup
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
