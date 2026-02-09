import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { ShieldCheck, Mail, ArrowRight, RefreshCw } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { toast } from 'react-toastify';
import { verifyOTP, clearVerification, loginUser } from '../store/slices/authSlice';

const OtpVerification = () => {
    const navigate = useNavigate();
    const dispatch = useDispatch();
    const { tempEmail, isVerifying, loading, isAuthenticated, flowType } = useSelector(state => state.auth);

    const [otp, setOtp] = useState(['', '', '', '', '', '']);
    const [timer, setTimer] = useState(60);
    const inputRefs = useRef([]);

    useEffect(() => {
        if (!isVerifying || !tempEmail) {
            if (!isAuthenticated) {
                navigate('/login');
            }
        }
    }, [isVerifying, tempEmail, navigate, isAuthenticated]);

    useEffect(() => {
        if (isAuthenticated) {
            navigate('/dashboard');
        }
    }, [isAuthenticated, navigate]);

    useEffect(() => {
        let interval = null;
        if (timer > 0) {
            interval = setInterval(() => {
                setTimer((prev) => prev - 1);
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [timer]);

    const handleChange = (index, value) => {
        if (isNaN(value)) return;

        const newOtp = [...otp];
        newOtp[index] = value.substring(value.length - 1);
        setOtp(newOtp);

        // Auto-focus next input
        if (value && index < 5) {
            inputRefs.current[index + 1].focus();
        }
    };

    const handleKeyDown = (index, e) => {
        if (e.key === 'Backspace' && !otp[index] && index > 0) {
            inputRefs.current[index - 1].focus();
        }
    };

    const handlePaste = (e) => {
        e.preventDefault();
        const data = e.clipboardData.getData('text').slice(0, 6).split('');
        const newOtp = [...otp];
        data.forEach((char, i) => {
            if (!isNaN(char)) newOtp[i] = char;
        });
        setOtp(newOtp);
        if (data.length > 0) {
            inputRefs.current[Math.min(data.length, 5)].focus();
        }
    };

    const handleVerify = (e) => {
        e.preventDefault();
        const otpValue = otp.join('');
        if (otpValue.length < 6) {
            toast.error("Please enter complete 6-digit OTP.");
            return;
        }

        dispatch(verifyOTP({
            email: tempEmail,
            otp: otpValue,
            type: flowType || 'register'
        }));
        // Note: The logic for determining 'login' or 'register' flow type might need state sync.
        // For now, let's assume 'register' is default or backend handles it.
        // Actually, the backend VerifyOTPView checks 'type'.
    };

    const handleResend = () => {
        setTimer(60);
        dispatch(loginUser({ email: tempEmail })); // Re-triggering login sends OTP
    };

    return (
        <div className="min-h-[calc(100vh-80px)] flex items-center justify-center p-6 bg-neutral-background">
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="max-w-md w-full bg-white rounded-3xl shadow-2xl shadow-medic-dark/10 p-10 text-center"
            >
                <div className="w-20 h-20 bg-medic-light/30 rounded-full flex items-center justify-center mx-auto mb-8">
                    <ShieldCheck className="w-10 h-10 text-medic-dark" />
                </div>

                <h2 className="text-2xl font-bold text-gray-900 mb-3">Verify your identity</h2>
                <div className="flex items-center justify-center gap-2 text-gray-500 text-sm mb-10">
                    <Mail className="w-4 h-4" />
                    <span>OTP sent to <strong>{tempEmail}</strong></span>
                </div>

                <form onSubmit={handleVerify} className="space-y-8">
                    <div className="flex justify-between gap-1.5 sm:gap-2">
                        {otp.map((digit, index) => (
                            <input
                                key={index}
                                ref={el => inputRefs.current[index] = el}
                                type="text"
                                maxLength="1"
                                value={digit}
                                onChange={e => handleChange(index, e.target.value)}
                                onKeyDown={e => handleKeyDown(index, e)}
                                onPaste={handlePaste}
                                className="w-10 h-12 sm:w-12 sm:h-14 bg-neutral-soft border-2 border-transparent focus:border-medic-dark focus:bg-white rounded-xl text-center text-lg sm:text-xl font-bold transition-all outline-none"
                            />
                        ))}
                    </div>

                    <div className="space-y-4">
                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full bg-medic-dark text-white py-4 rounded-xl font-bold text-lg shadow-lg shadow-medic-dark/10 hover:bg-medic-primary transition-all active:scale-[0.98] disabled:opacity-70 flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            ) : (
                                <>
                                    Verify Account
                                    <ArrowRight className="w-5 h-5" />
                                </>
                            )}
                        </button>

                        <div className="text-sm font-medium">
                            {timer > 0 ? (
                                <p className="text-gray-400">Resend code in <span className="text-medic-dark font-bold">{timer}s</span></p>
                            ) : (
                                <button
                                    type="button"
                                    onClick={handleResend}
                                    className="text-medic-dark hover:underline flex items-center gap-2 mx-auto"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    Resend OTP Code
                                </button>
                            )}
                        </div>
                    </div>
                </form>

                <button
                    onClick={() => {
                        dispatch(clearVerification());
                        navigate('/login');
                    }}
                    className="mt-12 text-xs font-bold text-gray-400 hover:text-gray-600 uppercase tracking-widest decoration-gray-300 underline-offset-8 underline"
                >
                    Cancel and go back
                </button>
            </motion.div>
        </div>
    );
};

export default OtpVerification;
