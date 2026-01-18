import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { verifyOTP } from '../store/slices/authSlice';
import { toast } from 'react-toastify';
import { Key, Loader2, RefreshCw } from 'lucide-react';
import { parseError } from '../utils/errorParser';

const OTPVerification = () => {
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const { loading, tempEmail, token, authType } = useSelector((state) => state.auth);
    const [otp, setOtp] = useState(['', '', '', '', '', '']);

    useEffect(() => {
        if (!tempEmail && !token) {
            navigate('/login');
        }
    }, [tempEmail, token, navigate]);

    const handleChange = (element, index) => {
        if (isNaN(element.value)) return false;

        setOtp([...otp.map((d, idx) => (idx === index ? element.value : d))]);

        // Focus next input
        if (element.nextSibling && element.value !== "") {
            element.nextSibling.focus();
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const otpString = otp.join('');
        if (otpString.length !== 6) {
            toast.error("Please enter all 6 digits");
            return;
        }

        const resultAction = await dispatch(verifyOTP({
            email: tempEmail,
            otp: otpString,
            type: authType || 'login'
        }));

        if (verifyOTP.fulfilled.match(resultAction)) {
            toast.success("Verification successful!");
            navigate('/dashboard');
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
                        <Key className="h-10 w-10 text-white" />
                    </div>
                </div>
                <h2 className="text-3xl font-extrabold text-gray-900">Verify your Email</h2>
                <p className="mt-2 text-sm text-gray-600 font-medium">
                    We've sent a 6-digit code to <span className="text-blue-600 font-bold">{tempEmail}</span>
                </p>
            </div>

            <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
                <div className="bg-white py-10 px-6 shadow-xl shadow-gray-200/50 rounded-3xl border border-gray-100 sm:px-10">
                    <form className="space-y-8" onSubmit={handleSubmit}>
                        <div className="flex justify-between gap-2">
                            {otp.map((data, index) => (
                                <input
                                    key={index}
                                    type="text"
                                    maxLength="1"
                                    className="w-12 h-14 text-center text-2xl font-bold bg-gray-50 border-2 border-gray-100 rounded-xl focus:border-blue-500 focus:bg-white focus:outline-none transition-all"
                                    value={data}
                                    onChange={(e) => handleChange(e.target, index)}
                                    onFocus={(e) => e.target.select()}
                                />
                            ))}
                        </div>

                        <div>
                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full flex justify-center py-4 px-4 border border-transparent rounded-2xl shadow-lg text-lg font-bold text-white bg-gradient-to-r from-blue-600 to-emerald-500 hover:from-blue-700 hover:to-emerald-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all disabled:opacity-50"
                            >
                                {loading ? <Loader2 className="animate-spin h-6 w-6" /> : "Verify Account"}
                            </button>
                        </div>
                    </form>

                    <div className="mt-8 text-center text-sm font-medium text-gray-500">
                        <p>Didn't receive the code?</p>
                        <button className="mt-2 text-blue-600 hover:text-blue-700 font-bold flex items-center justify-center gap-2 mx-auto">
                            <RefreshCw className="h-4 w-4" />
                            Resend Code
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default OTPVerification;
